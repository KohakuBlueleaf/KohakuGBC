"""SAM 3's vision encoder: the ViT (32 layers, windowed and global attention with 2D axial
RoPE) and the FPN neck, under transformers' parameter names (``vision_encoder.*``).

The neck builds only the levels asked for: the detector reads level 2 (72^2, scale 1.0); the
mask decoder also levels 0 and 1 (288^2, 144^2). Level 3 (scale 0.5) is never read."""

import math

import torch
from torch import nn

from kohakugbc_models.detect.sam3 import ops
from kohakugbc_models.kernels import fp8


def linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return ops.linear(x, module.weight, module.bias)


def norm(module: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    return ops.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


def drop(*linears: nn.Linear) -> None:
    """Free the 16-bit weights of linears that now run from their fp8 copies (biases kept)."""
    for module in linears:
        module.weight = nn.Parameter(module.weight.new_empty(0), requires_grad=False)


def inputs(module: nn.Module, device, *names: str) -> None:
    """``module``'s NVFP4 input scales (``fp8.Fp4Input``) under ``names``, made once, on
    ``device``."""
    for name in names:
        site = getattr(module, name)
        setattr(module, name, (fp8.Fp4Input() if site is None else site).to(device))


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.q_proj, self.k_proj = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.v_proj, self.o_proj = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.ops = ops.EAGER
        self.qkv_weight = self.qkv_bias = None
        self.fp8: tuple | None = None
        self.o_in: fp8.Fp4Input | None = None

    @torch.no_grad()
    def quantize(self, recipe: str, device) -> None:
        """The fused q / k / v and the output projection's weights in fp8 (``recipe``), the
        16-bit weights then dropped; an fp8 layer moves its fp8 weights to ``device``."""
        inputs(self, device, "o_in")
        if self.fp8 is not None:
            self.fp8 = (
                self.fp8[0],
                {k: tuple(t.to(device) for t in v) for k, v in self.fp8[1].items()},
            )
            return
        self.fp8 = (
            recipe,
            {
                "qkv": fp8.quantize_weight(self.qkv_weight, recipe),
                "o": fp8.quantize_weight(self.o_proj.weight, recipe),
            },
        )
        drop(self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        self.qkv_weight = None

    def forward_fp8(self, x: tuple, cos, sin, window: int) -> torch.Tensor:
        """``x`` the quantized first norm ``(q, scales, [B, S, S, C])``: the fp8 q / k / v
        GEMM, attention with RoPE on the unrotated q / k, the output quantized for the fp8
        output projection."""
        recipe, weights = self.fp8
        xq, xs, (batch, size, _, channels) = x
        qkv = fp8.mm(xq, xs, *weights["qkv"], recipe, self.qkv_bias)
        q, k, v = qkv.view(batch, size, size, 3, self.heads, self.head_dim).unbind(3)
        out = self.ops.vit_attention(q, k, v, cos, sin, window)
        oq, os = self.ops.quant(out.reshape(batch, size, size, channels), recipe, self.o_in)
        out = fp8.mm(oq, os, *weights["o"], recipe, self.o_proj.bias)
        return out.view(batch, size, size, channels)

    @torch.no_grad()
    def fuse_qkv(self) -> None:
        """One ``[3C, C]`` weight (and bias) for q, k, v; the three projections become views
        of it, so the fused GEMM costs no extra memory."""
        weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        bias = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])
        for i, proj in enumerate((self.q_proj, self.k_proj, self.v_proj)):
            rows = slice(i * weight.shape[1], (i + 1) * weight.shape[1])
            proj.weight = nn.Parameter(weight[rows], requires_grad=False)
            proj.bias = nn.Parameter(bias[rows], requires_grad=False)
        self.qkv_weight, self.qkv_bias = weight, bias

    def forward(self, x, cos, sin, window: int) -> torch.Tensor:
        """Grid features ``[B, S, S, C]``: attention within ``window`` windows (``S``: global),
        the output in the same layout (the window partition is the op's addressing)."""
        if self.fp8 is not None:
            return self.forward_fp8(x, cos, sin, window)
        batch, size, _, _ = x.shape
        q, k, v = self.ops.qkv_rope(x, self.qkv_weight, self.qkv_bias, cos, sin, window)
        out = self.ops.vit_attention(q, k, v, cos, sin, window, rotated=True)
        return linear(self.o_proj, out.reshape(batch, size, size, -1))


class Mlp(nn.Module):
    def __init__(self, hidden: int, inner: int, act: str = "gelu"):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(hidden, inner), nn.Linear(inner, hidden)
        self.act = act
        self.ops = ops.EAGER
        self.fp8: tuple | None = None
        self.fc2_in: fp8.Fp4Input | None = None

    @torch.no_grad()
    def quantize(self, recipe: str, device) -> None:
        inputs(self, device, "fc2_in")
        if self.fp8 is not None:
            self.fp8 = (
                self.fp8[0],
                {k: tuple(t.to(device) for t in v) for k, v in self.fp8[1].items()},
            )
            return
        self.fp8 = (
            recipe,
            {
                "fc1": fp8.quantize_weight(self.fc1.weight, recipe),
                "fc2": fp8.quantize_weight(self.fc2.weight, recipe),
            },
        )
        drop(self.fc1, self.fc2)

    def forward(self, x) -> torch.Tensor:
        """``x`` the input norm: a tensor, or quantized ``(q, scales, shape)`` under fp8 (the
        activation then emits fc2's fp8 input)."""
        if self.fp8 is not None:
            recipe, weights = self.fp8
            xq, xs, shape = x
            h = fp8.mm(xq, xs, *weights["fc1"], recipe, self.fc1.bias)
            hq, hs = self.ops.act_quant(h, self.act, recipe, self.fc2_in)
            return fp8.mm(hq, hs, *weights["fc2"], recipe, self.fc2.bias).view(shape)
        h = self.ops.linear_act(x, self.fc1.weight, self.fc1.bias, self.act)
        return linear(self.fc2, h)


def rope_tables(grid: int, window: int, head_dim: int, theta: float, device) -> tuple:
    """cos / sin ``[1, grid^2, head_dim]`` of SAM 3's axial RoPE on a ``grid x grid`` layout
    whose positions are scaled to the window size (transformers' ``precompute_positions`` and
    ``Sam3ViTRotaryEmbedding``). The angles' cos / sin are taken on ``device``, where
    transformers takes them: a host cos / sin differs from the device's in the last fp32 bit."""
    h, w = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    positions = torch.stack([w.flatten(), h.flatten()], dim=-1) * (window / grid)
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))
    freqs = positions[..., None].float().to(device) * inv_freq.to(device)
    tables = []
    for f in (freqs.cos(), freqs.sin()):
        f = torch.cat([f[:, 0], f[:, 1]], dim=-1)[None, ...]
        tables.append(f.repeat_interleave(2, dim=-1))
    return tuple(tables)


class Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, inner: int, eps: float, window: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.attention = Attention(hidden, heads)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = Mlp(hidden, inner)
        self.window = window
        self.ops = ops.EAGER
        self.recipe: str | None = None
        self.mlp_recipe: str | None = None
        self.qkv_in: fp8.Fp4Input | None = None
        self.fc1_in: fp8.Fp4Input | None = None

    def quantize(self, recipe: str, device, mlp: str | None = None) -> None:
        """The attention's linears in ``recipe``, the MLP's in ``mlp`` (``recipe`` if None);
        ``qkv_in`` / ``fc1_in``: the NVFP4 scales of the norms feeding q / k / v and fc1
        (``qkv_in`` written by the layer before)."""
        self.recipe, self.mlp_recipe = recipe, mlp or recipe
        inputs(self, device, "qkv_in", "fc1_in")
        self.attention.quantize(recipe, device)
        self.mlp.quantize(self.mlp_recipe, device)

    def add_norm(self, x, r, ln: nn.LayerNorm, site, recipe: str | None) -> tuple:
        """``(x + r, ln(x + r))``; with a ``recipe`` the norm quantized: ``(q, scales, shape)``."""
        if recipe is None:
            return self.ops.add_layer_norm(x, r, ln.weight, ln.bias, ln.eps)
        s, q, scales = self.ops.add_ln_quant(x, r, ln.weight, ln.bias, ln.eps, recipe, site)
        return s, (q, scales, s.shape)

    def forward(self, x, h, cos, sin, following) -> tuple:
        """The residual stream ``x`` and its first norm ``h`` in; the stream out with the
        ``following`` layer's first norm of it (fused with the residual add; None after the
        last layer)."""
        a = self.attention(h, cos, sin, self.window or x.shape[1])
        x, h = self.add_norm(x, a, self.layer_norm2, self.fc1_in, self.mlp_recipe)
        m = self.mlp(h)
        if following is None:
            return x + m, None
        return self.add_norm(x, m, following.layer_norm1, following.qkv_in, following.recipe)


class PatchEmbeddings(nn.Module):
    def __init__(self, channels: int, hidden: int, patch: int):
        super().__init__()
        self.projection = nn.Conv2d(channels, hidden, kernel_size=patch, stride=patch, bias=False)


class Embeddings(nn.Module):
    def __init__(self, channels: int, hidden: int, patch: int, pretrain_grid: int):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(channels, hidden, patch)
        self.position_embeddings = nn.Parameter(torch.zeros(1, pretrain_grid**2, hidden))

    def forward(self, pixels: torch.Tensor, grid: int) -> torch.Tensor:
        conv = self.patch_embeddings.projection
        x = conv(pixels.to(conv.weight.dtype)).flatten(2).transpose(1, 2)
        pretrain = int(self.position_embeddings.shape[1] ** 0.5)
        hidden = self.position_embeddings.shape[-1]
        if pretrain == grid:
            pos = self.position_embeddings.reshape(1, grid * grid, -1)
        else:
            pos = self.position_embeddings.reshape(1, pretrain, pretrain, hidden).permute(
                0, 3, 1, 2
            )
            repeat = grid // pretrain + 1
            pos = pos.tile([1, 1, repeat, repeat])[:, :, :grid, :grid]
            pos = pos.permute(0, 2, 3, 1).reshape(1, grid * grid, hidden)
        return x + pos


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch = cfg.patch_size
        self.window = cfg.window_size
        self.embeddings = Embeddings(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            cfg.pretrain_image_size // cfg.patch_size,
        )
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList(
            Layer(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.intermediate_size,
                cfg.layer_norm_eps,
                0 if i in cfg.global_attn_indexes else cfg.window_size,
            )
            for i in range(cfg.num_hidden_layers)
        )
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.theta = cfg.rope_parameters["rope_theta"]
        self.grid = cfg.image_size // cfg.patch_size
        self.tables: dict = {}

    def rope(self, device) -> None:
        """The RoPE tables in fp32 on ``device`` (plain attributes: a dtype cast of the module
        never reaches them, as transformers keeps them fp32)."""
        self.tables = {
            name: rope_tables(size, self.window, self.head_dim, self.theta, device)
            for name, size in (("window", self.window), ("global", self.grid))
        }

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, 3, S, S]`` pixels to ``[B, S / patch, S / patch, hidden]`` features."""
        grid = pixels.shape[-1] // self.patch
        x = self.embeddings(pixels, grid)
        x = norm(self.layer_norm, x.view(pixels.shape[0], grid, grid, -1))
        h = norm(self.layers[0].layer_norm1, x)
        first = self.layers[0]
        if first.recipe is not None:
            h = (*first.ops.quant(h, first.recipe, first.qkv_in), h.shape)
        for i, layer in enumerate(self.layers):
            following = self.layers[i + 1] if i + 1 < len(self.layers) else None
            x, h = layer(x, h, *self.tables["window" if layer.window else "global"], following)
        return x

    def fuse_qkv(self) -> None:
        for layer in self.layers:
            if layer.recipe is None:
                layer.attention.fuse_qkv()

    def quantize(self, recipe: str, device, mlp: str | None = None) -> None:
        """Every layer's linears in fp8 (``recipe``; the MLPs' in ``mlp`` when given; the
        16-bit weights freed); their activations then quantized by the norms and the GELU that
        produce them. Already in fp8, the fp8 weights move to ``device``."""
        for layer in self.layers:
            layer.quantize(recipe, device, mlp)


class FpnLayer(nn.Module):
    def __init__(self, channels: int, fpn: int, scale: float):
        super().__init__()
        self.scale_layers = nn.ModuleList()
        if scale == 4.0:
            self.scale_layers.append(nn.ConvTranspose2d(channels, channels // 2, 2, stride=2))
            self.scale_layers.append(nn.GELU())
            self.scale_layers.append(nn.ConvTranspose2d(channels // 2, channels // 4, 2, stride=2))
            inner = channels // 4
        elif scale == 2.0:
            self.scale_layers.append(nn.ConvTranspose2d(channels, channels // 2, 2, stride=2))
            inner = channels // 2
        elif scale == 1.0:
            inner = channels
        elif scale == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            inner = channels
        else:
            raise NotImplementedError(f"FPN scale {scale}")
        self.proj1 = nn.Conv2d(inner, fpn, kernel_size=1)
        self.proj2 = nn.Conv2d(fpn, fpn, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.proj1.weight.dtype)
        for layer in self.scale_layers:
            x = layer(x)
        return self.proj2(self.proj1(x))


def sine_positions(
    batch: int, channels: int, height: int, width: int, device, dtype
) -> torch.Tensor:
    """The neck's normalized sine position encoding ``[B, channels, height, width]``
    (transformers' ``build_sine_position_embedding`` with ``normalize=True``, no mask)."""
    features, temperature, scale, eps = channels // 2, 10000, 2 * math.pi, 1e-6
    y = torch.arange(1, height + 1, dtype=dtype, device=device)[None, :, None].expand(
        batch, height, width
    )
    x = torch.arange(1, width + 1, dtype=dtype, device=device)[None, None, :].expand(
        batch, height, width
    )
    y = y / (y[:, -1:, :] + eps) * scale
    x = x / (x[:, :, -1:] + eps) * scale
    dim_t = torch.arange(features, dtype=torch.int64, device=device).to(dtype)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / features)
    pos_x = x[:, :, :, None] / dim_t
    pos_y = y[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class Neck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.backbone_config.hidden_size
        self.fpn, self.scales = cfg.fpn_hidden_size, list(cfg.scale_factors)
        self.fpn_layers = nn.ModuleList(
            FpnLayer(channels, cfg.fpn_hidden_size, s) for s in cfg.scale_factors
        )
        self.tables: dict[int, torch.Tensor] = {}

    def positions(self, grid: int, device, dtype) -> None:
        """Each level's position encoding ``[1, C, h, w]`` for a ``grid x grid`` backbone, made
        once, eagerly (it depends on the shape only; Inductor cannot build its bf16 floor
        division: ``isinf`` on bf16)."""
        self.tables = {
            level: sine_positions(1, self.fpn, int(grid * s), int(grid * s), device, dtype)
            for level, s in enumerate(self.scales)
        }

    def forward(self, x: torch.Tensor, levels) -> dict:
        """``[B, h, w, C]`` backbone features to ``{level: (features, positions)}``."""
        x = x.permute(0, 3, 1, 2)
        out = {}
        for level in levels:
            feats = self.fpn_layers[level](x)
            out[level] = (feats, self.tables[level].expand(feats.shape[0], -1, -1, -1))
        return out


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg.backbone_config)
        self.neck = Neck(cfg)

    def forward(self, pixels: torch.Tensor, levels) -> dict:
        return self.neck(self.backbone(pixels), levels)
