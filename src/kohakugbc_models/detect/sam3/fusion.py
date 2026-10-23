"""SAM 3's text-image fusion under transformers' parameter names: the DETR encoder
(``detr_encoder.*``), the DETR decoder with box refinement, the presence token and the box
relative-position bias (``detr_decoder.*``), and the dot-product scoring
(``dot_product_scoring.*``). Text masks are bool ``[B, 1, 1, S]`` (True attends)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from kohakugbc_models.detect.sam3 import ops
from kohakugbc_models.detect.sam3.vision import Mlp as VisionMlp
from kohakugbc_models.detect.sam3.vision import linear, norm


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.q_proj, self.k_proj = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.v_proj, self.o_proj = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.ops = ops.EAGER

    def forward(self, query, key, value, mask=None, rpb=None) -> torch.Tensor:
        """``mask``: a bool key mask; ``rpb``: the separable box-bias tables (y, x)."""
        batch, q_len, k_len = query.shape[0], query.shape[1], key.shape[1]
        q = linear(self.q_proj, query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = linear(self.k_proj, key).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = linear(self.v_proj, value).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        if rpb is None:
            out = self.ops.attention(q, k, v, mask)
        else:
            out = self.ops.attention_rpb(q, k, v, *rpb)
        return linear(self.o_proj, out.transpose(1, 2).reshape(batch, q_len, -1))


class Mlp(VisionMlp):
    """The DETR MLP: ReLU between, in fp8 like the ViT's when quantized."""

    def __init__(self, hidden: int, inner: int):
        super().__init__(hidden, inner, act="relu")


def add_norm(layer_ops, x, r, ln: nn.LayerNorm) -> tuple:
    """``(x + r, ln(x + r))`` through the layer's ``add_layer_norm``."""
    return layer_ops.add_layer_norm(x, r, ln.weight, ln.bias, ln.eps)


class HeadMlp(nn.Module):
    """transformers' ``Sam3DecoderMLP``: 2 or 3 linears with ReLU between."""

    def __init__(self, inp: int, hidden: int, out: int, layers: int):
        super().__init__()
        self.layer1 = nn.Linear(inp, hidden)
        self.layer2 = nn.Linear(hidden, hidden if layers == 3 else out)
        self.layer3 = nn.Linear(hidden, out) if layers == 3 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(linear(self.layer1, x))
        if self.layer3 is None:
            return linear(self.layer2, x)
        return linear(self.layer3, F.relu(linear(self.layer2, x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = Attention(hidden, cfg.num_attention_heads)
        self.cross_attn = Attention(hidden, cfg.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = Mlp(hidden, cfg.intermediate_size)
        self.layer_norm3 = nn.LayerNorm(hidden)
        self.ops = ops.EAGER
        self.recipe: str | None = None

    def quantize(self, recipe: str, device) -> None:
        """The MLP in fp8: its input norm emits fp8 (``add_ln_quant``), its ReLU too."""
        self.recipe = recipe
        self.mlp.quantize(recipe, device)

    def forward(self, x, h, text, pos, text_mask, next_norm) -> tuple:
        """The stream ``x`` and its first norm ``h`` in; the stream out with ``next_norm`` of
        it (the next layer's first norm, fused with the residual add; None after the last)."""
        with_pos = h + pos
        x, h = add_norm(self.ops, x, self.self_attn(with_pos, with_pos, h), self.layer_norm2)
        a = self.cross_attn(h, text, text, text_mask)
        if self.recipe is None:
            x, h = add_norm(self.ops, x, a, self.layer_norm3)
        else:
            ln = self.layer_norm3
            x, q, scales = self.ops.add_ln_quant(x, a, ln.weight, ln.bias, ln.eps, self.recipe)
            h = (q, scales, x.shape)
        m = self.mlp(h)
        if next_norm is None:
            return x + m, None
        return add_norm(self.ops, x, m, next_norm)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))

    def quantize(self, recipe: str, device) -> None:
        for layer in self.layers:
            layer.quantize(recipe, device)

    def forward(self, vision, pos, text, text_mask) -> torch.Tensor:
        """``vision`` / ``pos`` ``[B, C, h, w]``, ``text`` ``[B, S, C]``: fused ``[B, h*w, C]``."""
        # Row-major once: elementwise ops would carry the transposed layout through the stream
        x = vision.flatten(2).transpose(1, 2).contiguous()
        pos = pos.flatten(2).transpose(1, 2)
        h = norm(self.layers[0].layer_norm1, x)
        for i, layer in enumerate(self.layers):
            following = self.layers[i + 1].layer_norm1 if i + 1 < len(self.layers) else None
            x, h = layer(x, h, text, pos, text_mask, following)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden, heads = cfg.hidden_size, cfg.num_attention_heads
        self.self_attn = Attention(hidden, heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        self.text_cross_attn = Attention(hidden, heads)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden)
        self.vision_cross_attn = Attention(hidden, heads)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden)
        self.mlp = Mlp(hidden, cfg.intermediate_size)
        self.mlp_layer_norm = nn.LayerNorm(hidden)
        self.ops = ops.EAGER

    def forward(self, x, query_pos, text, text_mask, vision, vision_with_pos, rpb):
        query_pos = F.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)
        q = x + query_pos
        _, x = add_norm(self.ops, x, self.self_attn(q, q, x), self.self_attn_layer_norm)
        a = self.text_cross_attn(x + query_pos, text, text, text_mask)
        _, x = add_norm(self.ops, x, a, self.text_cross_attn_layer_norm)
        a = self.vision_cross_attn(x + query_pos, vision_with_pos, vision, rpb=rpb)
        _, x = add_norm(self.ops, x, a, self.vision_cross_attn_layer_norm)
        return add_norm(self.ops, x, self.mlp(x), self.mlp_layer_norm)[1]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_frequencies(features: int, dtype: torch.dtype, device) -> torch.Tensor:
    """The divisors of ``encode_boxes`` in ``dtype``, as transformers computes them per call."""
    dim_t = torch.arange(features, dtype=torch.int64, device=device).to(dtype)
    return 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / features)


def encode_boxes(boxes: torch.Tensor, dim_t: torch.Tensor) -> torch.Tensor:
    """transformers' ``Sam3SinePositionEmbedding.encode_boxes`` (temperature 10000, 2 pi)
    with its divisors ``dim_t`` precomputed (``box_frequencies``)."""
    parts = []
    for i in (1, 0, 2, 3):  # y, x, w, h
        p = (boxes[:, :, i] * (2 * math.pi))[:, :, None] / dim_t
        parts.append(torch.stack((p[:, :, 0::2].sin(), p[:, :, 1::2].cos()), dim=3).flatten(2))
    return torch.cat(parts, dim=2)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden, heads = cfg.hidden_size, cfg.num_attention_heads
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.output_layer_norm = nn.LayerNorm(hidden)
        self.box_head = HeadMlp(hidden, hidden, 4, 3)
        self.query_embed = nn.Embedding(cfg.num_queries, hidden)
        self.reference_points = nn.Embedding(cfg.num_queries, 4)
        self.presence_token = nn.Embedding(1, hidden)
        self.presence_head = HeadMlp(hidden, hidden, 1, 3)
        self.presence_layer_norm = nn.LayerNorm(hidden)
        self.clamp_presence = 10.0
        self.ref_point_head = HeadMlp(2 * hidden, hidden, hidden, 2)
        self.box_rpb_embed_x = HeadMlp(2, hidden, heads, 2)
        self.box_rpb_embed_y = HeadMlp(2, hidden, heads, 2)
        self.hidden = hidden
        self.dim_t = None
        self.ops = ops.EAGER

    def frequencies(self, dtype: torch.dtype, device) -> None:
        """Precompute ``encode_boxes``' divisors eagerly (Inductor cannot build their bf16
        floor division: ``isinf`` on bf16), as a plain attribute a dtype cast never touches."""
        self.dim_t = box_frequencies(self.hidden // 2, dtype, device)

    def rpb(self, boxes: torch.Tensor, height: int, width: int) -> tuple:
        """The box relative-position bias as its separable tables ``[B, heads, 1 + Q, height]``
        and ``[B, heads, 1 + Q, width]`` (log-scale deltas; row 0, the presence token, zero):
        key ``(r, c)`` gets ``y[..., r] + x[..., c]``."""
        params = []
        for mlp in (self.box_rpb_embed_y, self.box_rpb_embed_x):
            l1, l2 = mlp.layer1, mlp.layer2
            params += [l1.weight, l1.bias, l2.weight, l2.bias]
        return self.ops.box_bias(boxes, *params, height, width)

    def forward(self, vision, pos, text, text_mask, height: int, width: int) -> tuple:
        """Fused vision ``[B, h*w, C]``, its positions, text ``[B, S, C]``: the last layer's
        normalized queries ``[B, Q, C]``, boxes ``[B, Q, 4]`` (xyxy, 0..1) and presence logits
        ``[B, 1]``."""
        batch = vision.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        boxes = self.reference_points.weight.unsqueeze(0).expand(batch, -1, -1).sigmoid()
        presence = self.presence_token.weight.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([presence, queries], dim=1)
        vision_with_pos = vision + pos
        for layer in self.layers:
            query_pos = self.ref_point_head(encode_boxes(boxes, self.dim_t))
            x = layer(
                x,
                query_pos,
                text,
                text_mask,
                vision,
                vision_with_pos,
                self.rpb(boxes, height, width),
            )
            out = norm(self.output_layer_norm, x[:, 1:])
            boxes = (inverse_sigmoid(boxes) + self.box_head(out)).sigmoid()
        presence = self.presence_head(norm(self.presence_layer_norm, x[:, :1])).squeeze(-1)
        presence = presence.clamp(min=-self.clamp_presence, max=self.clamp_presence)
        return out, cxcywh_to_xyxy(boxes), presence


class Scoring(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.text_mlp = HeadMlp(hidden, inner, hidden, 2)
        self.text_mlp_out_norm = nn.LayerNorm(hidden)
        self.text_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.scale = 1.0 / math.sqrt(hidden)
        self.clamp = 12.0

    def forward(self, queries, text, valid) -> torch.Tensor:
        """Logits ``[B, Q]`` of normalized queries ``[B, Q, C]`` against text ``[B, S, C]``
        mean-pooled over ``valid`` ``[B, S]`` tokens."""
        text = norm(self.text_mlp_out_norm, self.text_mlp(text) + text)
        is_valid = valid.to(text.dtype).unsqueeze(-1)
        pooled = (text * is_valid).sum(dim=1) / is_valid.sum(dim=1).clamp(min=1.0)
        proj_text = linear(self.text_proj, pooled).unsqueeze(-1)
        scores = torch.matmul(linear(self.query_proj, queries), proj_text) * self.scale
        return scores.clamp(min=-self.clamp, max=self.clamp).squeeze(-1)
