"""The Qwen3.5 forward over a step of packed sequences, on the checkpoint's own weights (loaded
through transformers' ``Qwen3_5ForConditionalGeneration``; its vision tower runs as it is).

A step is any mix of sequences, each adding one or more tokens after what its pool slot and
pages already hold; the forward returns the logits after each sequence's last new token. The
norms, rotary embedding and gated norm round where the reference model rounds.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import Qwen3_5ForConditionalGeneration

from qwen35.model.shape import Shape
from qwen35.ops import Ops


@dataclass
class Step:
    """One forward's inputs, on the model's device."""

    ids: torch.Tensor  # [N] token ids
    positions: torch.Tensor  # [3, N] mrope positions
    cu: torch.Tensor  # [B + 1] int32 sequence boundaries in the packed tokens
    slots: torch.Tensor  # [B] int32 state pool slots
    pages: torch.Tensor  # [B, max_pages] int32 page tables
    ctx: torch.Tensor  # [B] int32 cached tokens after this step
    write: torch.Tensor  # [N] int64 flat cache rows of the new tokens
    last: torch.Tensor  # [B] int64 index of each sequence's last new token
    max_len: int  # longest new-token run
    image_rows: torch.Tensor | None = None  # [M] rows of image tokens in the packed tokens
    image_embeds: torch.Tensor | None = None  # [M, hidden]


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero-centred RMSNorm: ``x / rms(x) * (1 + weight)`` in fp32, rounded once."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(x.dtype)


def gated_rms(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (weight * y * F.silu(z.float())).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Model:
    """The checkpoint at ``path`` on ``device``."""

    def __init__(self, path: str, device: str = "cuda:0", dtype=torch.bfloat16):
        """``path``: a checkpoint folder, or a Hugging Face model name (downloaded once)."""
        if not Path(path).is_dir():
            path = snapshot_download(path)
        self.path, self.device, self.dtype = path, torch.device(device), dtype
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        self.shape = Shape.from_config(config)
        self.hf = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype)
        self.hf = self.hf.to(self.device).eval()
        lm = self.hf.model.language_model
        self.embed = lm.embed_tokens.weight
        self.layers = lm.layers
        self.final_norm = lm.norm.weight
        self.visual = self.hf.model.visual
        self.conv_weights = {
            i: self.layers[i].linear_attn.conv1d.weight.squeeze(1).contiguous()
            for i in self.shape.delta_layers
        }
        rd = self.shape.rotary_dim
        exponents = torch.arange(0, rd, 2, dtype=torch.float32, device=self.device) / rd
        self.inv_freq = 1.0 / (self.shape.rope_theta**exponents)
        # Pool index of each layer among its kind
        self.delta_index = {layer: j for j, layer in enumerate(self.shape.delta_layers)}
        self.attention_index = {layer: j for j, layer in enumerate(self.shape.attention_layers)}

    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.hf.parameters())

    # ---- vision ----------------------------------------------------------------------------

    @torch.no_grad()
    def vision(self, pixels: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Merged image embeddings, one row per image token, every image of the batch at once."""
        pixels = pixels.to(self.device, self.visual.dtype)
        return self.visual(pixels, grid_thw=grid.to(self.device)).pooler_output.to(self.dtype)

    # ---- the text model --------------------------------------------------------------------

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleaved mrope: frequency ``i`` takes its position from axis T, H or W by ``i % 3``
        within the H and W sections."""
        freqs = positions[..., None].float() * self.inv_freq
        mixed = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.shape.mrope_section[axis] * 3
            mixed[..., offset:length:3] = freqs[axis, ..., offset:length:3]
        emb = torch.cat((mixed, mixed), dim=-1)
        return emb.cos().to(self.dtype)[:, None], emb.sin().to(self.dtype)[:, None]

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rd = cos.shape[-1]
        rot = x[..., :rd]
        return torch.cat((rot * cos + rotate_half(rot) * sin, x[..., rd:]), dim=-1)

    @torch.no_grad()
    def forward(self, ops: Ops, pools, step: Step) -> torch.Tensor:
        """Float32 logits ``[B, vocab]`` after each sequence's last new token."""
        sh = self.shape
        h = F.embedding(step.ids, self.embed)
        if step.image_rows is not None:
            h = h.index_copy(0, step.image_rows, step.image_embeds)
        cos, sin = self.rope(step.positions)
        for i, layer in enumerate(self.layers):
            x = rms(h, layer.input_layernorm.weight, sh.eps)
            if i in self.delta_index:
                x = self._delta(ops, pools, step, i, layer.linear_attn, x)
            else:
                x = self._attention(ops, pools, step, i, layer.self_attn, x, cos, sin)
            h = h + x
            x = rms(h, layer.post_attention_layernorm.weight, sh.eps)
            mlp = layer.mlp
            x = F.silu(F.linear(x, mlp.gate_proj.weight)) * F.linear(x, mlp.up_proj.weight)
            h = h + F.linear(x, mlp.down_proj.weight)
        last = rms(h[step.last], self.final_norm, sh.eps)
        return F.linear(last, self.embed).float()

    def _delta(self, ops, pools, step, i, la, x):
        sh = self.shape
        n = x.shape[0]
        j = self.delta_index[i]
        qkv = F.linear(x, la.in_proj_qkv.weight)
        z = F.linear(x, la.in_proj_z.weight).view(n, sh.value_heads, sh.value_dim)
        b = F.linear(x, la.in_proj_b.weight)
        a = F.linear(x, la.in_proj_a.weight)
        qkv = ops("conv", qkv, self.conv_weights[i], pools.conv[j], step.slots, step.cu)
        kd = sh.key_heads * sh.key_dim
        q = qkv[:, :kd].reshape(n, sh.key_heads, sh.key_dim).contiguous()
        k = qkv[:, kd : 2 * kd].reshape(n, sh.key_heads, sh.key_dim).contiguous()
        v = qkv[:, 2 * kd :].reshape(n, sh.value_heads, sh.value_dim).contiguous()
        o = ops(
            "delta", q, k, v, a, b, la.A_log, la.dt_bias,
            pools.state[j], step.slots, step.cu, step.max_len,
        )  # fmt: skip
        o = gated_rms(o, z, la.norm.weight, sh.eps)
        return F.linear(o.reshape(n, -1), la.out_proj.weight)

    def _attention(self, ops, pools, step, i, at, x, cos, sin):
        sh = self.shape
        n, hd = x.shape[0], sh.head_dim
        j = self.attention_index[i]
        qg = F.linear(x, at.q_proj.weight).view(n, sh.heads, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(n, sh.heads * hd)
        q = rms(q, at.q_norm.weight, sh.eps)
        k = rms(F.linear(x, at.k_proj.weight).view(n, sh.kv_heads, hd), at.k_norm.weight, sh.eps)
        v = F.linear(x, at.v_proj.weight).view(n, sh.kv_heads, hd)
        q = self.apply_rope(q, cos, sin).contiguous()
        k = self.apply_rope(k, cos, sin).contiguous()
        o = ops(
            "attention", q, k, v.contiguous(), pools.keys[j], pools.values[j],
            step.pages, step.ctx, step.cu, step.write, step.max_len, hd**-0.5,
        )  # fmt: skip
        o = o.reshape(n, sh.heads * hd) * torch.sigmoid(gate)
        return F.linear(o, at.o_proj.weight)
