"""EmbeddingGemma (``google/embeddinggemma-300m``: a bidirectional Gemma 3 text encoder, mean
pooling, two dense layers, L2 normalization) over one packed token stream.

A batch of texts is one stream of their tokens laid end to end (no padding); each token carries
its text's span, and attention keeps each token within its own text (and, on the sliding
layers, within the window). With ``graphs`` the pass from token ids to normalized vectors is
replayed from one CUDA graph per stream bucket (the next power of two, at least ``MIN_TOKENS``;
``max_batch`` texts per pass).

Numerics follow transformers' Gemma 3 in the model's dtype: RMS norms in fp32 rounded to the
dtype, cos / sin rounded to the dtype, the embedding scale in the dtype. The pooled mean is summed
in fp32 and the dense layers and the normalization run in fp32 (sentence-transformers runs them
in the model's dtype). ``residual``: the dtype of the residual stream (``float32``: the stream,
its adds and the norms reading it stay fp32, only GEMM / attention operands are in the model's
dtype; ``model``: transformers' casts exactly). The stream reaches |x| ~ 6e4 by the last layer
(a bf16 step of 256 there): in bf16 a text's vector can land at 1 - cos ~ 4e-3 from fp64.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.eager import varlen as eager_varlen
from kohakugbc_models.kernels.graphs import Graphs

FILES = ["*.json", "*.safetensors", "tokenizer*", "*/config.json", "*/model.safetensors"]
OPS = ("varlen_attention", "segment_mean")
EAGER = {name: getattr(eager_varlen, name) for name in OPS}
MIN_TOKENS = 64


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, dtype=None) -> torch.Tensor:
    """Gemma's RMS norm in fp32, rounded to ``dtype`` (``x``'s by default)."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(dtype or x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Norm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, dtype=None) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps, dtype)


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, self.dim = cfg["hidden_size"], cfg["head_dim"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.scale = cfg["query_pre_attn_scalar"] ** -0.5
        self.q_proj = nn.Linear(hidden, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dim, hidden, bias=False)
        self.q_norm = Norm(self.dim, cfg["rms_norm_eps"])
        self.k_norm = Norm(self.dim, cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, start, end, window: int, ops: dict) -> torch.Tensor:
        total = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(total, self.heads, self.dim))
        k = self.k_norm(self.k_proj(x).view(total, self.kv_heads, self.dim))
        v = self.v_proj(x).view(total, self.kv_heads, self.dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = ops["varlen_attention"](q, k, v, start, end, window, self.scale)
        return self.o_proj(out.reshape(total, -1))


class Mlp(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, inner = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: dict, kind: str):
        super().__init__()
        eps = cfg["rms_norm_eps"]
        self.kind = kind
        self.window = cfg["sliding_window"] if kind == "sliding_attention" else 0
        self.input_layernorm = Norm(cfg["hidden_size"], eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = Norm(cfg["hidden_size"], eps)
        self.pre_feedforward_layernorm = Norm(cfg["hidden_size"], eps)
        self.mlp = Mlp(cfg)
        self.post_feedforward_layernorm = Norm(cfg["hidden_size"], eps)

    def forward(self, x, cos, sin, start, end, ops: dict) -> torch.Tensor:
        """The stream ``x`` in its own dtype; GEMM / attention operands in the weights'."""
        dtype = self.mlp.up_proj.weight.dtype
        h = self.input_layernorm(x, dtype)
        h = self.self_attn(h, cos, sin, start, end, self.window, ops)
        x = x + self.post_attention_layernorm(h, x.dtype)
        h = self.mlp(self.pre_feedforward_layernorm(x, dtype))
        return x + self.post_feedforward_layernorm(h, x.dtype)


class Encoder(nn.Module):
    """The Gemma 3 text encoder under the checkpoint's own tensor names."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.norm = Norm(cfg["hidden_size"], cfg["rms_norm_eps"])


def rope_table(base: float, dim: int, positions: int, device, dtype) -> tuple:
    """cos / sin ``[positions, dim]`` in fp32, rounded to ``dtype`` (transformers' casts)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = torch.arange(positions, dtype=torch.float)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1).to(device)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def bucket(total: int) -> int:
    return max(MIN_TOKENS, 1 << max(0, (total - 1).bit_length()))


class EmbeddingGemma:
    """Texts to normalized fp32 vectors. ``kernels``: the kernel registry's varlen attention and
    segment mean (else the eager references); ``graphs``: CUDA-graph replay per stream bucket;
    ``residual``: ``float32`` or ``model`` (the module docstring)."""

    def __init__(
        self,
        model: str = "google/embeddinggemma-300m",
        max_batch: int = 64,
        max_tokens: int = 2048,
        kernels: bool = True,
        graphs: bool = True,
        residual: str = "float32",
    ):
        if residual not in ("float32", "model"):
            raise ValueError(f"residual {residual!r}: float32 or model")
        self.model_name, self.max_batch, self.max_tokens = model, max_batch, max_tokens
        self.kernels, self.use_graphs, self.residual = kernels, graphs, residual
        self.encoder = self.tokenizer = self.graphs = self.cfg = None
        self.dense: list[torch.Tensor] = []
        self.tables: dict[str, tuple] = {}
        self.ops = dict(EAGER)
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        path = Path(self.model_name)
        if not path.is_dir():
            path = Path(snapshot_download(self.model_name, allow_patterns=FILES))
        self.cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        encoder = Encoder(self.cfg)
        own = encoder.state_dict()
        found = set()
        with safe_open(str(path / "model.safetensors"), "pt") as f:
            for key in f.keys():  # noqa: SIM118 (a safetensors handle is not iterable)
                if key in own:
                    own[key].copy_(f.get_tensor(key))
                    found.add(key)
        missing = sorted(set(own) - found)
        if missing:
            raise ValueError(f"checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")
        modules = json.loads((path / "modules.json").read_text(encoding="utf-8"))
        self.dense = []
        for module in modules:
            if module["type"].endswith("Dense"):
                with safe_open(str(path / module["path"] / "model.safetensors"), "pt") as f:
                    self.dense.append(f.get_tensor("linear.weight").float())
        self.encoder = encoder.to(dtype=dtype).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.dtype = dtype
        self.to(device)

    def to(self, device) -> None:
        """Place the model on ``device``: the RoPE tables, the bound ops and the graphs are
        rebuilt there."""
        self.encoder.to(device)
        self.dense = [w.to(device) for w in self.dense]
        cfg, positions = self.cfg, self.cfg["max_position_embeddings"]
        bases = {
            "sliding_attention": cfg["rope_local_base_freq"],
            "full_attention": cfg["rope_theta"],
        }
        self.tables = {
            kind: rope_table(base, cfg["head_dim"], positions, device, self.dtype)
            for kind, base in bases.items()
        }
        self.scale = torch.tensor(cfg["hidden_size"] ** 0.5, device=device).to(self.dtype)
        self.ops = dict(EAGER)
        if self.kernels:
            self.ops = {name: registry.resolve(name, device, self.dtype) for name in OPS}
        self.device = device
        self.graphs = None
        if self.use_graphs and torch.device(device).type == "cuda":
            self.graphs = Graphs(self.forward, clone=(0,))

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Token ids per text (host), as sentence-transformers tokenizes (special tokens,
        truncated to ``max_tokens``)."""
        return self.tokenizer(texts, truncation=True, max_length=self.max_tokens)["input_ids"]

    def pack(self, ids: list[list[int]]) -> tuple:
        """One pass's host tensors: token ids, positions, each token's text span, each text's
        segment; the stream padded to its bucket (the padding one more text) under graphs."""
        lens = [len(t) for t in ids]
        total = sum(lens)
        size = bucket(total) if self.graphs is not None else total
        segments = self.max_batch + 1 if self.graphs is not None else len(ids) + 1
        flat = np.zeros(size, np.int64)
        pos = np.zeros(size, np.int64)
        start = np.full(size, total, np.int32)
        end = np.full(size, size, np.int32)
        seg_start = np.full(segments, total, np.int32)
        seg_len = np.zeros(segments, np.int32)
        at = 0
        for i, (tokens, n) in enumerate(zip(ids, lens)):
            flat[at : at + n] = tokens
            pos[at : at + n] = np.arange(n)
            start[at : at + n], end[at : at + n] = at, at + n
            seg_start[i], seg_len[i] = at, n
            at += n
        pos[total:] = np.arange(size - total)
        arrays = (flat, pos, start, end, seg_start, seg_len)
        return tuple(torch.from_numpy(a) for a in arrays)

    def forward(self, ids, pos, start, end, seg_start, seg_len) -> torch.Tensor:
        """The device pass: ``[segments, out]`` normalized fp32 vectors."""
        enc, ops = self.encoder, self.ops
        x = enc.embed_tokens(ids) * self.scale
        if self.residual == "float32":
            x = x.float()
        rope = {
            kind: (cos[pos][:, None], sin[pos][:, None]) for kind, (cos, sin) in self.tables.items()
        }
        for layer in enc.layers:
            x = layer(x, *rope[layer.kind], start, end, ops)
        x = enc.norm(x)
        y = ops["segment_mean"](x, seg_start, seg_len)
        for weight in self.dense:
            y = y @ weight.T
        return F.normalize(y, dim=-1)

    def run(self, packed: tuple) -> torch.Tensor:
        """One packed pass (``pack``) to its vectors ``[segments, out]`` (fp32, device)."""
        inputs = tuple(t.to(self.device, non_blocking=True) for t in packed)
        if self.graphs is not None:
            return self.graphs(inputs[0].shape[0], *inputs)
        return self.forward(*inputs)

    def embed(self, texts: list[str]) -> torch.Tensor:
        """Normalized vectors ``[len(texts), out]`` (fp32, device), ``max_batch`` per pass."""
        ids = self.tokenize(texts)
        out = []
        for lo in range(0, len(ids), self.max_batch):
            part = ids[lo : lo + self.max_batch]
            out.append(self.run(self.pack(part))[: len(part)])
        return torch.cat(out)
