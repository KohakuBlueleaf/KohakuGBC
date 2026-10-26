"""The PixAI tagger (``pixai-labs/pixai-tagger-v1.0``) on SAM 3's ViT layers: one pass per image,
the whole image's tag probabilities and, from the same tokens, each region's.

The checkpoint is a finetuned SAM 3 vision backbone (``Backbone`` runs it with its kernels, fp8
recipes and CUDA graphs; LayerNorm eps 1e-5) and an attention-pool head (LayerNorm, a learned
position per token, one learned query, a projection, the 30,877-tag linear, a sigmoid; eager).
A region's query attends only to the tokens inside its box (at least one): the whole-image head
read over part of the grid, which the tagger was not trained on.

Pictures arrive with their longer side at 1008 (``geometry.fit``); ``pixels`` pads them centred
with -1, as the checkpoint's ``RescalePadProcessor``.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

from kohakugbc_models.detect.sam3.ops import EAGER, Ops
from kohakugbc_models.detect.sam3.vision import Backbone
from kohakugbc_models.kernels import fp8
from kohakugbc_models.kernels.graphs import Graphs
from kohakugbc_models.tag.geometry import SIZE, pad_geometry

FILES = ["config.json", "model.safetensors"]


def backbone_config(cfg: dict) -> SimpleNamespace:
    """``Backbone``'s settings from the checkpoint's ``ViTDetCls`` config."""
    return SimpleNamespace(
        patch_size=cfg["patch_size"],
        window_size=cfg["window_size"],
        num_channels=cfg["in_chans"],
        hidden_size=cfg["embed_dim"],
        pretrain_image_size=cfg["pretrain_img_size"],
        layer_norm_eps=1e-5,
        num_attention_heads=cfg["num_heads"],
        intermediate_size=int(cfg["embed_dim"] * cfg["mlp_ratio"]),
        num_hidden_layers=cfg["depth"],
        global_attn_indexes=list(cfg["global_att_blocks"]),
        rope_parameters={"rope_theta": 10000.0},
        image_size=cfg["img_size"],
    )


class Pool(nn.Module):
    """The attention-pool head and the tag linear."""

    def __init__(self, dim: int, heads: int, tokens: int, tags: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(torch.zeros(tokens, dim))
        self.q = nn.Parameter(torch.zeros(dim))
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, tags)

    def forward(self, tokens: torch.Tensor, masks: list[torch.Tensor | None]) -> list:
        """``tokens`` ``[B, N, C]``; ``masks[b]`` ``[R, N]`` (True: the region's tokens) or None.
        Per image ``[1 + R, tags]`` probabilities: the whole image, then each region."""
        batch, n, dim = tokens.shape
        x = self.norm(tokens) + self.pos_embed[None]
        k, v = self.kv(x).chunk(2, dim=-1)
        k = k.view(batch, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, n, self.heads, self.head_dim).transpose(1, 2)
        q = self.q.view(1, self.heads, 1, self.head_dim)
        scores = (q * k).sum(-1) / math.sqrt(self.head_dim)
        out = []
        for b in range(batch):
            rows = scores[b][None]
            if masks[b] is not None:
                region = scores[b][None].expand(len(masks[b]), -1, -1)
                region = region.masked_fill(~masks[b][:, None, :], float("-inf"))
                rows = torch.cat([rows, region])
            weights = rows.float().softmax(-1).to(v.dtype)
            pooled = torch.einsum("rhn,hnd->rhd", weights, v[b]).reshape(len(rows), dim)
            out.append(self.head(self.proj(pooled)).float().sigmoid())
        return out


class PixaiTagger:
    """Load, place and run the tagger. ``fp8``: the ViT's linears in an fp8 recipe (``auto``:
    the fastest the card runs); ``kernels``: the kernel registry's ops; ``vision_graph``: the ViT
    replayed from one CUDA graph per batch size up to ``max_batch``, captured at load."""

    def __init__(
        self,
        model: str = "pixai-labs/pixai-tagger-v1.0",
        max_batch: int = 4,
        fp8: str | None = None,
        kernels: bool = True,
        vision_graph: bool = True,
    ):
        self.model_name, self.max_batch = model, max_batch
        self.fp8_setting, self.kernels, self.vision_graph = fp8, kernels, vision_graph
        self.backbone = self.pool = self.graphs = self.parked = None
        self.tags: list[str] = []
        self.categories: list[str] = []
        self.thresholds = None
        self.index: dict[str, int] = {}
        self.device, self.dtype = "cpu", torch.float32

    def load(self, device, dtype) -> None:
        path = Path(self.model_name)
        if not path.is_dir():
            path = Path(snapshot_download(self.model_name, allow_patterns=FILES))
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.tags = list(cfg["tags"])
        self.index = {t: i for i, t in enumerate(self.tags)}
        self.categories = [c for c, n in cfg["tags_split"] for _ in range(n)]
        best = cfg["category_best_threshold"]
        self.thresholds = torch.tensor([best[c] for c in self.categories])
        backbone = Backbone(backbone_config(cfg))
        grid = cfg["img_size"] // cfg["patch_size"]
        pool = Pool(cfg["embed_dim"], cfg["num_heads"], grid * grid, cfg["num_classes"])
        self.load_weights(backbone, pool, path / "model.safetensors")
        self.backbone = backbone.to(dtype=dtype).eval().requires_grad_(False)
        self.pool = pool.to(dtype=dtype).eval().requires_grad_(False)
        self.dtype = dtype
        self.to(device)

    @torch.no_grad()
    def load_weights(self, backbone: Backbone, pool: Pool, file: Path) -> None:
        """The checkpoint's tensors under ``Backbone``'s and ``Pool``'s names (q / k / v split
        from the fused ``qkv``, the class token's position dropped); a missing one is an error."""
        own = {f"backbone.{k}": v for k, v in backbone.state_dict().items()}
        own |= {f"pool.{k}": v for k, v in pool.state_dict().items()}
        found = set()

        def put(name: str, tensor: torch.Tensor) -> None:
            own[name].copy_(tensor)
            found.add(name)

        with safe_open(str(file), "pt") as f:
            for key in f.keys():  # noqa: SIM118 (a safetensors handle is not iterable)
                t = f.get_tensor(key)
                parts = key.split(".")
                match parts:
                    case ["patch_embed", "proj", "weight"]:
                        put("backbone.embeddings.patch_embeddings.projection.weight", t)
                    case ["pos_embed"]:
                        put("backbone.embeddings.position_embeddings", t[:, 1:])
                    case ["ln_pre", p]:
                        put(f"backbone.layer_norm.{p}", t)
                    case ["blocks", i, "norm1" | "norm2" as n, p]:
                        put(f"backbone.layers.{i}.layer_norm{n[-1]}.{p}", t)
                    case ["blocks", i, "attn", "qkv", p]:
                        for name, part in zip(("q", "k", "v"), t.chunk(3)):
                            put(f"backbone.layers.{i}.attention.{name}_proj.{p}", part)
                    case ["blocks", i, "attn", "proj", p]:
                        put(f"backbone.layers.{i}.attention.o_proj.{p}", t)
                    case ["blocks", i, "mlp", "fc1" | "fc2" as fc, p]:
                        put(f"backbone.layers.{i}.mlp.{fc}.{p}", t)
                    case ["head_pool", *rest]:
                        put("pool." + ".".join(rest), t)
                    case ["head", p]:
                        put(f"pool.head.{p}", t)
        missing = sorted(set(own) - found)
        if missing:
            raise ValueError(f"{file}: {len(missing)} tensors missing, e.g. {missing[:5]}")

    def to(self, device) -> None:
        """Place the model on ``device``. fp8 weights quantize once on a device that runs them;
        once quantized they move to any device, which keeps them without running them
        (``parked``)."""
        self.backbone.to(device)
        self.pool.to(device)
        self.backbone.rope(device)
        self.backbone.fuse_qkv()
        quantized = self.backbone.layers[0].recipe
        recipe = quantized or (fp8.best(device) if self.fp8_setting == "auto" else self.fp8_setting)
        self.parked = None
        if recipe:
            why = fp8.supports(device, recipe)
            if why is not None:
                if quantized is None:
                    raise ValueError(f"fp8 {recipe!r} on {device}: {why}")
                self.parked = f"fp8 {recipe!r} weights parked on {device}: {why}"
            self.backbone.quantize(recipe, device)
        bound = Ops.resolved(device, self.dtype) if self.kernels else EAGER
        for module in self.backbone.modules():
            if hasattr(module, "ops"):
                module.ops = bound
        self.thresholds = self.thresholds.to(device)
        self.device = device
        self.graphs = None
        if self.vision_graph and not self.parked and torch.device(device).type == "cuda":
            self.graphs = Graphs(self.tokens, clone=(0,))
            with torch.inference_mode():
                blank = torch.zeros(self.max_batch, 3, SIZE, SIZE, device=device, dtype=self.dtype)
                for n in range(self.max_batch, 0, -1):
                    self.graphs.warm(n, blank[:n])
            self.graphs.freeze()

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, 3, 1008, 1008]`` pixels to ``[B, 5184, C]`` tokens."""
        x = self.backbone(pixels)
        return x.reshape(x.shape[0], -1, x.shape[-1])

    def pixels(self, pictures: list[np.ndarray]) -> torch.Tensor:
        """Fitted pictures (uint8, longer side 1008) as normalised, centre-padded pixels
        ``[B, 3, 1008, 1008]`` (fp32, on the host)."""
        out = torch.full((len(pictures), 3, SIZE, SIZE), -1.0)
        for b, pic in enumerate(pictures):
            top, left, h, w = pad_geometry(*pic.shape[:2])
            t = torch.from_numpy(np.ascontiguousarray(pic)).permute(2, 0, 1).float()
            out[b, :, top : top + h, left : left + w] = t / 127.5 - 1.0
        return out

    def masks(self, picture: np.ndarray, boxes: list) -> torch.Tensor | None:
        """``[R, 5184]`` token masks of ``boxes`` (``[left, top, right, bottom]`` relative to
        the picture), each at least one token."""
        if not boxes:
            return None
        top, left, h, w = pad_geometry(*picture.shape[:2])
        grid = SIZE // 14
        out = torch.zeros(len(boxes), grid, grid, dtype=torch.bool)
        for r, (x0, y0, x1, y1) in enumerate(boxes):
            c0 = min(grid - 1, int((left + x0 * w) // 14))
            r0 = min(grid - 1, int((top + y0 * h) // 14))
            c1 = max(c0 + 1, min(grid, math.ceil((left + x1 * w) / 14)))
            r1 = max(r0 + 1, min(grid, math.ceil((top + y1 * h) / 14)))
            out[r, r0:r1, c0:c1] = True
        return out.reshape(len(boxes), -1)

    @torch.inference_mode()
    def run(self, pixels: torch.Tensor, masks: list) -> list[torch.Tensor]:
        """Per image ``[1 + R, tags]`` probabilities (fp32, on the device)."""
        if self.parked:
            raise RuntimeError(self.parked)
        out = []
        for start in range(0, len(pixels), self.max_batch):
            part = pixels[start : start + self.max_batch].to(
                self.device, self.dtype, non_blocking=pixels.is_pinned()
            )
            graphed = self.graphs is not None
            tokens = self.graphs(len(part), part) if graphed else self.tokens(part)
            group = masks[start : start + len(part)]
            out += self.pool(tokens, [m if m is None else m.to(self.device) for m in group])
        return out

    def answer(
        self,
        probs: torch.Tensor,
        regions: list,
        candidates: list[str],
        crops: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """One image's answer: ``image`` the tags at or above their category's threshold with
        their probabilities, ``categories`` / ``thresholds`` each named tag's, ``regions`` per
        region id the probabilities of the candidate tags (the image's tags and ``candidates``),
        pooled (``probs`` rows after the first) or from the region's crop (``crops``: its
        ``[tags]`` probabilities); ``crops`` per crop id the crop's own tags."""
        crops = crops or {}
        whole = probs[0]
        keep = (whole >= self.thresholds).nonzero().flatten().tolist()
        given = [self.index[t] for t in candidates if t in self.index]
        cand = sorted(set(keep) | set(given))
        image = {self.tags[i]: round(float(whole[i]), 4) for i in keep}
        given_probs = {self.tags[i]: round(float(whole[i]), 4) for i in given}
        rows = probs[1:, cand].cpu().tolist() if cand and len(probs) > 1 else []
        names = [self.tags[i] for i in cand]
        scored = {rid: dict(zip(names, row)) for rid, row in zip(regions, rows)}
        own, named = {}, set(cand)
        for cid, p in crops.items():
            mine = (p >= self.thresholds).nonzero().flatten().tolist()
            own[cid] = {self.tags[i]: round(float(p[i]), 4) for i in mine}
            named |= set(mine)
            scored[cid] = dict(zip(names, p[cand].cpu().tolist())) if cand else {}
        return {
            "image": image,
            "given": given_probs,
            "categories": {self.tags[i]: self.categories[i] for i in sorted(named)},
            "thresholds": {self.tags[i]: float(self.thresholds[i]) for i in sorted(named)},
            "regions": {
                rid: {n: round(p, 4) for n, p in row.items()} for rid, row in scored.items()
            },
            "crops": own,
        }
