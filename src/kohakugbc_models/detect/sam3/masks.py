"""SAM 3's mask decoder under transformers' parameter names (``mask_decoder.*``): only for the
mask box sources. The semantic head is not computed (GBC reads no semantic map)."""

import torch
import torch.nn.functional as F
from torch import nn

from kohakugbc_models.detect.sam3 import ops
from kohakugbc_models.detect.sam3.fusion import Attention
from kohakugbc_models.detect.sam3.vision import linear, norm


class PixelDecoder(nn.Module):
    def __init__(self, hidden: int, stages: int):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1) for _ in range(stages)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(8, hidden) for _ in range(stages))

    def forward(self, features: list) -> torch.Tensor:
        """Features fine to coarse ``[B, C, H_i, W_i]``: pixel embeddings at the finest."""
        x = features[-1]
        for i, feat in enumerate(reversed(features[:-1])):
            x = F.interpolate(x, size=feat.shape[-2:], mode="nearest") + feat
            x = F.relu(self.norms[i](self.conv_layers[i](x)))
        return x


class MaskEmbedder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = linear(layer, x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        self.pixel_decoder = PixelDecoder(hidden, cfg.num_upsampling_stages)
        self.mask_embedder = MaskEmbedder(hidden)
        self.instance_projection = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.semantic_projection = nn.Conv2d(hidden, 1, kernel_size=1)
        self.prompt_cross_attn = Attention(hidden, cfg.num_attention_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden)
        self.ops = ops.EAGER

    def embed(self, queries, levels: list, fused, text, text_mask) -> tuple:
        """The mask embeddings ``[B, Q, C]`` of normalized queries ``[B, Q, C]`` and the pixel
        embeddings ``[B, C, H0, W0]`` of the neck levels ``[B, C, H_i, W_i]`` (fine to coarse;
        the coarsest replaced by the fused vision)."""
        normed = norm(self.prompt_cross_attn_norm, fused)
        fused = fused + self.prompt_cross_attn(normed, text, text, text_mask)
        height, width = levels[-1].shape[-2:]
        batch, _, channels = fused.shape
        coarse = fused[:, : height * width].transpose(1, 2).reshape(batch, channels, height, width)
        pixels = self.pixel_decoder(list(levels[:-1]) + [coarse])
        return self.mask_embedder(queries), pixels

    def forward(self, queries, levels: list, fused, text, text_mask) -> torch.Tensor:
        """Mask logits ``[B, Q, H0, W0]``."""
        emb, pixels = self.embed(queries, levels, fused, text, text_mask)
        return torch.einsum("bqc,bchw->bqhw", emb, self.instance_projection(pixels))

    def boxes(self, queries, levels: list, fused, text, text_mask, threshold: float) -> tuple:
        """Each query's mask box on the ``H0 x W0`` grid ``[B, Q, 4]`` (int32; left, top, right,
        bottom, right / bottom one past) and whether its mask is empty ``[B, Q]``, the logits
        never written (``mask_box``)."""
        emb, pixels = self.embed(queries, levels, fused, text, text_mask)
        proj = self.instance_projection
        return self.ops.mask_box(emb, pixels, proj.weight, proj.bias, threshold)
