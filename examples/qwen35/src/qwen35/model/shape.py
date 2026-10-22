"""The shape of a Qwen3.5 checkpoint, read from its ``config.json``: every size the engine's
pools, kernels and forward take, for any member of the family (2B, 4B, 9B)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    hidden: int
    intermediate: int
    layers: tuple[str, ...]  # "linear_attention" / "full_attention" per layer
    vocab: int
    eps: float
    # Full attention
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    mrope_section: tuple[int, int, int]
    # Gated DeltaNet
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_width: int
    # Tokens
    image_token: int
    vision_start: int
    vision_end: int
    eos: int

    @classmethod
    def from_config(cls, config: dict) -> "Shape":
        text = config["text_config"]
        rope = text["rope_parameters"]
        head_dim = int(text["head_dim"])
        return cls(
            hidden=int(text["hidden_size"]),
            intermediate=int(text["intermediate_size"]),
            layers=tuple(text["layer_types"]),
            vocab=int(text["vocab_size"]),
            eps=float(text["rms_norm_eps"]),
            heads=int(text["num_attention_heads"]),
            kv_heads=int(text["num_key_value_heads"]),
            head_dim=head_dim,
            rotary_dim=int(head_dim * float(rope.get("partial_rotary_factor", 1.0))),
            rope_theta=float(rope["rope_theta"]),
            mrope_section=tuple(rope.get("mrope_section", [11, 11, 10])),
            key_heads=int(text["linear_num_key_heads"]),
            value_heads=int(text["linear_num_value_heads"]),
            key_dim=int(text["linear_key_head_dim"]),
            value_dim=int(text["linear_value_head_dim"]),
            conv_width=int(text["linear_conv_kernel_dim"]),
            image_token=int(config["image_token_id"]),
            vision_start=int(config["vision_start_token_id"]),
            vision_end=int(config["vision_end_token_id"]),
            eos=int(text["eos_token_id"]),
        )

    @property
    def delta_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layers) if t == "linear_attention"]

    @property
    def attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layers) if t == "full_attention"]

    @property
    def conv_channels(self) -> int:
        """Channels of the DeltaNet's short convolution: q, k and v side by side."""
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    def state_bytes(self, itemsize: int) -> int:
        """One sequence's recurrent state over every DeltaNet layer."""
        per = self.value_heads * self.key_dim * self.value_dim * itemsize
        return per * len(self.delta_layers)

    def conv_bytes(self, itemsize: int) -> int:
        return self.conv_channels * (self.conv_width - 1) * itemsize * len(self.delta_layers)

    def kv_bytes_per_token(self, itemsize: int) -> int:
        return 2 * self.kv_heads * self.head_dim * itemsize * len(self.attention_layers)
