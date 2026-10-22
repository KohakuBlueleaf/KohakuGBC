"""Adapters that need no engine library, by name: the ``remote`` entry names the adapter of
units it cannot build where the machine runs (a vLLM host on a Linux node, a torch host on a
GPU node)."""

from togeari.adapters import TorchAdapter


class VllmChatAdapter:
    """The ``chat`` node's request handed to a ``vllm`` unit as it is; the reply text back."""

    protocols = {"vllm"}
    light = True

    def check(self, signature) -> list[str]:
        return []

    def prepare(self, inputs: dict, protocol: str) -> dict:
        return inputs["request"]

    def decode(self, raw, protocol: str) -> str:
        return raw


ADAPTERS = {
    "vllm": VllmChatAdapter,
    "torch": TorchAdapter,
    "torch-text": lambda: TorchAdapter(arg="text"),
}
