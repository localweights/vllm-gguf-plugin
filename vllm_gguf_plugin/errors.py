# SPDX-License-Identifier: Apache-2.0
"""Typed errors for the vLLM GGUF plugin."""


class GGUFUnmappedTensorError(RuntimeError):
    """Raised when GGUF tensor names cannot be mapped to HF param names.

    The message lists the offending tensor names so the operator can diagnose
    a corrupted GGUF, an unsupported architecture, or a genuine mapping gap.
    """

    def __init__(self, tensor_names: list[str] | set[str]):
        super().__init__(
            f"GGUF tensors with no HF mapping: {sorted(tensor_names)}"
        )
        self.tensor_names = sorted(tensor_names)