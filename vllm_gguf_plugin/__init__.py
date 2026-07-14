# SPDX-License-Identifier: Apache-2.0

from .op_guard import install_idempotent_op_registration

# Must run before .quantization (and before vllm core's quantization/gguf.py)
# imports register duplicate vllm:: custom ops — see op_guard.py.
install_idempotent_op_registration()

from .config_parser import GGUFConfigParser  # noqa: E402
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
