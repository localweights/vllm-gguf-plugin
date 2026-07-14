# SPDX-License-Identifier: Apache-2.0
"""Make vLLM custom-op registration idempotent.

Both this plugin (``vllm_gguf_plugin.quantization``) and vLLM core
(``vllm.model_executor.layers.quantization.gguf``) register the ops
``vllm::_fused_mul_mat_gguf`` / ``vllm::_fused_moe_gguf`` at module import,
unguarded.  Whichever module is imported second raises::

    RuntimeError: Tried to register an operator (vllm::_fused_mul_mat_gguf ...)
    with the same name and overload name multiple times.

The plugin loads first (vllm.general_plugins entry point fires during CLI arg
parsing), so its implementations win; this wrapper makes any later duplicate
registration a silent no-op instead of a crash.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_installed = False


def install_idempotent_op_registration() -> None:
    """Wrap ``direct_register_custom_op`` to skip already-registered op names."""
    global _installed
    if _installed:
        return

    import torch

    from vllm.utils import torch_utils

    original = torch_utils.direct_register_custom_op

    def _idempotent_register(op_name, op_func, *args, **kwargs):
        target_lib = kwargs.get("target_lib")
        # Only the default vllm namespace can collide between core and plugin.
        if target_lib is None and hasattr(torch.ops.vllm, op_name):
            logger.debug(
                "custom op vllm::%s already registered; skipping duplicate", op_name
            )
            return
        return original(op_name, op_func, *args, **kwargs)

    torch_utils.direct_register_custom_op = _idempotent_register
    _installed = True
