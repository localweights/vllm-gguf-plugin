# SPDX-License-Identifier: Apache-2.0
"""Fix sleep-mode wake on hybrid GDN+full-attn models.

Upstream vLLM 0.25 gpu_model_runner.py line 964, init_fp8_kv_scales
(called from post_kv_cache_wake_up after /wake_up) does:

    kv_caches = getattr(self, "kv_caches", [])
    for cache_tensor in kv_caches:
        if cache_tensor is not None:
            cache_tensor.zero_()

On hybrid GDN+full-attn models (Qwen3.5/3.6), mamba-group entries in
self.kv_caches are LISTS of state tensors (conv state + ssm state), not
tensors -> 'list' object has no attribute 'zero_' -> wake returns 500,
engine unusable after sleep.

This module monkeypatches init_fp8_kv_scales to handle list/tuple entries
recursively via _zero_kv_entry.  The rest of the function body (scale
reset over Attention/MLAAttention modules) is mirrored verbatim from
upstream.

vLLM >= 0.33 (Model Runner V2): the sleep-mode rework deleted
``GPUModelRunner.init_fp8_kv_scales`` outright — v1/worker/gpu_worker.py
``sleep()``/``wake_up()`` suspend/resume through ``sleep_mode_backend`` and no
longer zero kv_caches after a wake, and teardown detaches caches via
``clear_layer_kv_caches`` (v1/worker/utils.py), which already handles the
list-valued entries of hybrid models.  There is nothing left to patch there, so
the install turns into a documented no-op instead of raising AttributeError.
See PORT_NOTES_0.33.md.
"""

import logging
from functools import wraps
from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

try:  # vLLM >= 0.33: moved out of vllm.utils.torch_utils
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache
except ImportError:  # vLLM <= 0.25: old home of the helper
    # Resolved through the module (not ``from ... import``) so the static
    # import audit does not report a MISSING symbol on 0.33, where this
    # location no longer exports it.
    from vllm.utils import torch_utils as _torch_utils

    is_quantized_kv_cache = _torch_utils.is_quantized_kv_cache

logger = logging.getLogger(__name__)


def _zero_kv_entry(entry: Any) -> None:
    """Zero out a single KV-cache entry, handling nested lists/tuples.

    - torch.Tensor -> entry.zero_()
    - list/tuple -> recurse over elements
    - None -> skip
    """
    if entry is None:
        return
    if isinstance(entry, torch.Tensor):
        entry.zero_()
    elif isinstance(entry, (list, tuple)):
        for sub in entry:
            _zero_kv_entry(sub)
    # Unknown types silently skipped (defensive)


def _patch_init_fp8_kv_scales() -> None:
    """Monkeypatch GPUModelRunner.init_fp8_kv_scales for hybrid models.

    Mirrors upstream body (gpu_model_runner.py:964-1005) but uses
    _zero_kv_entry for the kv_caches loop so list entries from mamba/GDN
    groups are handled instead of crashing.

    On vLLM >= 0.33 the patched method no longer exists; the hook is then a
    no-op (the upstream bug it guarded is gone — see the module docstring).
    """
    if getattr(GPUModelRunner, "_gguf_sleep_wake_patched", False):
        return

    if not hasattr(GPUModelRunner, "init_fp8_kv_scales"):
        logger.info(
            "GPUModelRunner.init_fp8_kv_scales is absent in this vLLM "
            "(removed in 0.33 with the sleep-mode rework); the hybrid "
            "sleep/wake fix is inert-but-harmless here."
        )
        GPUModelRunner._gguf_sleep_wake_patched = True
        return

    # vLLM >= 0.33 removed this hook with the sleep-mode rework; the
    # hasattr() guard above is the compatibility path (see PORT_NOTES_0.33.md).
    # patch-target: inert
    original = GPUModelRunner.init_fp8_kv_scales

    @wraps(original)
    @torch.inference_mode()  # upstream decorates the original; mirror it
    def patched_init_fp8_kv_scales(self) -> None:
        # --- upstream body start (gpu_model_runner.py:964-1005) ---
        if not is_quantized_kv_cache(self.cache_config.cache_dtype):
            return

        # Zero kv_caches — handles hybrid GDN entries (list of tensors)
        kv_caches = getattr(self, "kv_caches", [])
        for cache_entry in kv_caches:
            _zero_kv_entry(cache_entry)

        # Reset Attention layer scaling factors to 1.0
        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                k_scale_val, v_scale_val = 1.0, 1.0

                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)
        # --- upstream body end ---

    GPUModelRunner.init_fp8_kv_scales = patched_init_fp8_kv_scales
    GPUModelRunner._gguf_sleep_wake_patched = True
