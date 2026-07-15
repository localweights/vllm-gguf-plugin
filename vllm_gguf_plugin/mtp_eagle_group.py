# SPDX-License-Identifier: Apache-2.0
"""Annotate Qwen3.5 MTP draft KV-cache group as is_eagle_group.

The upstream offloading scheduler (OffloadingConnector in
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)
collects eagle group indices from KVCacheConfig via

    eagle_groups = {idx for idx, g in enumerate(spec.kv_cache_config.kv_cache_groups)
                    if g.is_eagle_group}

and falls back to ALL groups when use_eagle() is True but none are marked.
Our GGUF Qwen3.6 hybrid MTP lane never sets is_eagle_group, so EVERY
KV-cache group (full-attn + GDN/mamba) gets flagged volatile — the trailing
block of every group is excluded from offload store and popped on load,
functionally killing KV spill.

The upstream fix (vllm/v1/core/kv_cache_utils.py, _annotate_eagle_groups_deepseek_v4)
only handles DeepSeek V4.  This module extends the same logic to Qwen3.5 MTP
draft attention layers via a monkeypatch of get_kv_cache_groups.
"""

from functools import wraps

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


def annotate_mtp_eagle_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Mark the group containing Qwen3.5 MTP draft attention layer(s).

    Pure function — no GPU, no engine boot.  Called after the upstream
    group-building logic.

    Strategy:
      1. Early-exit if speculative config is absent / not eagle-style.
      2. Early-exit if any group already has is_eagle_group=True (upstream
         or a prior patch already handled it).
      3. Name-based match: look for keys in kv_cache_spec whose layer name
         starts with "mtp.layers".  This prefix is set by
         Qwen3_5MultiTokenPredictor.__init__ in
         vllm/model_executor/models/qwen3_5_mtp.py:
             self.layers = ModuleList(
                 Qwen3_5DecoderLayer(..., prefix=f"{prefix}.layers.{idx}")
                 for idx in range(self.num_mtp_layers)
             )
         where prefix = "mtp" (passed from Qwen3_5MTP.__init__).
         Each DecoderLayer creates its Attention module at
         prefix + ".self_attn", which registers in
         compilation_config.static_forward_context[prefix].
      4. Set is_eagle_group = True on the group that contains the identified
         draft layer.

    Name-based match is preferred because the Qwen3.5 MTP draft prefix
    "mtp.layers" is distinctive — set by Qwen3_5MultiTokenPredictor in
    vllm/model_executor/models/qwen3_5_mtp.py L79-83.  The deepseek-style
    "last key of kv_cache_spec" heuristic (vllm/v1/core/kv_cache_utils.py
    @ _annotate_eagle_groups_deepseek_v4) is deliberately NOT implemented
    here: it would blindly mark a target-model group when no Qwen MTP
    draft layer exists, which breaks the offload scheduler's assumption
    that only the draft group is an eagle group.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return

    # Bail if upstream already annotated (deepseek path or future fix).
    if any(g.is_eagle_group for g in kv_cache_groups):
        return

    # --- Identify MTP draft attention layer name(s) ---
    # Name-based match for Qwen3.5 MTP prefix "mtp.layers".
    # The draft register prefix="mtp.layers.{idx}.self_attn" in
    # compilation_config.static_forward_context — see
    # vllm/model_executor/models/qwen3_5_mtp.py L79-83 for the prefix chain.
    mtp_layer_names = [
        name for name in kv_cache_spec if name.startswith("mtp.layers")
    ]
    if not mtp_layer_names:
        return  # no Qwen-style MTP draft layers found

    # --- Mark the containing group(s) ---
    for name in mtp_layer_names:
        for group in kv_cache_groups:
            if name in group.layer_names:
                group.is_eagle_group = True
                break


def _patch_eagle_group_annotation() -> None:
    """Monkeypatch vllm.v1.core.kv_cache_utils.get_kv_cache_groups.

    Installs a wrapper that calls the original then runs
    annotate_mtp_eagle_groups on the result.  Idempotent via the
    _gguf_eagle_group_patched attribute guard (matching the pattern in
    mtp_enable.py).
    """
    import vllm.v1.core.kv_cache_utils as kv_utils

    if getattr(kv_utils.get_kv_cache_groups, "_gguf_eagle_group_patched", False):
        return

    original = kv_utils.get_kv_cache_groups

    @wraps(original)
    def wrapper(
        vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        groups = original(vllm_config, kv_cache_spec)
        annotate_mtp_eagle_groups(vllm_config, kv_cache_spec, groups)
        return groups

    wrapper._gguf_eagle_group_patched = True  # type: ignore[attr-defined]
    kv_utils.get_kv_cache_groups = wrapper
