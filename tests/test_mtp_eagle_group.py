# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests pinning the EAGLE/MTP KV-group annotation for the GGUF MTP lane.

History: on vLLM 0.25 the plugin carried ``vllm_gguf_plugin/mtp_eagle_group.py``
(branch commit c830a77) which monkeypatched
``vllm.v1.core.kv_cache_utils.get_kv_cache_groups`` to set
``KVCacheGroupSpec.is_eagle_group`` on the group holding the Qwen3.5 MTP draft
attention layers (``mtp.layers.*``).

vLLM 0.33 annotates that group natively, from the KV-cache *spec* side:

* ``vllm/v1/worker/gpu/spec_decode/speculator.py:240-242`` (MRV2) and
  ``vllm/v1/spec_decode/llm_base_proposer.py:1334-1335`` (MRV1) flag every
  draft attention layer with ``is_draft_layer = True``;
* ``vllm/v1/worker/gpu/attn_utils.py:150-151`` (MRV2, same code in
  ``gpu_model_runner.py:7114``) turns that into ``KVCacheSpec.is_draft = True``;
* ``vllm/v1/core/kv_cache_planning.py:1650-1656`` (in
  ``get_global_kv_cache_groups``, called from ``v1/engine/core.py:292``) sets
  ``group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)``
  whenever ``speculative_config.use_eagle_block_drop()``.

The patch file was therefore deleted; these tests assert the native contract the
lane now relies on (the intent of the old tests: only the draft's group is
flagged, never the whole model).
"""

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

# ---- helpers ----


class _SpecConfig:
    """Minimal SpeculativeConfig stand-in (only what planning reads)."""

    def __init__(self, method: str = "mtp", block_drop: bool = True):
        self.method = method
        self.disable_eagle_block_drop = not block_drop

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    def needs_eagle_block_drop(self) -> bool:
        return self.use_eagle() and self.method not in ("dflash", "dspark")

    def use_eagle_block_drop(self) -> bool:
        return self.needs_eagle_block_drop() and not self.disable_eagle_block_drop


def _make_vllm_config(spec_config):
    from types import SimpleNamespace

    # The KV cache layout is normally resolved by the engine core before the
    # groups are built; any member works for grouping/annotation.
    cache_config = CacheConfig(block_size=16, mamba_cache_mode="align")
    cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout.LBHNC
    return SimpleNamespace(
        speculative_config=spec_config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(supports_kv_cache_grouping=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=cache_config,
        num_prefill_lookahead_tokens=0,
    )


def _full_attn_spec(draft: bool = False) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
        is_draft=draft,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1, 256),),
        dtypes=(torch.float16,),
    )


def _grouped(config, kv_cache_spec: dict[str, KVCacheSpec]) -> list:
    return get_global_kv_cache_groups(config, [kv_cache_spec])


def _eagle_layer_names(groups: list[KVCacheGroupSpec]) -> set:
    return {
        name
        for group in groups
        if group.is_eagle_group
        for name in group.layer_names
    }


# ---- test cases ----


def test_mtp_draft_group_marked_natively():
    """Draft full-attn group -> flagged; target full-attn and mamba groups -> not.

    This is the behaviour mtp_eagle_group.py used to provide on 0.25.
    """
    config = _make_vllm_config(_SpecConfig("mtp"))
    groups = _grouped(
        config,
        {
            "model.layers.0.self_attn": _full_attn_spec(),
            "model.layers.1.linear_attn": _mamba_spec(),
            "mtp.layers.0.self_attn": _full_attn_spec(draft=True),
        },
    )

    assert len(groups) == 3
    assert _eagle_layer_names(groups) == {"mtp.layers.0.self_attn"}
    for group in groups:
        if "mtp.layers.0.self_attn" in group.layer_names:
            assert group.is_eagle_group
        else:
            assert not group.is_eagle_group, (
                f"non-draft group {group.layer_names} must not be flagged volatile"
            )


def test_no_speculative_config_marks_nothing():
    config = _make_vllm_config(None)
    groups = _grouped(
        config,
        {
            "model.layers.0.self_attn": _full_attn_spec(),
            "mtp.layers.0.self_attn": _full_attn_spec(draft=True),
        },
    )
    assert not any(group.is_eagle_group for group in groups)


def test_block_drop_disabled_marks_nothing():
    """``--speculative-config disable_eagle_block_drop`` opts out of the
    volatile-tail semantics, so no group may be flagged."""
    config = _make_vllm_config(_SpecConfig("mtp", block_drop=False))
    groups = _grouped(
        config,
        {
            "model.layers.0.self_attn": _full_attn_spec(),
            "mtp.layers.0.self_attn": _full_attn_spec(draft=True),
        },
    )
    assert not any(group.is_eagle_group for group in groups)


def test_no_draft_layer_marks_nothing():
    """No spec carries is_draft (drafter owns no KV cache) -> nothing flagged,
    so the offloading scheduler treats all groups as non-draft."""
    config = _make_vllm_config(_SpecConfig("mtp"))
    groups = _grouped(
        config,
        {
            "model.layers.0.self_attn": _full_attn_spec(),
            "model.layers.1.linear_attn": _mamba_spec(),
        },
    )
    assert not any(group.is_eagle_group for group in groups)


def test_multiple_mtp_layers_all_marked():
    """mtp_num_hidden_layers > 1: every draft layer's group is flagged."""
    config = _make_vllm_config(_SpecConfig("mtp"))
    groups = _grouped(
        config,
        {
            "model.layers.0.self_attn": _full_attn_spec(),
            "model.layers.1.linear_attn": _mamba_spec(),
            "mtp.layers.0.self_attn": _full_attn_spec(draft=True),
            "mtp.layers.1.self_attn": _full_attn_spec(draft=True),
        },
    )
    assert _eagle_layer_names(groups) == {
        "mtp.layers.0.self_attn",
        "mtp.layers.1.self_attn",
    }


def test_use_eagle_block_drop_true_for_mtp():
    """The native annotation is gated on use_eagle_block_drop(); MTP must pass."""
    spec = _SpecConfig("mtp")
    assert spec.use_eagle()
    assert spec.use_eagle_block_drop()
    assert not _SpecConfig("draft_model").use_eagle_block_drop()


def test_plugin_does_not_patch_kv_cache_groups():
    """Wiring proof for the deletion: no plugin wrapper is installed on the
    (moved) group builder in either the old or the new module."""
    import vllm.v1.core.kv_cache_planning as planning
    import vllm.v1.core.kv_cache_utils as legacy

    import vllm_gguf_plugin

    vllm_gguf_plugin.register()

    assert not hasattr(planning.get_kv_cache_groups, "_gguf_eagle_group_patched")
    assert not hasattr(legacy, "get_kv_cache_groups")


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC])
def test_groups_partition_every_layer(layout):
    """Sanity: the native builder still partitions all layers exactly once."""
    config = _make_vllm_config(_SpecConfig("mtp"))
    config.cache_config.get_resolved_kv_cache_layout = lambda: layout
    spec = {
        "model.layers.0.self_attn": _full_attn_spec(),
        "model.layers.1.linear_attn": _mamba_spec(),
        "mtp.layers.0.self_attn": _full_attn_spec(draft=True),
    }
    groups = _grouped(config, spec)
    seen = [name for group in groups for name in group.layer_names]
    assert sorted(seen) == sorted(spec)
