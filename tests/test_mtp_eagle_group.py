# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for annotate_mtp_eagle_groups.

Constructs real vLLM data classes with minimal fake specs — no engine
boot, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_gguf_plugin.mtp_eagle_group import annotate_mtp_eagle_groups


# ---- helpers ----

def _make_vllm_config(method: str | None) -> SimpleNamespace:
    """Minimal VllmConfig stand-in with just what annotate reads."""
    if method is None:
        speculative_config = None
    else:
        speculative_config = SimpleNamespace(
            use_eagle=lambda: method in ("eagle", "eagle3", "mtp", "dflash", "dspark"),
        )
    return SimpleNamespace(speculative_config=speculative_config)


def _full_attn_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1, 256),),
        dtypes=(torch.float16,),
    )


def _make_kv_cache_spec(
    num_full: int,
    num_mamba: int,
    mtp_layer_name: str | None,
) -> dict[str, object]:
    """Build a kv_cache_spec dict.

    Args:
        num_full: Number of target-model full-attention layers.
        num_mamba: Number of target-model mamba/GDN layers.
        mtp_layer_name: If set, key for the MTP draft attention layer
            (same spec as full-attention).
    """
    spec: dict[str, object] = {}
    fa = _full_attn_spec()
    mb = _mamba_spec()
    for i in range(num_full):
        spec[f"model.layers.{i}.self_attn"] = fa
    for i in range(num_mamba):
        spec[f"model.layers.{num_full + i}.linear_attn"] = mb
    if mtp_layer_name is not None:
        spec[mtp_layer_name] = fa
    return spec


def _make_groups_for_spec(
    kv_cache_spec: dict[str, object],
) -> list[KVCacheGroupSpec]:
    """Simulate the grouping get_kv_cache_groups would produce.

    Full-attention layers (including MTP draft) are in one group, mamba
    layers in another — matches the real behaviour for uniform-page-size
    hybrid models.
    """
    full_names: list[str] = []
    mamba_names: list[str] = []
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, FullAttentionSpec):
            full_names.append(name)
        else:
            mamba_names.append(name)

    groups: list[KVCacheGroupSpec] = []
    if full_names:
        groups.append(
            KVCacheGroupSpec(layer_names=full_names, kv_cache_spec=_full_attn_spec())
        )
    if mamba_names:
        groups.append(
            KVCacheGroupSpec(layer_names=mamba_names, kv_cache_spec=_mamba_spec())
        )
    return groups


# ---- test cases ----


def test_mtp_group_marked():
    """MTP spec config + 4 groups (1 full-attn incl. draft, 3 mamba)
    -> only the full-attn group (containing the draft layer) marked."""
    config = _make_vllm_config("mtp")
    spec = _make_kv_cache_spec(num_full=1, num_mamba=3, mtp_layer_name="mtp.layers.0.self_attn")
    groups = _make_groups_for_spec(spec)

    annotate_mtp_eagle_groups(config, spec, groups)

    # Full-attention group contains the MTP draft layer -> marked.
    for g in groups:
        if "mtp.layers.0.self_attn" in g.layer_names:
            assert g.is_eagle_group, \
                "group containing MTP draft layer must be flagged"
        else:
            assert not g.is_eagle_group, \
                "mamba-only group must NOT be flagged"


def test_no_speculative_config():
    """No speculative_config -> nothing marked."""
    config = _make_vllm_config(None)
    spec = _make_kv_cache_spec(num_full=1, num_mamba=3, mtp_layer_name="mtp.layers.0.self_attn")
    groups = _make_groups_for_spec(spec)

    annotate_mtp_eagle_groups(config, spec, groups)

    assert not any(g.is_eagle_group for g in groups), \
        "no speculative config: nothing must be marked"


def test_already_marked():
    """A group already has is_eagle_group=True -> function makes no changes."""
    config = _make_vllm_config("mtp")
    spec = _make_kv_cache_spec(num_full=1, num_mamba=3, mtp_layer_name="mtp.layers.0.self_attn")
    groups = _make_groups_for_spec(spec)
    # Pre-mark the mamba group (simulating prior annotation of a wrong group).
    groups[1].is_eagle_group = True

    annotate_mtp_eagle_groups(config, spec, groups)

    # Should not modify anything (already marked -> early return).
    assert groups[1].is_eagle_group, "pre-existing mark must survive"
    assert not groups[0].is_eagle_group, \
        "function must not add marks when one already exists"


def test_use_eagle_false():
    """use_eagle() returns False -> nothing marked (MTP is in the True set)."""
    config = _make_vllm_config("draft_model")  # not in eagle set
    spec = _make_kv_cache_spec(num_full=1, num_mamba=3, mtp_layer_name="mtp.layers.0.self_attn")
    groups = _make_groups_for_spec(spec)

    annotate_mtp_eagle_groups(config, spec, groups)

    assert not any(g.is_eagle_group for g in groups)


def test_no_draft_layer():
    """No MTP draft layer in kv_cache_spec -> must not mark anything."""
    config = _make_vllm_config("mtp")
    spec = _make_kv_cache_spec(
        num_full=16, num_mamba=48, mtp_layer_name=None  # no draft layer
    )
    groups = _make_groups_for_spec(spec)

    annotate_mtp_eagle_groups(config, spec, groups)

    assert not any(g.is_eagle_group for g in groups), \
        "no draft layer: nothing must be marked"


def test_multiple_mtp_layers():
    """When multiple MTP draft layers exist (mtp_num_hidden_layers > 1),
    all groups containing any MTP draft layer are marked."""
    config = _make_vllm_config("mtp")
    spec = _make_kv_cache_spec(num_full=1, num_mamba=3, mtp_layer_name=None)
    spec["mtp.layers.0.self_attn"] = _full_attn_spec()
    spec["mtp.layers.1.self_attn"] = _full_attn_spec()
    groups = _make_groups_for_spec(spec)

    annotate_mtp_eagle_groups(config, spec, groups)

    for g in groups:
        mtp_names = [n for n in g.layer_names if n.startswith("mtp.layers")]
        if mtp_names:
            assert g.is_eagle_group, \
                "group with MTP draft layers must be marked"
        else:
            assert not g.is_eagle_group, \
                "non-MTP group must not be marked"
