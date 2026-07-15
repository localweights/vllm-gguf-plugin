# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for sleep_wake_hybrid patch.

Tests _zero_kv_entry and the monkeypatched init_fp8_kv_scales using
fake runner objects (SimpleNamespace + real CacheConfig) — no engine
boot, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, CompilationConfig

from vllm_gguf_plugin.sleep_wake_hybrid import (
    _patch_init_fp8_kv_scales,
    _zero_kv_entry,
)


# ---- _zero_kv_entry unit tests ----


def test_zero_tensor():
    """A plain tensor is zeroed."""
    t = torch.tensor([1.0, 2.0, 3.0])
    _zero_kv_entry(t)
    assert t.tolist() == [0.0, 0.0, 0.0]


def test_zero_list_of_tensors():
    """A list of tensors: all zeroed."""
    tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    _zero_kv_entry(tensors)
    assert all(t.tolist() == [0.0, 0.0] for t in tensors)


def test_zero_nested_list():
    """A list containing another list + tensor: flattened recursion."""
    inner = [torch.tensor([5.0]), torch.tensor([6.0])]
    outer = [torch.tensor([1.0, 2.0]), inner]
    _zero_kv_entry(outer)
    assert outer[0].tolist() == [0.0, 0.0]
    assert inner[0].tolist() == [0.0]
    assert inner[1].tolist() == [0.0]


def test_zero_none():
    """None is skipped without error."""
    _zero_kv_entry(None)  # no exception = pass


def test_zero_mixed():
    """Mixed list with tensor, list, None: all zeroed."""
    inner = [torch.tensor([4.0, 5.0])]
    entries = [torch.tensor([1.0]), inner, None, torch.tensor([6.0])]
    _zero_kv_entry(entries)
    assert entries[0].tolist() == [0.0]
    assert inner[0].tolist() == [0.0, 0.0]
    assert entries[3].tolist() == [0.0]


# ---- patched init_fp8_kv_scales tests ----
# We monkeypatch GPUModelRunner.init_fp8_kv_scales, then call it as
# GPUModelRunner.init_fp8_kv_scales(runner) since SimpleNamespace
# instances don't inherit the class method.


def _make_fake_runner(
    cache_dtype: str,
    kv_caches_mixed: bool,
) -> SimpleNamespace:
    """Build a fake GPUModelRunner stand-in.

    Args:
        cache_dtype: Value for cache_config.cache_dtype.
        kv_caches_mixed: If True, include list entries (hybrid GDN).
    """
    cache_config = CacheConfig(cache_dtype=cache_dtype)
    compilation_config = CompilationConfig()
    # static_forward_context is empty by default — fine for this test

    if kv_caches_mixed:
        # Hybrid model: plain tensor for full-attn layers,
        # list of tensors for mamba/GDN layers.
        kv_caches = [
            torch.full((2, 2), 3.0),
            [torch.full((2,), 1.0), torch.full((2,), 2.0)],
            None,
            torch.full((2, 4), 7.0),
        ]
    else:
        kv_caches = [
            torch.full((2, 2), 3.0),
            torch.full((2, 2), 5.0),
        ]

    return SimpleNamespace(
        cache_config=cache_config,
        compilation_config=compilation_config,
        kv_caches=kv_caches,
    )


def test_patched_fp8_mixed_kv():
    """FP8 dtype with mixed kv_caches (tensor + list + None) -> no crash, all zeroed."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _patch_init_fp8_kv_scales()  # idempotent install
    runner = _make_fake_runner(cache_dtype="fp8", kv_caches_mixed=True)

    GPUModelRunner.init_fp8_kv_scales(runner)

    # All tensor entries must be zero
    assert runner.kv_caches[0].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    # List entry: both sub-tensors zeroed
    for sub in runner.kv_caches[1]:
        assert sub.tolist() == [0.0, 0.0]
    # None entry unchanged (was None, still None)
    assert runner.kv_caches[2] is None
    assert runner.kv_caches[3].tolist() == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_patched_non_quantized():
    """Non-quantized cache_dtype -> early return, tensors untouched."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _patch_init_fp8_kv_scales()  # idempotent
    runner = _make_fake_runner(cache_dtype="auto", kv_caches_mixed=True)

    GPUModelRunner.init_fp8_kv_scales(runner)

    # Tensors must retain original values
    assert runner.kv_caches[0].tolist() == [[3.0, 3.0], [3.0, 3.0]]
    for sub in runner.kv_caches[1]:
        assert sub.tolist() == [1.0, 1.0] or sub.tolist() == [2.0, 2.0]


def test_patched_tensor_only():
    """FP8 dtype with only tensor entries -> works (regression: non-hybrid path)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _patch_init_fp8_kv_scales()
    runner = _make_fake_runner(cache_dtype="fp8", kv_caches_mixed=False)

    GPUModelRunner.init_fp8_kv_scales(runner)

    assert all(t.tolist() == [[0.0, 0.0], [0.0, 0.0]] for t in runner.kv_caches)


def test_patched_idempotent():
    """Double call to _patch_init_fp8_kv_scales installs only once."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Clean slate
    if hasattr(GPUModelRunner, "_gguf_sleep_wake_patched"):
        delattr(GPUModelRunner, "_gguf_sleep_wake_patched")

    _patch_init_fp8_kv_scales()
    first = GPUModelRunner.init_fp8_kv_scales
    _patch_init_fp8_kv_scales()  # second call -> no-op
    second = GPUModelRunner.init_fp8_kv_scales

    assert first is second, "second call must not re-wrap"

    # Patched function still works
    runner = _make_fake_runner(cache_dtype="fp8", kv_caches_mixed=True)
    GPUModelRunner.init_fp8_kv_scales(runner)
    assert runner.kv_caches[0].tolist() == [[0.0, 0.0], [0.0, 0.0]]
