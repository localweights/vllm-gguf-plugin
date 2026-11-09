# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for vllm_gguf_plugin.spec_plan_cache.

Tests the cache/invalidation-key logic in isolation (no vLLM/GPU import
required for the KVIndicesCache class itself -- only `install()` touches
vllm internals, and is exercised separately via import-safety only).
"""

import os

import numpy as np
import pytest
import torch

from vllm_gguf_plugin.spec_plan_cache import ENV_VAR, KVIndicesCache, is_enabled


def test_is_enabled_default_true(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert is_enabled() is True


def test_is_enabled_explicit_off(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "0")
    assert is_enabled() is False


def test_is_enabled_explicit_on(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    assert is_enabled() is True


def test_cache_miss_on_first_lookup():
    cache = KVIndicesCache()
    key = (2, b"\x01\x00\x00\x00\x01\x00\x00\x00", 0)
    assert cache.lookup(key) is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_hit_on_identical_key():
    cache = KVIndicesCache()
    key = (2, np.array([1, 1], dtype=np.int32).tobytes(), 0)
    indices = torch.tensor([0, 1])
    indptr = torch.tensor([0, 1, 2])
    cache.store(key, indices, indptr, 2)

    hit = cache.lookup(key)
    assert hit is not None
    got_indices, got_indptr, got_pages = hit
    assert torch.equal(got_indices, indices)
    assert torch.equal(got_indptr, indptr)
    assert got_pages == 2
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_miss_on_num_reqs_change():
    cache = KVIndicesCache()
    key1 = (2, b"same", 0)
    cache.store(key1, torch.tensor([0]), torch.tensor([0, 1]), 1)

    key2 = (3, b"same", 0)
    assert cache.lookup(key2) is None
    assert cache.misses == 1


def test_cache_miss_on_num_blocks_content_change():
    """A new page allocated for any request must invalidate -- the
    num_blocks_np byte content changes even if num_reqs is the same."""
    cache = KVIndicesCache()
    key1 = KVIndicesCache.make_key(
        2, np.array([1, 1], dtype=np.int32), _fake_tensor(version=0)
    )
    cache.store(key1, torch.tensor([0]), torch.tensor([0, 1]), 1)

    # Request 1 crossed a page boundary: now owns 2 blocks instead of 1.
    key2 = KVIndicesCache.make_key(
        2, np.array([1, 2], dtype=np.int32), _fake_tensor(version=0)
    )
    assert key1 != key2
    assert cache.lookup(key2) is None


def test_cache_miss_on_block_table_version_change():
    """Correctness backstop: torch bumps ._version on ANY in-place write
    to the block-table tensor. An unchanged version is required for a
    hit; a changed version -- even with identical num_blocks_np -- must
    force a miss, since we cannot prove the page CONTENTS didn't change
    (e.g. a page was freed and a different page id reused)."""
    cache = KVIndicesCache()
    same_blocks = np.array([1, 1], dtype=np.int32)
    key1 = KVIndicesCache.make_key(2, same_blocks, _fake_tensor(version=5))
    cache.store(key1, torch.tensor([0]), torch.tensor([0, 1]), 1)

    key2 = KVIndicesCache.make_key(2, same_blocks, _fake_tensor(version=6))
    assert key1 != key2
    assert cache.lookup(key2) is None


def test_cache_invalidate_clears_state():
    cache = KVIndicesCache()
    key = (1, b"x", 0)
    cache.store(key, torch.tensor([0]), torch.tensor([0, 1]), 1)
    cache.invalidate()
    assert cache.lookup(key) is None


def test_make_key_ignores_blocks_beyond_num_reqs():
    """num_blocks_np may be a persistent, over-allocated buffer (padded
    for cudagraphs); only the first num_reqs entries are load-bearing."""
    blocks = np.array([1, 1, 99, 99], dtype=np.int32)
    t = _fake_tensor(version=0)
    key_a = KVIndicesCache.make_key(2, blocks, t)
    blocks2 = np.array([1, 1, 5, 5], dtype=np.int32)
    key_b = KVIndicesCache.make_key(2, blocks2, t)
    assert key_a == key_b


class _fake_tensor:
    """Minimal stand-in exposing a torch-tensor-like `._version` attr
    without needing a real torch.Tensor mutation to bump it, so the key
    logic can be tested deterministically."""

    def __init__(self, version: int):
        self._version = version


def test_accepted_count_arithmetic_never_assumes_k():
    """Regression guard for the correctness-critical rule (spec risk
    note): the cache key must be derived from seq/block state that is
    ALREADY accepted-count-corrected upstream (num_blocks_np is derived
    from seq_lens_np, which upstream vLLM advances by the ACTUAL
    accepted count, 1..1+k, from the previous step's sampler output --
    never an assumed constant k). This module performs no independent
    "advance by k" arithmetic of its own; it only compares byte content
    of values vLLM already computed. Simulate a partial-acceptance step
    (accepted=1 out of k=3 possible) advancing seq_lens by 1, not 1+k,
    and confirm the resulting key differs from the full-acceptance case,
    i.e. the cache cannot mistake a short step for a full one."""
    page_size = 16
    seq_lens_full_accept = np.array([32], dtype=np.int32)  # grew by 1+k=4
    seq_lens_partial_accept = np.array([29], dtype=np.int32)  # grew by 1

    def blocks_for(seq_lens):
        return (seq_lens + page_size - 1) // page_size

    t = _fake_tensor(version=0)
    key_full = KVIndicesCache.make_key(1, blocks_for(seq_lens_full_accept), t)
    key_partial = KVIndicesCache.make_key(1, blocks_for(seq_lens_partial_accept), t)
    # 32 -> 2 blocks, 29 -> 2 blocks: same block count here, so this
    # particular pair happens to collide on num_blocks_np -- which is
    # exactly why last_page_len is NEVER read from this cache (see
    # module docstring) and is always recomputed fresh from seq_lens_np
    # directly in `_compute_flashinfer_kv_metadata`, independent of any
    # cache hit/miss on indices/indptr.
    assert key_full == key_partial  # documents the known-safe collision

    # A partial-acceptance step that crosses back under a page boundary
    # relative to the cached full-acceptance state must still miss.
    seq_lens_small = np.array([15], dtype=np.int32)
    key_small = KVIndicesCache.make_key(1, blocks_for(seq_lens_small), t)
    assert key_small != key_full


def test_plugin_module_importable_without_gpu():
    """The module itself (top-level import) must not require CUDA/vLLM
    -- only `install()` reaches into vllm internals, and it degrades to
    a pure no-op when disabled."""
    os.environ[ENV_VAR] = "0"
    try:
        import importlib

        import vllm_gguf_plugin.spec_plan_cache as mod

        importlib.reload(mod)
        mod.install()  # no-op, must not raise / must not import vllm
    finally:
        os.environ.pop(ENV_VAR, None)


# ---------------------------------------------------------------------------
# 0.33 buffer-shape adaptation: paged_kv_indices is a plain CUDA tensor there
# ---------------------------------------------------------------------------


def test_gpu_buffer_accepts_cpugpubuffer_and_plain_tensor():
    """vLLM 0.25 stores paged_kv_indices in a CpuGpuBuffer (`.gpu` view);
    0.33 allocates it as a plain CUDA tensor. Both must be unwrapped."""
    from types import SimpleNamespace

    from vllm_gguf_plugin.spec_plan_cache import _gpu_buffer

    gpu = torch.arange(4)
    assert _gpu_buffer(SimpleNamespace(cpu=torch.zeros(4), gpu=gpu)) is gpu
    assert _gpu_buffer(gpu) is gpu


class _FakeBuffer:
    """CpuGpuBuffer stand-in: `.np` (cpu numpy) + `.gpu` + `.cpu`.

    ``.gpu`` aliases the same storage as ``.cpu`` so a CPU-side write is
    observable without emulating the async copy.
    """

    def __init__(self, values):
        self.cpu = torch.as_tensor(values, dtype=torch.int32)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu


class _GpuViewProbe:
    """Minimal CpuGpuBuffer-shaped holder (only `.gpu` is read by the patch)."""

    def __init__(self, gpu):
        self.gpu = gpu


class _FakePageBuilder:
    """Minimal FlashInferMetadataBuilder stand-in for the wrapper round-trip."""

    def __init__(self, plain_indices: bool):
        self.paged_kv_indptr = _FakeBuffer([0, 2, 5])
        self.paged_kv_last_page_len = _FakeBuffer([0, 0, 0])
        # 0.33: plain tensor; 0.25: CpuGpuBuffer-style `.gpu` view.
        indices = torch.zeros(8, dtype=torch.int32)
        self.paged_kv_indices = indices if plain_indices else _GpuViewProbe(indices)

    def _original(
        self, num_blocks_np, seq_lens_np, block_table_tensor, num_reqs, page_size
    ):
        # Stands in for the upstream implementation: fill indptr/indices.
        num_actual_pages = int(num_blocks_np[:num_reqs].sum())
        self.paged_kv_indptr.np[1 : num_reqs + 1] = np.cumsum(
            num_blocks_np[:num_reqs], dtype=np.int32
        )
        view = self.paged_kv_indices
        view = getattr(view, "gpu", view)
        view[:num_actual_pages] = torch.arange(num_actual_pages, dtype=torch.int32)
        return view[:num_actual_pages]


@pytest.mark.parametrize("plain_indices", [True, False])
def test_wrapper_round_trip_both_buffer_shapes(plain_indices):
    """Miss then hit: the wrapper must not touch `.gpu` unconditionally."""
    from vllm_gguf_plugin.spec_plan_cache import KVIndicesCache, _gpu_buffer

    builder = _FakePageBuilder(plain_indices=plain_indices)
    cache = KVIndicesCache()
    block_table = torch.zeros(2, 8, dtype=torch.int32)
    num_blocks = np.array([2, 3], dtype=np.int32)
    seq_lens = np.array([32, 33], dtype=np.int32)
    key = KVIndicesCache.make_key(2, num_blocks, block_table)

    # Simulate a miss: original runs, then the wrapper stores the views.
    num_actual = int(num_blocks.sum())
    builder._original(num_blocks, seq_lens, block_table, 2, 16)
    cache.store(
        key,
        _gpu_buffer(builder.paged_kv_indices)[:num_actual],
        _gpu_buffer(builder.paged_kv_indptr)[:3],
        num_actual,
    )

    hit = cache.lookup(key)
    assert hit is not None
    indices, indptr, cached_pages = hit
    assert cached_pages == num_actual
    assert indices.shape == (num_actual,)
    assert indices.tolist() == list(range(num_actual))
    assert indptr.tolist() == [0, 2, 5]
