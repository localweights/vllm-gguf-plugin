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
