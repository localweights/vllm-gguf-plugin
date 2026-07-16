# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for offload_instrument bounds validator.

Builds a minimal fake handler and calls wrap_swap_blocks directly.
"""

import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gguf_plugin.offload_instrument import wrap_swap_blocks


def _fake_handler(num_src: int, num_dst: int) -> SimpleNamespace:
    """Build a minimal handler stand-in with small CPU tensors."""
    src_tensors = [torch.zeros(4, dtype=torch.uint8) for _ in range(num_src)]
    dst_tensors = [torch.zeros(4, dtype=torch.uint8) for _ in range(num_dst)]
    return SimpleNamespace(
        src_tensors=src_tensors,
        dst_tensors=dst_tensors,
        gpu_to_cpu=True,
        src_block_size_factor=1,
        dst_block_size_factor=1,
    )


def _in_bounds_spec(handler) -> tuple:
    """Build src/dst/sizes tensors whose pointers fall within handler's ranges."""
    src_ptrs = np.array([t.data_ptr() for t in handler.src_tensors], dtype=np.uint64)
    dst_ptrs = np.array([t.data_ptr() for t in handler.dst_tensors], dtype=np.uint64)
    sizes = np.array([t.numel() * t.element_size() for t in handler.src_tensors], dtype=np.uint64)
    return (
        torch.from_numpy(src_ptrs),
        torch.from_numpy(dst_ptrs),
        torch.from_numpy(sizes),
    )


def _out_of_bounds_spec(handler) -> tuple:
    """Build src/dst/sizes with one bad src pointer outside any tensor range."""
    # Use the same tensors but off by an absurd amount
    src_ptrs = np.array([t.data_ptr() for t in handler.src_tensors], dtype=np.uint64)
    src_ptrs[0] = 0xDEAD0000  # clearly invalid
    dst_ptrs = np.array([t.data_ptr() for t in handler.dst_tensors], dtype=np.uint64)
    sizes = np.array([t.numel() * t.element_size() for t in handler.src_tensors], dtype=np.uint64)
    return (
        torch.from_numpy(src_ptrs),
        torch.from_numpy(dst_ptrs),
        torch.from_numpy(sizes),
    )


# ---- tests ----


def test_wrap_in_bounds():
    """In-bounds descriptors -> real fn called, no crash."""
    handler = _fake_handler(num_src=2, num_dst=2)
    called = False

    def real_swap(src, dst, sizes):
        nonlocal called
        called = True

    wrapped = wrap_swap_blocks(handler, real_swap)
    src, dst, sizes = _in_bounds_spec(handler)
    result = wrapped(src, dst, sizes)

    assert called, "real fn must be called"
    assert result is None, "real fn returned None"


def test_wrap_out_of_bounds_src():
    """Out-of-bounds src ptr -> RuntimeError, real fn NOT called, JSONL written."""
    handler = _fake_handler(num_src=2, num_dst=2)
    called = False

    def real_swap(src, dst, sizes):
        nonlocal called
        called = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        log_path = tmp.name

    try:
        os.environ["VLLM_GGUF_OFFLOAD_INSTRUMENT_LOG"] = log_path
        wrapped = wrap_swap_blocks(handler, real_swap)
        src, dst, sizes = _out_of_bounds_spec(handler)

        with pytest.raises(RuntimeError, match="offload transfer bounds violation"):
            wrapped(src, dst, sizes)

        assert not called, "real fn must NOT be called on violation"

        # Verify JSONL written
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1, "exactly one JSONL line written"
        record = json.loads(lines[0])
        assert record["gpu_to_cpu"] is True
        assert record["num_ops"] == 2
        assert len(record["offending_indices"]) == 1  # only first op is bad
        assert record["offending_indices"][0] == 0
        assert "src_ranges" in record
        assert "dst_ranges" in record
    finally:
        os.unlink(log_path)
        os.environ.pop("VLLM_GGUF_OFFLOAD_INSTRUMENT_LOG", None)


def test_kill_switch():
    """VLLM_GGUF_OFFLOAD_INSTRUMENT=0 -> wrap returns original fn unchanged."""
    handler = _fake_handler(num_src=2, num_dst=2)

    def real_swap(src, dst, sizes):
        return "original"

    os.environ["VLLM_GGUF_OFFLOAD_INSTRUMENT"] = "0"
    try:
        wrapped = wrap_swap_blocks(handler, real_swap)
        assert wrapped is real_swap, "kill-switch must return original fn"
    finally:
        os.environ.pop("VLLM_GGUF_OFFLOAD_INSTRUMENT", None)


def test_cache_recomputes_on_tensor_identity_change():
    """When handler tensor list changes identity, ranges are recomputed."""
    handler = _fake_handler(num_src=2, num_dst=2)
    call_count = 0

    def real_swap(src, dst, sizes):
        nonlocal call_count
        call_count += 1

    wrapped = wrap_swap_blocks(handler, real_swap)
    src1, dst1, sizes1 = _in_bounds_spec(handler)

    # First call builds cache
    wrapped(src1, dst1, sizes1)
    assert call_count == 1

    # Replace src tensor with new one (different id()) — recompute spec
    handler.src_tensors[0] = torch.zeros(4, dtype=torch.uint8)
    src2, dst2, sizes2 = _in_bounds_spec(handler)
    wrapped(src2, dst2, sizes2)
    assert call_count == 2, "must still work after tensor identity change"

    # Original spec should now be stale but we don't call it again
