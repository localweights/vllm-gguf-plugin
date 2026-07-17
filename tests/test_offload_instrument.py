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


def test_strided_view_row_in_bounds_and_scalar_wrong_block_caught(tmp_path):
    """View-footprint ranges: whole-row transfers of a strided view into a
    shared region validate in-bounds (the 2026-07-16 false-positive class),
    while a wrong-block scalar op past the view's footprint is caught even
    though it stays inside the same backing storage (which the old
    untyped_storage extent would have masked)."""
    base = torch.zeros(10 * 64, dtype=torch.uint8)  # shared region
    # view: 4 blocks x 16B rows, row_stride 64 -> footprint 3*64+16 = 208B
    view = torch.as_strided(base, size=(4, 16), stride=(64, 1))
    handler = SimpleNamespace(
        src_tensors=[view],
        dst_tensors=[torch.zeros(256, dtype=torch.uint8)],
        gpu_to_cpu=True,
        src_block_size_factor=1,
        dst_block_size_factor=1,
    )
    calls = []

    def real_swap(src, dst, sizes):
        calls.append(len(src))

    wrapped = wrap_swap_blocks(handler, real_swap)
    dst_ptr = handler.dst_tensors[0].data_ptr()

    # all 4 whole-row ops, including last row (would OOB under
    # data_ptr+numel*elemsize = 64B extent) -> must pass
    src_ptrs = np.array([view.data_ptr() + b * 64 for b in range(4)], dtype=np.uint64)
    dst_ptrs = np.array([dst_ptr + b * 16 for b in range(4)], dtype=np.uint64)
    sizes = np.full(4, 16, dtype=np.uint64)
    wrapped(torch.from_numpy(src_ptrs), torch.from_numpy(dst_ptrs), torch.from_numpy(sizes))
    assert calls == [4]

    # wrong-block scalar: 4B read at block index 5 (past the 4-block view
    # footprint) but still inside the 640B backing storage
    log = tmp_path / "instr.jsonl"
    os.environ["VLLM_GGUF_OFFLOAD_INSTRUMENT_LOG"] = str(log)
    try:
        bad_src = np.array([view.data_ptr() + 5 * 64], dtype=np.uint64)
        with pytest.raises(RuntimeError, match="bounds violation"):
            wrapped(
                torch.from_numpy(bad_src),
                torch.from_numpy(np.array([dst_ptr], dtype=np.uint64)),
                torch.from_numpy(np.array([4], dtype=np.uint64)),
            )
        rec = json.loads(log.read_text().splitlines()[-1])
        assert rec["offending_ops"][0]["size"] == 4
    finally:
        os.environ.pop("VLLM_GGUF_OFFLOAD_INSTRUMENT_LOG", None)
