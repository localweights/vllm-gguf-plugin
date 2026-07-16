# SPDX-License-Identifier: Apache-2.0
"""Offload transfer bounds validator — monkeypatch for SingleDirectionOffloadingHandler.

Intercepts _swap_blocks_batch to validate every copy descriptor against
the true tensor address ranges before the kernel launches.  A bad
descriptor raises RuntimeError with a JSONL evidence log — converting
silent corruption (Xid 31) into clean evidence.

Kill-switch: VLLM_GGUF_OFFLOAD_INSTRUMENT=0 disables the wrapper.
Log path: VLLM_GGUF_OFFLOAD_INSTRUMENT_LOG env (default
          /mnt/scratch/llmportal/logs/offload-instrument.jsonl).
"""

import json
import logging
import os
from datetime import datetime, timezone
from functools import wraps

import numpy as np

logger = logging.getLogger(__name__)


def _compute_ranges(tensors):
    """Build (lo, hi) uint64 array from a list of torch Tensors."""
    ranges = np.array(
        [
            (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
            for t in tensors
        ],
        dtype=np.uint64,
    )
    return ranges


def _tensor_ids(tensors):
    """Tuple of object ids for identity tracking."""
    return tuple(id(t) for t in tensors)


def wrap_swap_blocks(handler, fn):
    """Wrap a _swap_blocks_batch function with bounds validation.

    Returns a validator-wrapped version of *fn* that checks src/dst
    pointers against *handler*.src_tensors and *handler*.dst_tensors
    before delegating.  If env VLLM_GGUF_OFFLOAD_INSTRUMENT=0 returns
    *fn* unchanged.

    Exported for direct use in tests without the real vllm class.
    """
    if os.environ.get("VLLM_GGUF_OFFLOAD_INSTRUMENT", "1") == "0":
        return fn

    # Cache ranges on the handler instance.
    handler._offload_instr_src_ranges = _compute_ranges(handler.src_tensors)
    handler._offload_instr_dst_ranges = _compute_ranges(handler.dst_tensors)
    handler._offload_instr_tensor_ids = _tensor_ids(handler.src_tensors + handler.dst_tensors)

    # vLLM calls with extra kwargs (is_src_access_order_any=...) — pass through.
    def validated(src, dst, sizes, *args, **kwargs):
        # --- rebuild range cache if tensor identity changed ---
        current_ids = _tensor_ids(handler.src_tensors + handler.dst_tensors)
        if current_ids != handler._offload_instr_tensor_ids:
            handler._offload_instr_src_ranges = _compute_ranges(handler.src_tensors)
            handler._offload_instr_dst_ranges = _compute_ranges(handler.dst_tensors)
            handler._offload_instr_tensor_ids = current_ids

        src_np = src.numpy()
        dst_np = dst.numpy()
        sizes_np = sizes.numpy()
        n_ops = len(src_np)

        src_ranges = handler._offload_instr_src_ranges
        dst_ranges = handler._offload_instr_dst_ranges

        # --- vectorised bounds check (no Python loop over ops) ---
        def _check(ptrs, sizes, ranges):
            if len(ranges) == 0:
                return np.zeros(len(ptrs), dtype=bool)
            lo = ranges[:, 0][None, :]   # (1, n_ranges)
            hi = ranges[:, 1][None, :]   # (1, n_ranges)
            p = ptrs[:, None]            # (n_ops, 1)
            e = (ptrs + sizes)[:, None]  # (n_ops, 1)
            in_range = (lo <= p) & (e <= hi)  # (n_ops, n_ranges)
            return in_range.any(axis=1)  # (n_ops,)

        src_ok = _check(src_np, sizes_np, src_ranges)
        dst_ok = _check(dst_np, sizes_np, dst_ranges)
        any_violation = ~(src_ok & dst_ok)

        if not any_violation.any():
            return fn(src, dst, sizes, *args, **kwargs)

        # --- violation: log evidence + raise ---
        log_path = os.environ.get(
            "VLLM_GGUF_OFFLOAD_INSTRUMENT_LOG",
            "/mnt/scratch/llmportal/logs/offload-instrument.jsonl",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        bad_indices = np.where(any_violation)[0]
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "gpu_to_cpu": bool(handler.gpu_to_cpu) if hasattr(handler, "gpu_to_cpu") else None,
            "num_ops": int(n_ops),
            "offending_indices": [int(i) for i in bad_indices[:20]],
            "offending_ops": [
                {
                    "idx": int(i),
                    "src": int(src_np[i]),
                    "dst": int(dst_np[i]),
                    "size": int(sizes_np[i]),
                }
                for i in bad_indices[:20]
            ],
            "src_ranges": [[int(lo), int(hi)] for lo, hi in src_ranges],
            "dst_ranges": [[int(lo), int(hi)] for lo, hi in dst_ranges],
            "src_block_size_factor": getattr(handler, "src_block_size_factor", None),
            "dst_block_size_factor": getattr(handler, "dst_block_size_factor", None),
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        raise RuntimeError("offload transfer bounds violation (see offload-instrument.jsonl)")

    return validated


def install() -> None:
    """Monkeypatch SingleDirectionOffloadingHandler.__init__.

    Wraps the instance's _swap_blocks_batch with a bounds validator
    after the original __init__ runs.  Idempotent via the
    _gguf_offload_instrument_patched guard.
    """
    try:
        from vllm.v1.kv_offload.cpu.gpu_worker import (
            SingleDirectionOffloadingHandler,
        )
    except ImportError:
        logger.info("vllm not available — offload instrument not installed")
        return

    if getattr(SingleDirectionOffloadingHandler, "_gguf_offload_instrument_patched", False):
        return

    original_init = SingleDirectionOffloadingHandler.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._swap_blocks_batch = wrap_swap_blocks(self, self._swap_blocks_batch)
        logger.info(
            "offload instrumentation armed (n_src_tensors=%d, n_dst_tensors=%d)",
            len(self.src_tensors),
            len(self.dst_tensors),
        )

    SingleDirectionOffloadingHandler.__init__ = patched_init
    SingleDirectionOffloadingHandler._gguf_offload_instrument_patched = True
