# SPDX-License-Identifier: Apache-2.0
"""P0 MTP-phase-2 spec-shape plan/prepare caching (narrow safe subset of B1+B2).

Profiling (`.council/mtp-phase2/spec.md`, P0 4-way matrix) showed, for MTP
k=1 27B decode at 8k context: `_prepare_inputs` ~50% self CPU, FlashInfer
attention-metadata build/replan ~27% CPU, vs an 83% GPU-bound MTP-off
baseline. The stated hypothesis was that spec decode's per-step shape
(batch, k) is CONSTANT in steady state and the FlashInfer plan could be
reused across steps keyed on (batch composition, qo_indptr shape,
page-table version).

READING THE 0.25 CODE (vllm/v1/attention/backends/flashinfer.py,
`FlashInferMetadataBuilder.build` / `._compute_flashinfer_kv_metadata`,
and `fast_plan_decode`) shows that assumption does NOT hold for the
FlashInfer-native decode path used by this hybrid model on Ampere/Ada
(the TRTLLM-gen decode API path -- `decode_with_flashinfer_trtllm_api`
-- is NOT taken here, since it skips paged_kv_indices/plan() entirely
and constructs a bare dataclass from `block_tables`/`seq_lens`; the
profiled 27% replan bucket is only reachable via the FI-native path,
confirming that path is active):

  ``_compute_flashinfer_kv_metadata`` (flashinfer.py:1000) recomputes
  ``paged_kv_last_page_len`` every call as::

      paged_kv_last_page_len_np = seq_lens_np % page_size

  ``seq_lens_np`` grows by the ACTUAL accepted-token count (1..1+k)
  every single decode step -- that is the whole point of decoding.
  ``last_page_len`` therefore changes on essentially every step
  regardless of whether the request set, k, or block-table page count
  is unchanged. ``fast_plan_decode`` (called immediately after, feeding
  ``indptr_cpu``/``last_page_len_cpu`` straight into the FlashInfer
  kernel-side plan) therefore CANNOT be safely skipped or memoized on a
  "constant spec shape" key alone: full B2 (skip the FlashInfer .plan()
  call itself across steps) would serve a stale KV-length view to the
  decode kernel and silently corrupt output. This contradicts the
  spec's stated invalidation key ("page-table version" does not track
  ``last_page_len``, which mutates on every accepted token, not just on
  new-page allocation) -- B2 as specced is therefore NOT IMPLEMENTED.

  Similarly, ``_prepare_inputs`` (gpu_model_runner.py:1914) already runs
  its OWN GPU-side incremental accepted-count correction for async spec
  decode (`update_num_computed_tokens_for_batch_change`, positions/
  seq_lens updated via GPU kernels keyed off the previous step's
  ``valid_sampled_token_count_gpu`` -- the actual sampler-verified
  accepted count, not an assumed k). Re-deriving that machinery in a
  plugin-side monkeypatch would duplicate correctness-critical GPU
  bookkeeping vLLM already performs safely; per the spec's own
  pragmatism rule this repo does NOT attempt a broad B1 rewrite of
  ``_prepare_inputs``.

SAFE SUBSET ACTUALLY IMPLEMENTED (B2-narrow):
  Inside ``_compute_flashinfer_kv_metadata``, the ``paged_kv_indices``
  GPU tensor (built by a Triton kernel copying block-table page ids)
  and the ``paged_kv_indptr`` GPU tensor (cumsum of per-request block
  counts + an async H2D copy) depend ONLY on ``num_blocks_np`` (how
  many pages each request currently owns) and the block-table tensor's
  page-id contents -- NOT on ``last_page_len``. Both are byte-for-byte
  reusable across steps where no request has crossed a page boundary
  since the last call, i.e. where:
    - ``num_reqs`` is unchanged, AND
    - ``num_blocks_np`` (per-request block/page counts) is unchanged, AND
    - the block-table tensor's torch ``._version`` counter is unchanged
      (a monotonic, torch-enforced guarantee: unchanged version is a
      hard proof the tensor's contents did not change since we last
      read it -- this cannot pass on stale data).
  When all three hold, this patch skips the Triton kernel launch
  (``_copy_page_indices_kernel``) and the indptr H2D copy, and reuses
  the previous step's GPU tensors directly. ``paged_kv_last_page_len``
  is ALWAYS recomputed fresh (cheap numpy op, correctness-critical,
  changes every step) -- it is never cached.

  This is bounded, low-risk, and skips real per-step GPU-kernel-launch
  + copy overhead on the (common, since pages are typically tens of
  tokens) steps where no new page was allocated. It is a fraction of
  the profiled 27% bucket, not the dominant cost inside it (the
  FlashInfer `.plan()` call itself, gated above as unsafe to skip).

Gated by env var GGUF_PLUGIN_PLAN_CACHE (default "1" / on). Set to "0"
to fully disable and fall back to unpatched upstream behavior.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any

import numpy as np
import torch

ENV_VAR = "GGUF_PLUGIN_PLAN_CACHE"


def is_enabled() -> bool:
    return os.environ.get(ENV_VAR, "1") != "0"


class KVIndicesCache:
    """Per-builder-instance cache for the page-id/indptr GPU tensors
    produced by ``_compute_flashinfer_kv_metadata``.

    Correctness is enforced by the cache KEY, not by any external
    "eligibility" signal: a cache hit requires ``num_reqs``,
    ``num_blocks_np`` (as bytes), and the block-table tensor's torch
    ``._version`` to all be bit-identical to the invocation that
    produced the cached value. Any of those changing (new page
    allocated, request added/removed/reordered, or literally any other
    in-place write ever made to the block-table tensor) forces a miss
    and a full recompute -- there is no path for this cache to return
    stale data.
    """

    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._indices: torch.Tensor | None = None
        self._indptr: torch.Tensor | None = None
        self._num_actual_pages: int | None = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        num_reqs: int,
        num_blocks_np: np.ndarray,
        block_table_tensor: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            num_reqs,
            num_blocks_np[:num_reqs].tobytes(),
            block_table_tensor._version,
        )

    def lookup(self, key: tuple[Any, ...]):
        if self._key is not None and key == self._key:
            self.hits += 1
            return self._indices, self._indptr, self._num_actual_pages
        self.misses += 1
        return None

    def store(
        self,
        key: tuple[Any, ...],
        indices: torch.Tensor,
        indptr: torch.Tensor,
        num_actual_pages: int,
    ) -> None:
        self._key = key
        self._indices = indices
        self._indptr = indptr
        self._num_actual_pages = num_actual_pages

    def invalidate(self) -> None:
        self._key = None
        self._indices = None
        self._indptr = None
        self._num_actual_pages = None


def install() -> None:
    """Monkeypatch ``FlashInferMetadataBuilder._compute_flashinfer_kv_metadata``
    to reuse ``paged_kv_indices``/``paged_kv_indptr`` GPU tensors when the
    key described in ``KVIndicesCache`` is unchanged. No-op if
    GGUF_PLUGIN_PLAN_CACHE=0 or if already installed.
    """
    if not is_enabled():
        return

    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    if getattr(
        FlashInferMetadataBuilder, "_gguf_kv_indices_cache_patched", False
    ):
        return

    _orig_compute = FlashInferMetadataBuilder._compute_flashinfer_kv_metadata

    @wraps(_orig_compute)
    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        cache: KVIndicesCache | None = getattr(self, "_gguf_kv_indices_cache", None)
        if cache is None:
            cache = KVIndicesCache()
            self._gguf_kv_indices_cache = cache

        key = KVIndicesCache.make_key(num_reqs, num_blocks_np, block_table_tensor)
        hit = cache.lookup(key)

        # paged_kv_last_page_len ALWAYS recomputed fresh -- correctness
        # critical, changes essentially every decode step.
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True
        )

        if hit is not None:
            indices, indptr, num_actual_pages = hit
            # Buffers are the persistent self.paged_kv_indices/indptr
            # tensors -- their content is unchanged since the version
            # check passed, so the views below are still correct.
            return self.paged_kv_indices.gpu[:num_actual_pages]

        # Miss: fall back to the original (correct, unmodified) path.
        result = _orig_compute(
            self, num_blocks_np, seq_lens_np, block_table_tensor, num_reqs, page_size
        )
        num_actual_pages = int(self.paged_kv_indptr.np[num_reqs])
        cache.store(
            key,
            self.paged_kv_indices.gpu[:num_actual_pages],
            self.paged_kv_indptr.gpu[: num_reqs + 1],
            num_actual_pages,
        )
        return result

    FlashInferMetadataBuilder._compute_flashinfer_kv_metadata = (
        _compute_flashinfer_kv_metadata
    )
    FlashInferMetadataBuilder._gguf_kv_indices_cache_patched = True
