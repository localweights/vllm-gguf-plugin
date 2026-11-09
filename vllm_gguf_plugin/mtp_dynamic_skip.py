# SPDX-License-Identifier: Apache-2.0
"""Skip the drafter forward entirely when Dynamic SD resolves k=0.

Root cause (vllm 0.25, v1/spec_decode/llm_base_proposer.py):
``SpecDecodeBaseProposer.propose`` runs the full draft-model forward
(``self.model(**model_kwargs)``, ~line 592) UNCONDITIONALLY, and only
checks ``self.num_speculative_tokens == 0`` AFTER the forward (~line 610),
returning an empty ``(batch, 0)`` tensor. The stated reason is keeping the
drafter KV cache in sync. For a MoE draft head (Qwen3.6-35B-A3B nextn,
256 experts) that discarded forward costs ~40% of aggregate 2-stream
throughput; for the 27B dense head ~9%.

This patch wraps ``SpecDecodeBaseProposer.propose`` so that when the
per-step ``num_speculative_tokens`` resolved from the dynamic schedule
(``num_speculative_tokens_per_batch_size``) is 0, it returns the same
empty ``(batch, 0)`` int64 tensor immediately — no drafter attention
metadata build, no first-pass input prep, no draft forward, no draft
sampling. It is a no-op (delegates to the original) whenever:
  * the speculative config does not use a dynamic schedule, or
  * the resolved k for this step is > 0.

Semantics:
  * k>0 steps are byte-identical to upstream (original propose runs).
  * k=0 steps return exactly what upstream returns today (empty tensor);
    the scheduler already schedules 0 spec tokens for these steps, so the
    verify path is unchanged.
  * ``_last_draft_probs`` is cleared on the skip so a stale probs tensor
    from a previous k>0 step can never be re-consumed by
    ``take_last_draft_probs``.

Known constraint (documented, accepted): skipping the forward means the
drafter's own KV cache (and, for hybrid draft layers, recurrent state) is
NOT advanced over tokens decoded during k=0 steps. After a 0 -> k>0
transition the drafter attends over an unwritten span, so proposal
quality (acceptance rate) may degrade for affected sequences. Output
correctness is unaffected — every proposal is verified by the target
model's rejection sampler; bad proposals are simply rejected, degrading
to plain decode speed. With the production schedule
([[1,1,1],[2,N,0]]: MTP only at batch size 1) the alternative baseline is
running with MTP off entirely, which also has no drafter KV — so the
transition penalty is bounded by the no-MTP case. The runner-side prep in
``GPUModelRunner.propose_draft_token_ids`` (prepare_next_token_ids_padded,
valid_sampled_token_count copies) is intentionally NOT skipped: the async
spec-decode bookkeeping of the NEXT step consumes those counts whenever
THIS step verified drafts from the previous step.

vLLM >= 0.29 (Model Runner V2, default since 0.29): the drafter is a
``Speculator`` (vllm/v1/worker/gpu/spec_decode/), not
``SpecDecodeBaseProposer``.  ``TargetDependentARSpeculator.propose`` — the base
of ``MTPSpeculator`` — still runs the full draft prefill forward on a k=0 step
and only skips *sampling* (``skip_sample=num_speculative_tokens == 0``, followed
by ``draft_tokens[:num_reqs].fill_(-1)``).  ``install()`` therefore wraps both
classes: MRV1's proposer (kept for the 0.25 image) and the MRV2 speculator, with
the same guard (dynamic schedule resolved k == 0, never a dummy/profile run).
"""

from functools import wraps

import torch


def _install_mrv1() -> None:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    if getattr(SpecDecodeBaseProposer, "_gguf_dynamic_k0_skip_patched", False):
        return

    _orig_propose = SpecDecodeBaseProposer.propose

    @wraps(_orig_propose)
    def propose(self, num_speculative_tokens, *args, **kwargs):
        spec_config = getattr(self, "speculative_config", None)
        if (
            num_speculative_tokens == 0
            and spec_config is not None
            and spec_config.uses_dynamic_speculative_decoding()
        ):
            # Resolve tensors we need for shape/device without touching
            # anything expensive. Both are keyword args at the only call
            # site (GPUModelRunner.propose_draft_token_ids), but accept
            # positionals for robustness: propose(self, k, target_token_ids,
            # target_positions, target_hidden_states, next_token_ids,
            # token_indices_to_sample, common_attn_metadata, ...).
            common_attn_metadata = kwargs.get("common_attn_metadata")
            if common_attn_metadata is None and len(args) >= 6:
                common_attn_metadata = args[5]
            next_token_ids = kwargs.get("next_token_ids")
            if next_token_ids is None and len(args) >= 4:
                next_token_ids = args[3]

            batch_size = common_attn_metadata.batch_size()
            device = (
                next_token_ids.device
                if isinstance(next_token_ids, torch.Tensor)
                else "cpu"
            )
            # Mirror the state the original propose would leave behind on
            # its k==0 path (llm_base_proposer.py:522 and :610-617).
            self.num_speculative_tokens = 0
            self._last_draft_probs = None
            return torch.empty(batch_size, 0, device=device, dtype=torch.int64)
        return _orig_propose(self, num_speculative_tokens, *args, **kwargs)

    SpecDecodeBaseProposer.propose = propose
    SpecDecodeBaseProposer._gguf_dynamic_k0_skip_patched = True


def _mrv2_step_flags(self, args, kwargs):
    """``(k, dummy_run, is_profile)`` for the MRV2 speculator ``propose``.

    The in-tree caller passes these as keywords (v1/worker/gpu/model_runner.py),
    so the keyword path is the fast one; positional indices are the fallback,
    from the signature
    ``propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
    aux_hidden_states, num_sampled, num_rejected, last_sampled,
    next_prefill_tokens, temperature, seeds, dp_sync_state=None,
    dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
    is_profile=False, num_speculative_tokens=None)``.
    ``k is None`` means "use self.num_speculative_steps", as in the original.
    """
    k = kwargs.get("num_speculative_tokens")
    if k is None and len(args) >= 16:
        k = args[15]
    dummy_run = kwargs.get("dummy_run") or (len(args) > 11 and args[11])
    is_profile = kwargs.get("is_profile") or (len(args) > 14 and args[14])
    if k is None:
        k = getattr(self, "num_speculative_steps", None)
    return k, bool(dummy_run), bool(is_profile)


def _install_mrv2() -> None:
    """Skip the drafter forward on dynamic-schedule k=0 steps (Model Runner V2)."""
    try:  # MRV2 speculators do not exist on vLLM <= 0.25
        from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
            TargetDependentARSpeculator,
        )
    except ImportError:
        return

    if getattr(
        TargetDependentARSpeculator, "_gguf_dynamic_k0_skip_patched", False
    ):
        return

    # patch-target: version-guarded — Model Runner V2 speculators exist only on
    # vLLM >= 0.29 (the import above is wrapped in try/except ImportError).
    _orig_propose = TargetDependentARSpeculator.propose

    @wraps(_orig_propose)
    def propose(self, input_batch, *args, **kwargs):
        spec_config = getattr(self, "speculative_config", None)
        k, dummy_run, is_profile = _mrv2_step_flags(self, args, kwargs)
        if (
            k == 0
            and spec_config is not None
            and spec_config.uses_dynamic_speculative_decoding()
            # Never skip inside a dummy/profiling/cudagraph-capture run: the
            # capture has to replay the real drafter forward.
            and not dummy_run
            and not is_profile
        ):
            # Mirror exactly what the original leaves behind on its k==0 path
            # (target_dependent_ar/speculator.py: `if num_speculative_tokens
            # <= 1: if == 0: draft_tokens[:num_reqs].fill_(-1); return ...`).
            num_reqs = input_batch.num_reqs
            self.draft_tokens[:num_reqs].fill_(-1)
            return self.draft_tokens[:num_reqs]
        return _orig_propose(self, input_batch, *args, **kwargs)

    TargetDependentARSpeculator.propose = propose
    TargetDependentARSpeculator._gguf_dynamic_k0_skip_patched = True


def install() -> None:
    _install_mrv1()
    _install_mrv2()
