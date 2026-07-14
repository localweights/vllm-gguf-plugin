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
"""

from functools import wraps

import torch


def install() -> None:
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
