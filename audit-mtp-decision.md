# MTP patch decision for vLLM 0.33.0 (T-3 + T-4 merge)

Decision record for `mtp_enable.py`, `mtp_eagle_group.py`, `mtp_dynamic_skip.py`,
`spec_plan_cache.py` and branch commit `c830a77` against vLLM 0.33.0
(venv `/mnt/scratch/vllm033-venv`, git sha d7b8be8 base).
Model Runner V2 (`vllm/v1/worker/gpu/`) is the default runner in 0.33; MRV1
(`vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/spec_decode/`) still ships.

Line numbers are for the installed 0.33.0 tree
(`/mnt/scratch/vllm033-venv/lib/python3.12/site-packages/vllm/`).

| file | decision | why / 0.33 code it relies on or replaces |
| --- | --- | --- |
| `mtp_eagle_group.py` | **DELETE** | 0.33 marks the MTP draft KV group as `is_eagle_group` natively. Draft attention layers are flagged `is_draft_layer = True` by the MRV2 speculator (`v1/worker/gpu/spec_decode/speculator.py:240-242`) or MRV1's proposer (`v1/spec_decode/llm_base_proposer.py:1334-1335`); `get_kv_cache_spec` turns that into `KVCacheSpec.is_draft = True` (`v1/worker/gpu/attn_utils.py:150-151`); `get_global_kv_cache_groups` then sets `group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)` under `speculative_config.use_eagle_block_drop()` (`v1/core/kv_cache_planning.py:1650-1656`), and the engine calls it at `v1/engine/core.py:292`. `use_eagle_block_drop()` is True for `method="mtp"` with defaults (`config/speculative.py:2124-2150`), so the Mamba-shared-prefix-checkpoint path (`MambaPrefillCheckpointBuilder.drop_eagle_block`, `model_executor/layers/mamba/checkpoint.py:104-111`) gets exactly the "EAGLE on the draft group" semantics the old patch provided. Verified behaviourally in `tests/test_mtp_eagle_group.py` (`get_global_kv_cache_groups` marks only the `mtp.layers.*` group). The old 0.25 fallback (offloading scheduler flagging ALL groups when none is marked) is also gone: 0.33 logs and treats all groups as non-draft (`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:214-232`), so deleting the patch cannot re-introduce the spill bug. |
| `mtp_eagle_group.py` — helper `annotate_mtp_eagle_groups` | **DELETE** | Pure helper for the deleted wrapper; nothing else imports it (grep: only `plugin.py` and the patch's own tests). The intent is preserved and re-tested against the native path. |
| `c830a77` (`origin/fix-mtp-eagle-group`, "fix(offload): mark Qwen3.5 MTP draft KV group is_eagle_group") | **DELETE (absorbed upstream)** | That commit *is* `mtp_eagle_group.py` + its `plugin.py` wiring + tests, i.e. the row above. Its only functional content — annotating the MTP draft group — is native in 0.33 (evidence above). Its target symbol `vllm.v1.core.kv_cache_utils.get_kv_cache_groups` no longer exists at all (moved to `v1/core/kv_cache_planning.py:1113`), which is also why `plugin.register()` crashed with `AttributeError` on 0.33. |
| `mtp_enable.py` | **ADAPT (keep)** | Config rewrite is now redundant but harmless, and the patch is still needed for the GGUF weight map. Changes: (a) `SpeculativeConfig.hf_config_override` gained a second parameter in 0.33 — `(hf_config, transformers_mtp_overrides=None)` (`config/speculative.py:711-717`) — the wrapper now accepts it and forwards it to the original (detected via `inspect.signature`, so the 0.25 one-arg form still works). Without this the 0.33 caller `compose_draft_hf_overrides` (`config/speculative.py:1178-1204`) raises `TypeError`. (b) The `qwen3_5_text` → `qwen3_5_mtp` rewrite it performed is native in 0.33 (`config/speculative.py:992-1007`), so the wrapper is kept as the same superset the lane always used. (c) The `__post_init__` `model_weights` propagation is now also native — 0.33 builds the MTP draft from `target_model_config.model_weights` (`config/speculative.py:1253-1265`) — kept as a no-op fallback for the 0.25 image. (d) The `GGUFWeightsAdapter.build_name_map` override (GGUF tensor names for the `blk.<L>.nextn.*` head) is untouched: 0.33 has no GGUF path for `qwen3_5_mtp` (Qwen3_5MTP only loads HF/NVFP4-style names, `model_executor/models/qwen3_5_mtp.py:216-330`), so this is still load-bearing. (e) The `_maybe_share_embeddings` / `_maybe_share_lm_head` AttributeError guards target the MRV1 proposer (`v1/spec_decode/llm_base_proposer.py`); under MRV2 sharing happens in `load_eagle_model`/`_should_share` (`v1/worker/gpu/spec_decode/eagle/utils.py:66-134`), which never touches `.weight` for a draft that does not declare `has_own_embed_tokens`/`has_own_lm_head`, so the GGUF-quantized target no longer needs the guard there. Harmless on both runners, kept for the 0.25 image. |
| `mtp_dynamic_skip.py` | **ADAPT (keep, retargeted to MRV2)** | MRV1 wrapper (`SpecDecodeBaseProposer.propose`, `v1/spec_decode/llm_base_proposer.py:...`) is unchanged — that class exists in 0.33 with an identical signature. But under MRV2 the drafter is a `Speculator`: `MTPSpeculator(TargetDependentARSpeculator)` (`v1/worker/gpu/spec_decode/mtp/speculator.py`), selected by `init_speculator` (`v1/worker/gpu/spec_decode/__init__.py:70-72`). Its `propose` still runs the full draft prefill on a k=0 step and only skips *sampling* (`target_dependent_ar/speculator.py:287-297`: `skip_sample=num_speculative_tokens == 0`, then `draft_tokens[:num_reqs].fill_(-1)`), and the caller rewrites the result anyway (`v1/worker/gpu/model_runner.py:2328-2329`). A second wrapper now short-circuits that forward when the dynamic schedule resolves k=0, mirroring the original's k=0 return exactly, and never during `dummy_run`/`is_profile` (cudagraph capture must replay the real forward). |
| `spec_plan_cache.py` | **ADAPT (keep)** | Patch target `FlashInferMetadataBuilder._compute_flashinfer_kv_metadata` still exists with an identical signature (`v1/attention/backends/flashinfer.py:1511-1518`), but the persistent buffer it caches changed shape: 0.25 kept `self.paged_kv_indices` in a `CpuGpuBuffer` (`.gpu` view), 0.33 allocates a plain CUDA tensor (`v1/attention/backends/flashinfer.py:1119-1121`). The wrapper now unwraps through `_gpu_buffer()` (`getattr(buf, "gpu", buf)`), so the cache store/hit path works on both. `paged_kv_indptr` / `paged_kv_last_page_len` are still `CpuGpuBuffer`s, and `_copy_page_indices_kernel` still fills `paged_kv_indices[:num_actual_pages]` (`:1532-1541`), so the cached-tensor invariant (num_reqs + per-request block counts + block-table `._version`) is unchanged. |

## Not changed (verified as still needed / still valid)

* `spec_plan_cache.py` env kill-switch (`GGUF_PLUGIN_PLAN_CACHE=0`) and its
  correctness argument — unchanged upstream semantics.
* The MTP weight map in `mtp_enable.py` (`blk.<L>.nextn.*` → `mtp.*`), the
  qwen3.5 config mapping (`qwen35_config.py`) and `embed_quant_patch.py`
  (`VocabParallelEmbedding.__init__` gained keyword-only params in 0.33; the
  patch binds through `inspect.signature`, so it adapts automatically).

## Gate evidence

```
$ CUDA_VISIBLE_DEVICES="" /mnt/scratch/vllm033-venv/bin/python tools/check_patch_targets.py vllm_gguf_plugin
| vllm.v1.core.kv_cache_utils.get_kv_cache_groups | ... | (gone) |   # removed with the patch
SUMMARY targets=21 missing=0 skipped=2 inert=0 guarded=1     # 0.33, exit 0

$ docker run --rm -v $PWD:/p --entrypoint python3 \
    ghcr.io/localweights/llmportal-vllm:r2-0.25.0-gguf12 /p/tools/check_patch_targets.py /p/vllm_gguf_plugin
SUMMARY targets=21 missing=0 skipped=2 inert=0 guarded=1     # 0.25, exit 0
#   (on 0.25 the one `guarded` row is TargetDependentARSpeculator.propose — MRV2
#    does not exist there; on 0.33 the one `inert` row is the fp8 sleep/wake hook)

$ CUDA_VISIBLE_DEVICES="" /mnt/scratch/vllm033-venv/bin/python -c "import vllm_gguf_plugin as p; p.register()"
[mtp_enable] qwen3_5 MTP speculative override INSTALLED
register ok
```

No GPU was used for any of the above (the plugin's MTP path is exercised on CPU
via stubs; GPU validation is Phase A Gate A, owned by the chair).
