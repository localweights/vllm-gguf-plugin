# PORT_NOTES_0.33 — plugin port to vLLM 0.33.0

Branch `ws2-vllm033`, base `d7b8be8`. Target: vLLM 0.33.0 with Model Runner V2
(default since 0.29) and torch 2.14 / CUDA 13. Plugin `register()` and every
monkeypatch target are verified on CPU only (`CUDA_VISIBLE_DEVICES=""`) — no GPU
was used for this port.

Gate:

```
CUDA_VISIBLE_DEVICES="" /mnt/scratch/vllm033-venv/bin/python tools/check_patch_targets.py vllm_gguf_plugin
SUMMARY targets=21 missing=0 skipped=2 inert=1 guarded=0        # 0.33, exit 0
# same tool inside the 0.25 image:
SUMMARY targets=21 missing=0 skipped=2 inert=0 guarded=1        # 0.25, exit 0
CUDA_VISIBLE_DEVICES="" /mnt/scratch/vllm033-venv/bin/python -c "import vllm_gguf_plugin; register()"
CUDA_VISIBLE_DEVICES="" /mnt/scratch/vllm033-venv/bin/python -m pytest tests -x -q -m "not slow" --ignore=tests/diffusion --ignore=tests/test_gguf_generation.py
```

## One line per change (old → new, why)

| # | change | why |
| --- | --- | --- |
| 1 | `sleep_wake_hybrid.py`: `from vllm.utils.torch_utils import is_quantized_kv_cache` → `from vllm.v1.kv_cache_interface import is_quantized_kv_cache` (module-level, `except ImportError` falls back to `vllm.utils.torch_utils` via `getattr`) | The helper moved: 0.33 defines it in `v1/kv_cache_interface.py`; the old location no longer exports it (the single MISSING-SYMBOL in `audit/imports-0.33.md`). Fallback keeps the r2-0.25.0-gguf12 image working; resolving it with `getattr` keeps `tools/audit_vllm_imports.py` clean on 0.33. |
| 2 | `sleep_wake_hybrid.py`: `GPUModelRunner.init_fp8_kv_scales` patch → guarded by `hasattr`, documented **inert-but-harmless** on 0.33 (sorted under `# patch-target: inert`) | 0.33's sleep-mode rework (MRV2 `v1/worker/gpu_worker.py` `sleep()`/`wake_up()` + `sleep_mode_backend.suspend/resume`) deleted `init_fp8_kv_scales` entirely: grep of the 0.33 tree finds no `init_fp8_kv_scales` and no `post_kv_cache_wake_up`, so no post-wake `kv_caches.zero_()` loop exists to crash on hybrid list entries. Teardown detaches caches through `clear_layer_kv_caches` (`v1/worker/utils.py:888-906`), which already handles list/tensor entries. Keeping the 0.25 wrapper intact preserves the old image. |
| 3 | `mtp_eagle_group.py` (and its `plugin.py` wiring, and the `annotate_mtp_eagle_groups` helper) → **deleted** | `vllm.v1.core.kv_cache_utils.get_kv_cache_groups` is gone in 0.33 (moved to `v1/core/kv_cache_planning.py:1113`) — this was the `register()` `AttributeError`. 0.33 annotates the MTP draft group natively from `KVCacheSpec.is_draft` in `get_global_kv_cache_groups` (`v1/core/kv_cache_planning.py:1650-1656`), fed by `is_draft_layer` (`v1/worker/gpu/spec_decode/speculator.py:240-242` → `v1/worker/gpu/attn_utils.py:150-151`). Proof + behaviour test: `audit-mtp-decision.md`, `tests/test_mtp_eagle_group.py`. |
| 4 | `mtp_enable.py`: `hf_config_override(hf_config)` → `hf_config_override(hf_config, transformers_mtp_overrides=None)`, forwarding the new argument to the original when the original accepts it | 0.33 calls it as `SpeculativeConfig.hf_config_override(hf_config, transformers_mtp_overrides=...)` (`config/speculative.py:711, 1178-1204`); the one-arg wrapper raised `TypeError` for every MTP boot. Detected via `inspect.signature`, so 0.25 keeps its one-arg call. **Pre-existing caveat, unchanged by this port:** the wrapper is a local closure, so `SpeculativeConfig.compose_draft_hf_overrides(None)` is not picklable (`Can't get local object`) on 0.25 *and* 0.33 alike — the current single-process serving topology never pickles it, so it is out of scope here; if Gate A ever trips on it, lift `hf_config_override` to module level. |
| 5 | `mtp_dynamic_skip.py`: MRV1-only wrapper → MRV1 wrapper + `TargetDependentARSpeculator.propose` wrapper (guarded `try/except ImportError` for 0.25) | Under MRV2 the drafter is `MTPSpeculator(TargetDependentARSpeculator)` (`v1/worker/gpu/spec_decode/mtp/speculator.py`), which still runs the full draft prefill forward on dynamic-schedule k=0 steps and only skips sampling (`target_dependent_ar/speculator.py:287-297`). The new wrapper preserves the fork's "no wasted drafter forward at k=0" win on the default runner; `dummy_run`/`is_profile` are never skipped. |
| 6 | `spec_plan_cache.py`: `self.paged_kv_indices.gpu[...]` → `_gpu_buffer(self.paged_kv_indices)[...]` (`getattr(buf, "gpu", buf)`) | In 0.25 `paged_kv_indices` is a `CpuGpuBuffer`; in 0.33 it is a plain CUDA tensor (`v1/attention/backends/flashinfer.py:1119-1121`), so the old `.gpu` access would `AttributeError` on the first cache store (i.e. on the first decode step). `paged_kv_indptr` / `paged_kv_last_page_len` are unchanged `CpuGpuBuffer`s. |
| 7 | `plugin.py`: dropped `_patch_eagle_group_annotation()` from `register()` | Follows (3). |
| 8 | `tests/test_kernels.py`: module-level `pytest.mark.skipif(not torch.cuda.is_available())` | Those tests launch GGUF CUDA kernels; the brief requires CUDA tests to skip cleanly when no GPU is visible instead of failing 1518 times (they failed the CPU gate before this). |
| 9 | `tools/check_patch_targets.py`: **new** gate tool | Statically finds every upstream callable the plugin replaces/wraps (attribute assignment, `setattr`, `_orig = Cls.method` reads; constant-valued marker assignments excluded; `except ImportError` fallbacks excluded) and resolves it in the current interpreter, printing `\| target \| file:line \| exists \| signature \|`. Exit 1 on any missing target. `# patch-target: inert` marks a deliberately-removed upstream symbol (item 2); `# patch-target: version-guarded` marks one that exists only on some supported vLLM versions (item 5, MRV2); optional roots like `vllm_omni` report `skip-absent`. |

## 0.25 vs 0.33 signature diff for every patch target

Produced by running the same tool in both interpreters
(`ghcr.io/localweights/llmportal-vllm:r2-0.25.0-gguf12` vs the 0.33 venv):

| target | 0.25 | 0.33 | action |
| --- | --- | --- | --- |
| `vllm.config.speculative.SpeculativeConfig.hf_config_override` | `(hf_config)` | `(hf_config, transformers_mtp_overrides=None)` | wrapper adapted (item 4) |
| `vllm.v1.core.kv_cache_utils.get_kv_cache_groups` | `(vllm_config, kv_cache_spec)` | **removed** (now `vllm.v1.core.kv_cache_planning.get_kv_cache_groups`, plus native draft annotation) | patch deleted (item 3) |
| `vllm.v1.worker.gpu_model_runner.GPUModelRunner.init_fp8_kv_scales` | `(self)` | **removed** | inert, guarded (item 2) |
| `vllm.v1.attention.backends.flashinfer.FlashInferMetadataBuilder._compute_flashinfer_kv_metadata` | `(self, num_blocks_np, seq_lens_np, block_table_tensor, num_reqs, page_size)` | identical | wrapper adapted for the buffer type (item 6) |
| `vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer.propose` | same 11 params | identical | wrapper kept as-is |
| `vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator.TargetDependentARSpeculator.propose` | **absent** (MRV2 does not exist) | `(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync_state=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False, num_speculative_tokens=None)` | new wrapper added (item 5), import guarded |
| `…SpecDecodeBaseProposer._maybe_share_embeddings` / `_maybe_share_lm_head` | `(self, target_language_model)` | identical | wrapper kept (MRV1 path) |
| `vllm.engine.arg_utils.EngineArgs.create_model_config` | `(self)` | identical | unchanged |
| `vllm.engine.arg_utils.maybe_override_with_speculators` | `(model, tokenizer, trust_remote_code, revision=None, vllm_speculative_config=None, hf_token=None, **kwargs)` | identical | unchanged |
| `vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__` | `(self, num_embeddings, embedding_dim, params_dtype=None, org_num_embeddings=None, padding_size=64, quant_config=None, prefix='')` | adds keyword-only `disable_tp`, `reduce_results`, `quant_method`, `parallel_group` | no change needed — the patch binds through `inspect.signature` |
| `vllm.v1.kv_offload.cpu.gpu_worker.SingleDirectionOffloadingHandler.__init__` | `(self, gpu_tensors, cpu_tensors, block_size_factor, kv_cache_groups_data_refs, gpu_to_cpu, mmap_region=None)` | `(self, gpu_tensors, cpu_tensors, blocks_per_chunk, layer_refs_per_group, gpu_to_cpu, canonical_layout=False, host_memory_is_pinned=True)` (fully rewritten class) | no change needed — the wrapper is `(self, *args, **kwargs)` and only wraps the instance's `_swap_blocks_batch`, which 0.33 still assigns in `__init__` (`:352`) and calls at `:710` with the same `(src, dst, sizes, …)` shape |
| `vllm.utils.torch_utils.direct_register_custom_op` | same | same | unchanged |
| `vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration.get_mamba_state_{shape,dtype}_from_config` / `get_mamba_state_copy_func` | present | present, same | unchanged |
| `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` | `(path, return_tensors=False, model_to_load=None, torch_dtype=None)` | identical | unchanged |
| `vllm_omni.*.DiffusersPipelineLoader.load_{weights,model}` | optional | optional (not installed in the scratch venv) | unchanged; patch already `try/except ImportError` |

## MRV2 audit of `vllm.v1.worker.gpu_model_runner` patches

Nothing in the non-MTP plugin patches MRV1 internals any more except
`sleep_wake_hybrid.py`, which is inert (item 2). The MTP `propose` retarget
(item 5) was the only MRV2 gap and is now covered on both runners. No code path
forces MRV1 (`grep -rn 'VLLM_USE_V1\|MRV1\|use_v2_model_runner *= *False'
vllm_gguf_plugin` → empty).

## Optional-package imports (spec A.2)

Already lazy and guarded before this change; verified, no edit needed:

* `vllm_omni.quantization.register_quantization_override` — inside
  `plugin._register_omni_diffusion_quantization()`, `except ImportError`.
* `vllm_omni.diffusion.model_loader.diffusers_loader.DiffusersPipelineLoader` —
  inside `weights_adapter/diffusion/integration._patch_diffusers_loader()`,
  `except ImportError`.
* `vllm.v1.kv_offload.cpu.gpu_worker.SingleDirectionOffloadingHandler` — inside
  `offload_instrument.install()`, `except ImportError`.

`import vllm_gguf_plugin` therefore only needs `vllm` (+ `transformers`, `gguf`,
`numpy`, `torch`), which is what `tests/test_plugin.py` asserts.
