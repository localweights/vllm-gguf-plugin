# SPDX-License-Identifier: Apache-2.0
"""Enable MTP (nextn) speculative decode for the qwen35/qwen35moe GGUF lane.

The GGUFs ship a baked MTP head (blk.64.nextn.* + blk.64 transformer tensors,
meta qwen35.nextn_predict_layers). vLLM natively supports qwen3_5 MTP
(Qwen3_5MTP / Qwen3_5MoeMTP), but auto-detection keys off the DRAFT config's
model_type being in MTPModelTypes and expects `qwen3_5` / `qwen3_5_moe`. Our
plugin produces bare text configs (model_type `qwen3_5_text` / `qwen3_5_moe_text`)
and never sets `mtp_num_hidden_layers`. This module bridges both gaps by:
  1. tagging the config with mtp_num_hidden_layers (from nextn_predict_layers),
  2. extending SpeculativeConfig.hf_config_override so a draft built from our
     gguf becomes model_type=qwen3_5_mtp / arch=Qwen3_5MTP(MoeMTP).

vLLM >= 0.33 rewrites ``qwen3_5``/``qwen3_5_text`` (and the ``_moe`` variants)
into ``qwen3_5_mtp`` natively (vllm/config/speculative.py, the
``hf_config_override`` qwen3_5 branch), and it already prefers the target's
``model_weights`` when building the draft model.  The override below is kept as
the superset the GGUF lane has always used (it also sets
``num_nextn_predict_layers``) and now accepts 0.33's second
``transformers_mtp_overrides`` argument, which it forwards to the original.

Second 0.33 gap: ``SpeculativeConfig.model`` for MTP arrives as the raw
``.gguf`` FILE path (the CLI's --speculative-model) while the draft needs the
target's hf-config DIRECTORY; ``__post_init__`` below rewrites it before
calling the original, so the draft ``ModelConfig`` is built from a valid
config source and gets the ``.gguf`` via ``model_weights``.
"""

import inspect


def install() -> None:
    from vllm.config import speculative as _spec

    if getattr(_spec.SpeculativeConfig, "_gguf_mtp_patched", False):
        return

    _orig = _spec.SpeculativeConfig.hf_config_override
    # vLLM >= 0.33 passes a second argument (transformers modeling backend
    # overrides); vLLM <= 0.25 passes only the config.  Detect once instead of
    # guessing per call.
    _orig_takes_transformers_overrides = (
        "transformers_mtp_overrides" in inspect.signature(_orig).parameters
    )

    def hf_config_override(hf_config, transformers_mtp_overrides=None):
        if transformers_mtp_overrides is not None:
            # vLLM 0.33: the Transformers modeling backend builds its own MTP
            # draft from the target config and wants only the plain overrides.
            if _orig_takes_transformers_overrides:
                return _orig(
                    hf_config,
                    transformers_mtp_overrides=transformers_mtp_overrides,
                )
            return _orig(hf_config)
        mt = getattr(hf_config, "model_type", None)
        if mt in ("qwen3_5_text", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_moe"):
            is_moe = "moe" in mt
            n_predict = (
                getattr(hf_config, "mtp_num_hidden_layers", None)
                or getattr(hf_config, "num_nextn_predict_layers", None)
                or 1
            )
            hf_config.model_type = "qwen3_5_mtp"
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "mtp_num_hidden_layers": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": [
                        "Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"
                    ],
                }
            )
            return hf_config
        return _orig(hf_config)

    _spec.SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)

    # The draft ModelConfig is built with model=<config-source-dir> (the plugin
    # rewrites the target's `model` to the hf-config dir) and an empty
    # model_weights, so the gguf loader can't find the .gguf. Propagate the
    # target's model_weights (the actual .gguf file) to the draft.
    #
    # vLLM >= 0.33 also needs the PRE-step below: `arg_utils.create_speculative_config`
    # passes --speculative-model straight through, so for the GGUF lane
    # `SpeculativeConfig.model` is the raw .gguf FILE, and upstream's own
    # `__post_init__` builds `ModelConfig(model=self.model, runner="draft", ...)`
    # from it -> HFValidationError (a .gguf path is not an HF repo id / config
    # dir). `model=None` (mtp) is just as broken: upstream falls back to
    # `target_model_config.model_weights` (also the .gguf). Fix: before the
    # original runs, point `self.model` at the target's hf-config DIRECTORY.
    _orig_post = _spec.SpeculativeConfig.__post_init__

    def _is_mtp_method(method) -> bool:
        # 0.33 normalises every MTP alias in MTPModelTypes to "mtp"
        # (`deepseek_mtp`, `qwen3_5_mtp`, ... -> "mtp"), but that happens inside
        # the original __post_init__, i.e. after this wrapper's pre-step.
        return isinstance(method, str) and (
            method == "mtp" or method.endswith("_mtp")
        )

    def _needs_config_dir(model) -> bool:
        # None -> upstream would substitute the .gguf; a .gguf path is never a
        # config source. Anything else is left exactly as configured.
        return model is None or str(model).lower().endswith(".gguf")

    def __post_init__(self):
        tmc = getattr(self, "target_model_config", None)
        if (
            tmc is not None
            and _is_mtp_method(getattr(self, "method", None))
            and _needs_config_dir(getattr(self, "model", None))
        ):
            config_dir = getattr(tmc, "model", None)
            if config_dir:
                print(
                    f"[mtp_enable] draft model source {self.model!r} is not an "
                    f"hf-config dir; using target config dir {config_dir!r} "
                    f"(weights come from model_weights).",
                    flush=True,
                )
                self.model = config_dir
        _orig_post(self)
        dmc = getattr(self, "draft_model_config", None)
        tmc = getattr(self, "target_model_config", None)
        if dmc is not None and tmc is not None:
            tw = getattr(tmc, "model_weights", None)
            if tw and not getattr(dmc, "model_weights", None):
                dmc.model_weights = tw

    _spec.SpeculativeConfig.__post_init__ = __post_init__
    _spec.SpeculativeConfig._gguf_mtp_patched = True

    # ---- Explicit gguf->HF weight map for the Qwen3_5MTP draft ----
    # The generic adapter builds its map from AutoModelForCausalLM.from_config(),
    # which cannot instantiate the vLLM-only Qwen3_5MTP. Provide a hardcoded map
    # for the baked blk.<L>.nextn.* head. arch_name is None for qwen3_5_mtp, so
    # prepare_loading skips the partition/GDN-fixup block and prepare_weights
    # yields only mapped tensors (the main blk.0..L-1 tensors are dropped).
    from vllm_gguf_plugin.weights_adapter import default as _wa

    A = _wa.GGUFWeightsAdapter
    if not getattr(A, "_gguf_mtp_namemap_patched", False):
        _orig_build = A.build_name_map

        def build_name_map(self, model_config):
            hf = model_config.hf_config
            if getattr(hf, "model_type", None) == "qwen3_5_mtp":
                text_cfg = hf.get_text_config()
                L = text_cfg.num_hidden_layers  # MTP block index
                b = f"blk.{L}"
                if getattr(text_cfg, "num_experts", 0):
                    # MoE MTP layer (e.g. Qwen3.6-35B-A3B): router + fused
                    # routed experts (3D ffn_*_exps, consumed whole by the
                    # GGUFMoEMethod weight_loader via the experts.0.*
                    # convention, same as the target model) + shared expert
                    # and its gate.
                    mlp = {
                        f"{b}.ffn_gate_inp.weight": "mtp.layers.0.mlp.gate.weight",
                        f"{b}.ffn_gate_inp_shexp.weight": "mtp.layers.0.mlp.shared_expert_gate.weight",
                        f"{b}.ffn_gate_shexp.weight": "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                        f"{b}.ffn_up_shexp.weight": "mtp.layers.0.mlp.shared_expert.up_proj.weight",
                        f"{b}.ffn_down_shexp.weight": "mtp.layers.0.mlp.shared_expert.down_proj.weight",
                        f"{b}.ffn_gate_exps.weight": "mtp.layers.0.mlp.experts.0.gate_proj.weight",
                        f"{b}.ffn_up_exps.weight": "mtp.layers.0.mlp.experts.0.up_proj.weight",
                        f"{b}.ffn_down_exps.weight": "mtp.layers.0.mlp.experts.0.down_proj.weight",
                    }
                else:
                    mlp = {
                        f"{b}.ffn_gate.weight": "mtp.layers.0.mlp.gate_proj.weight",
                        f"{b}.ffn_up.weight": "mtp.layers.0.mlp.up_proj.weight",
                        f"{b}.ffn_down.weight": "mtp.layers.0.mlp.down_proj.weight",
                    }
                return mlp | {
                    f"{b}.nextn.eh_proj.weight": "mtp.fc.weight",
                    f"{b}.nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
                    f"{b}.nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
                    f"{b}.nextn.shared_head_norm.weight": "mtp.norm.weight",
                    f"{b}.attn_norm.weight": "mtp.layers.0.input_layernorm.weight",
                    f"{b}.post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
                    f"{b}.attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
                    f"{b}.attn_k.weight": "mtp.layers.0.self_attn.k_proj.weight",
                    f"{b}.attn_v.weight": "mtp.layers.0.self_attn.v_proj.weight",
                    f"{b}.attn_output.weight": "mtp.layers.0.self_attn.o_proj.weight",
                    f"{b}.attn_q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
                    f"{b}.attn_k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
                    # NOTE: token_embd/output are deliberately NOT mapped: the
                    # draft shares the target's embed_tokens/lm_head modules
                    # (see the placeholder patch below), so loading its own
                    # copies would only waste ~1.3 GiB.
                }
            return _orig_build(self, model_config)

        A.build_name_map = build_name_map
        A._gguf_mtp_namemap_patched = True

    # ---- Draft shares the target's (GGUF-quantized) embed_tokens / lm_head ----
    # Qwen3_5MTP allocates its own VocabParallelEmbedding + ParallelLMHead
    # (~1.3 GiB for the 248k vocab) which both runners then replace with the
    # target's modules -- but the memory is already allocated (and counted as
    # weights) by then. So under the GGUF lane we construct the draft with
    # parameter-less placeholders; the share step installs the real modules:
    #   MRV2: eagle/utils.load_eagle_model (assigns target embed_tokens/lm_head;
    #         Qwen3_5MTP declares no has_own_* so _should_share -> True).
    #   MRV1: SpecDecodeBaseProposer._maybe_share_* (wrapped below so module-
    #         level sharing works even when the target params are quantized).
    import torch.nn as _nn

    class _SharedPlaceholder(_nn.Module):
        """Parameter-less stand-in, replaced by the target's module at share time."""

        def forward(self, *a, **k):
            raise RuntimeError(
                "[mtp_enable] draft embed_tokens/lm_head placeholder was never "
                "replaced by the target's module (share step did not run)"
            )

        def tie_weights(self, *a, **k):
            return self

    def _is_gguf_cfg(vllm_config) -> bool:
        qc = getattr(vllm_config, "quant_config", None)
        try:
            if qc is not None and qc.get_name() == "gguf":
                return True
        except Exception:
            pass
        lc = getattr(vllm_config, "load_config", None)
        return str(getattr(lc, "load_format", "")).lower() == "gguf"

    try:
        from vllm.model_executor.models import qwen3_5_mtp as _q
    except Exception:  # model file absent on some vLLM versions
        _q = None
    if _q is not None and not getattr(_q, "_gguf_mtp_placeholder_patched", False):

        def _wrap_init(cls):
            _orig_init = cls.__init__

            def __init__(self, *args, **kwargs):
                vc = kwargs.get("vllm_config")
                if vc is None and args:
                    vc = args[0]
                if not _is_gguf_cfg(vc):
                    return _orig_init(self, *args, **kwargs)
                saved = (_q.VocabParallelEmbedding, _q.ParallelLMHead)
                _q.VocabParallelEmbedding = lambda *a, **k: _SharedPlaceholder()
                _q.ParallelLMHead = lambda *a, **k: _SharedPlaceholder()
                try:
                    _orig_init(self, *args, **kwargs)
                finally:
                    _q.VocabParallelEmbedding, _q.ParallelLMHead = saved

            cls.__init__ = __init__

        # Only the two classes that allocate embed/lm_head (MoE MTP inherits
        # Qwen3_5MTP.__init__ via super()).
        _wrap_init(_q.Qwen3_5MultiTokenPredictor)
        _wrap_init(_q.Qwen3_5MTP)
        _q._gguf_mtp_placeholder_patched = True

    def _is_placeholder_draft(draft) -> bool:
        return isinstance(
            getattr(getattr(draft, "model", None), "embed_tokens", None),
            _SharedPlaceholder,
        ) or isinstance(getattr(draft, "lm_head", None), _SharedPlaceholder)

    # MRV1 (vLLM 0.25 and 0.33's SpecDecodeBaseProposer): share at MODULE
    # level -- no `.weight` access, so quantized target params are fine.
    from vllm.v1.spec_decode import llm_base_proposer as _prop

    P = _prop.SpecDecodeBaseProposer
    if not getattr(P, "_gguf_mtp_share_patched", False):

        def _share_embeddings(self, target_language_model):
            inner = getattr(target_language_model, "model", None)
            target_embed = getattr(inner, "embed_tokens", None) or getattr(
                inner, "embedding", None
            )
            if target_embed is None:
                raise AttributeError("Target model has no embed_tokens/embedding")
            draft_inner = self.model.model
            if hasattr(draft_inner, "embed_tokens"):
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed
            print(
                "[mtp_enable] draft shares target embed_tokens/lm_head "
                "(no draft copy allocated) [MRV1 embed_tokens]",
                flush=True,
            )

        def _share_lm_head(self, target_language_model):
            target_head = getattr(target_language_model, "lm_head", None)
            if target_head is None:
                raise AttributeError("Target model has no lm_head")
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_head
            # Per-layer shared_head.head references (DeepSeek-style MTP) too.
            layers = getattr(getattr(self.model, "model", None), "layers", None)
            if layers is not None:
                items = (
                    layers.values() if isinstance(layers, _nn.ModuleDict) else layers
                )
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_head
            print(
                "[mtp_enable] draft shares target embed_tokens/lm_head "
                "(no draft copy allocated) [MRV1 lm_head]",
                flush=True,
            )

        def _wrap(orig, mine):
            def f(self, target_language_model):
                # Only the GGUF MTP draft (placeholders) needs this; every
                # other drafter keeps upstream logic.
                if _is_placeholder_draft(getattr(self, "model", None)):
                    return mine(self, target_language_model)
                return orig(self, target_language_model)

            return f

        P._maybe_share_embeddings = _wrap(P._maybe_share_embeddings, _share_embeddings)
        P._maybe_share_lm_head = _wrap(P._maybe_share_lm_head, _share_lm_head)
        P._gguf_mtp_share_patched = True

    # MRV2 (0.33): eagle/utils.load_eagle_model calls maybe_share_target_embed
    # via a module-global lookup (patchable; load_eagle_model itself is bound by
    # name into the speculators, so it is not). Wrap it to log + verify.
    try:
        from vllm.v1.worker.gpu.spec_decode.eagle import utils as _eu
    except Exception:
        _eu = None
    if _eu is not None and not getattr(_eu, "_gguf_mtp_log_patched", False):
        _orig_share = _eu.maybe_share_target_embed

        def maybe_share_target_embed(draft_model, draft_inner, target_inner):
            was_placeholder = isinstance(
                getattr(draft_inner, "embed_tokens", None), _SharedPlaceholder
            )
            _orig_share(draft_model, draft_inner, target_inner)
            if was_placeholder:
                if isinstance(
                    getattr(draft_inner, "embed_tokens", None), _SharedPlaceholder
                ):
                    raise RuntimeError(
                        "[mtp_enable] MRV2 share left the draft embed_tokens "
                        "placeholder in place"
                    )
                print(
                    "[mtp_enable] draft shares target embed_tokens/lm_head "
                    "(no draft copy allocated) [MRV2]",
                    flush=True,
                )

        _eu.maybe_share_target_embed = maybe_share_target_embed
        _eu._gguf_mtp_log_patched = True

    print("[mtp_enable] qwen3_5 MTP speculative override INSTALLED", flush=True)
