# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 (qwen35 / qwen35moe) GGUF → HF config mapping + registration.

The transformers GGUF loader refuses arch `qwen35` / `qwen35moe` because they
are not in `GGUF_SUPPORTED_ARCHITECTURES` and have no `GGUF_CONFIG_MAPPING`
entry.  This module:

1. Provides ``map_qwen35_config()`` — a pure function that takes the raw GGUF
   metadata dict and returns the correct ``PreTrainedConfig`` with all fields
   populated per ``docs/qwen35-config-mapping.md``.
2. Provides ``register()`` — monkeypatches the transformers GGUF loader so that
   real GGUF files with these architectures produce the right config.
"""

from __future__ import annotations

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import PretrainedConfig
from transformers.models.qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig

# ── arch → target text config ────────────────────────────────────────

_CONFIG_CLASS: dict[str, type[PretrainedConfig]] = {
    "qwen35": Qwen3_5TextConfig,
    "qwen35moe": Qwen3_5MoeTextConfig,
}


# ── pure mapping function ────────────────────────────────────────────

def _safe_int(meta: dict, key: str) -> int:
    """Pull an integer value, tolerating ``__omitted_array__`` sentinel dicts."""
    v = meta.get(key)
    if v is None:
        return 0
    if isinstance(v, dict):
        return v.get("__omitted_array__", 0)
    return int(v)


def _safe_float(meta: dict, key: str, default: float = 1e-6) -> float:
    v = meta.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def map_qwen35_config(arch: str, meta: dict) -> PretrainedConfig:
    """Build a ``Qwen3_5TextConfig`` or ``Qwen3_5MoeTextConfig`` from GGUF KV.

    *arch* is the raw ``general.architecture`` value (``"qwen35"`` or
    ``"qwen35moe"``).  *meta* is the full GGUF metadata dict (keys like
    ``"qwen35.block_count"`` or ``"qwen35moe.expert_count"``).

    The raw ``block_count`` includes the trailing MTP layer, so we subtract
    ``nextn_predict_layers`` to get the true number of transformer layers.
    """

    prefix = arch + "."  # "qwen35." or "qwen35moe."

    # ── core dimensions ────────────────────────────────────────────────
    block_count = _safe_int(meta, f"{prefix}block_count")
    nextn = _safe_int(meta, f"{prefix}nextn_predict_layers")
    num_hidden_layers = block_count - nextn  # critical: MTP layer is not a model layer

    hidden_size = _safe_int(meta, f"{prefix}embedding_length")
    intermediate_size = _safe_int(meta, f"{prefix}feed_forward_length")
    max_position_embeddings = _safe_int(meta, f"{prefix}context_length")

    num_attention_heads = _safe_int(meta, f"{prefix}attention.head_count")
    num_key_value_heads = _safe_int(meta, f"{prefix}attention.head_count_kv")
    head_dim = _safe_int(meta, f"{prefix}attention.key_length")  # full-attn dim
    rms_norm_eps = _safe_float(meta, f"{prefix}attention.layer_norm_rms_epsilon")
    rope_theta = _safe_float(meta, f"{prefix}rope.freq_base", default=10000000.0)

    # ── GDN / linear attention dims ────────────────────────────────────
    linear_conv_kernel_dim = _safe_int(meta, f"{prefix}ssm.conv_kernel")
    state_size = _safe_int(meta, f"{prefix}ssm.state_size")
    inner_size = _safe_int(meta, f"{prefix}ssm.inner_size")
    linear_num_key_heads = _safe_int(meta, f"{prefix}ssm.group_count")

    # linear_num_value_heads = inner_size // state_size (cross-checked with ssm.time_step_rank)
    linear_num_value_heads = inner_size // state_size if state_size else 0

    # ── full attention interval → config derives layer_types itself ────
    full_attention_interval = _safe_int(meta, f"{prefix}full_attention_interval") or 4

    # ── vocab_size (from tokenizer token list length, or default) ──────
    tok_tokens = meta.get("tokenizer.ggml.tokens")
    if isinstance(tok_tokens, list):
        vocab_size = len(tok_tokens)
    elif isinstance(tok_tokens, dict):
        vocab_size = tok_tokens.get("__omitted_array__", 248320)
    else:
        vocab_size = 248320  # Qwen3.5 default

    # ── construct the base config dict ─────────────────────────────────
    cfg_kwargs: dict = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": rms_norm_eps,
        "linear_conv_kernel_dim": linear_conv_kernel_dim,
        "linear_key_head_dim": state_size,
        "linear_value_head_dim": state_size,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "full_attention_interval": full_attention_interval,
        # rope_theta is passed via rope_parameters; the config class wraps it
        "rope_parameters": {"rope_theta": rope_theta, "rope_type": "default"},
    }

    # ── MoE-specific fields ────────────────────────────────────────────
    if arch == "qwen35moe":
        cfg_kwargs.update({
            "num_experts": _safe_int(meta, f"{prefix}expert_count"),
            "num_experts_per_tok": _safe_int(meta, f"{prefix}expert_used_count"),
            "moe_intermediate_size": _safe_int(meta, f"{prefix}expert_feed_forward_length"),
            "shared_expert_intermediate_size": _safe_int(
                meta, f"{prefix}expert_shared_feed_forward_length"
            ),
        })

    config_cls = _CONFIG_CLASS[arch]
    return config_cls(**cfg_kwargs)


# ── monkey-patch registration ────────────────────────────────────────

def _read_gguf_metadata(path) -> dict:
    """Read the full GGUF metadata KV dict with native-python values.

    Uses ``GGUFReader.get_field(...).contents()`` — the correct value API (the
    same one T-1's dump script uses).  ``field.parts[-1]`` is an index array,
    NOT the value, and must never be used here.  Large arrays (tokenizer
    vocab/merges) are returned as-is; callers only read scalar config keys.
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)
    meta: dict = {}
    for key in reader.fields.keys():
        field = reader.get_field(key)
        try:
            meta[key] = field.contents() if field is not None else None
        except Exception:
            meta[key] = None
    return meta



def _register_qwen35_causallm() -> None:
    """Register the text-only Qwen3_5ForCausalLM / Qwen3_5MoeForCausalLM arches.

    vLLM's registry only exposes the multimodal *ForConditionalGeneration variants,
    which unconditionally build a vision tower (config.vision_config). A text-only
    GGUF serve of this VL model must use the CausalLM class instead.
    """
    from vllm import ModelRegistry

    # Register the plugin's hybrid-flagged subclasses (see qwen35_models.py —
    # the flag must live on the class in its home module because vLLM
    # inspects model classes in a fresh subprocess).
    for arch, path in (
        ("Qwen3_5ForCausalLM", "vllm_gguf_plugin.qwen35_models:Qwen3_5ForCausalLM"),
        ("Qwen3_5MoeForCausalLM", "vllm_gguf_plugin.qwen35_models:Qwen3_5MoeForCausalLM"),
    ):
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, path)


def register() -> None:
    """Monkey-patch the transformers GGUF loader for qwen35 / qwen35moe.

    Safe to call multiple times (idempotent).  Called from the plugin's main
    registration entrypoint so the support is active on any ``import`` of the
    plugin package.

    The stock loader cannot map these arches: its field table is flat (cannot
    express ``num_hidden_layers = block_count - nextn`` or the dual
    ``state_size → {key,value}_head_dim`` map) and it has no arch-normalisation
    branch (``qwen35 → qwen3_5``).  So rather than seed its tables, we wrap
    ``load_gguf_checkpoint`` and, for OUR arches, build the config directly with
    the (unit-tested) ``map_qwen35_config``.  ``map_qwen35_config`` is the single
    source of truth for the field mapping.
    """
    # Add arches to the supported set so the stock loader does not raise on the
    # tensor path (return_tensors=True); tensor-name correctness is T-6.
    for arch in ("qwen35", "qwen35moe"):
        if arch not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
            _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append(arch)

    # Idempotency: never double-wrap.
    if getattr(_gguf_utils.load_gguf_checkpoint, "_qwen35_wrapped", False):
        return

    original_load = _gguf_utils.load_gguf_checkpoint

    def _wrapped_load_gguf_checkpoint(
        gguf_checkpoint_path,
        return_tensors=False,
        model_to_load=None,
        torch_dtype=None,
    ):
        # Peek the RAW architecture first (keyed on general.architecture, not on
        # any post-parse model_type the stock loader would mis-derive).
        try:
            arch = _read_gguf_metadata(gguf_checkpoint_path).get("general.architecture")
        except Exception:
            arch = None

        if arch not in _CONFIG_CLASS:
            return original_load(
                gguf_checkpoint_path, return_tensors, model_to_load, torch_dtype
            )

        # Our arch: build the correct config ourselves.
        meta = _read_gguf_metadata(gguf_checkpoint_path)
        config = map_qwen35_config(arch, meta)

        if not return_tensors:
            # Config-only load (AutoConfig path) — the stock loader cannot map
            # our arch, so do not call it; return just the config.
            return {"config": config, "tensors": {}}

        # Tensor path: let the stock loader extract tensors/tokenizer, then
        # override its (incorrect) config with ours.  Tensor-name mapping
        # correctness is T-6's concern.
        result = original_load(
            gguf_checkpoint_path, return_tensors, model_to_load, torch_dtype
        )
        result["config"] = config
        return result

    _wrapped_load_gguf_checkpoint._qwen35_wrapped = True  # type: ignore[attr-defined]
    _gguf_utils.load_gguf_checkpoint = _wrapped_load_gguf_checkpoint

    _gguf_utils.load_gguf_checkpoint = _wrapped_load_gguf_checkpoint
    _register_qwen35_causallm()

    # Wire GGUF quant into embeddings/lm_head for models that omit quant_config
    # on VocabParallelEmbedding (Qwen3.5/Qwen3-Next) — see embed_quant_patch.
    from .embed_quant_patch import patch_vocab_embedding_gguf

    patch_vocab_embedding_gguf()
