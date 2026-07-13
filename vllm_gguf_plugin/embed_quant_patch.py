# SPDX-License-Identifier: Apache-2.0
"""Wire GGUF quantization into VocabParallelEmbedding for models that omit it.

Some vLLM models (Qwen3-Next / Qwen3.5 hybrids) construct
``VocabParallelEmbedding(vocab_size, hidden_size)`` WITHOUT passing
``quant_config``, so a GGUF-quantized token embedding / lm_head (e.g. Q4_K
``token_embd``, Q6_K ``output``) never binds ``GGUFEmbeddingMethod`` and loading
fails with ``no module or parameter named 'embed_tokens.qweight_type'``.

This patch makes ``VocabParallelEmbedding.__init__`` fall back to the ACTIVE
vLLM config's quant_config when the caller passed none — but ONLY when that
active quant_config is the GGUF one, so non-GGUF serving is completely unaffected.
"""

from __future__ import annotations


def patch_vocab_embedding_gguf() -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    import inspect

    orig_init = VocabParallelEmbedding.__init__
    if getattr(orig_init, "_gguf_embed_patched", False):
        return
    sig = inspect.signature(orig_init)

    def _active_gguf_quant_config():
        try:
            from vllm.config import get_current_vllm_config

            active = getattr(get_current_vllm_config(), "quant_config", None)
            if active is not None and type(active).__name__ == "GGUFConfig":
                return active
        except Exception:
            pass
        return None

    def patched_init(self, *args, **kwargs):
        # Bind to the real signature so we read/set quant_config regardless of
        # whether the caller passed it positionally (ParallelLMHead) or by
        # keyword — or omitted it (Qwen3.5 embed_tokens).
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("quant_config") is None:
            gguf = _active_gguf_quant_config()
            if gguf is not None:
                bound.arguments["quant_config"] = gguf
        return orig_init(*bound.args, **bound.kwargs)

    patched_init._gguf_embed_patched = True  # type: ignore[attr-defined]
    VocabParallelEmbedding.__init__ = patched_init
