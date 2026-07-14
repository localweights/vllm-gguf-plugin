# SPDX-License-Identifier: Apache-2.0
"""Text-only Qwen3.5 model classes with the hybrid flag.

vLLM ships ``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM`` but neither
registers them nor marks them ``is_hybrid = True`` (only the multimodal
*ForConditionalGeneration variants carry the flag). Without the flag,
``HybridAttentionMambaModelConfig`` never runs: no ``mamba_block_size`` is
derived (AssertionError in mamba ``get_kv_cache_spec``) and the KV cache is
sized as if all 64 layers were full attention (~50x over-estimate).

Registering these subclasses BY PLUGIN MODULE PATH matters: vLLM inspects
model classes in a fresh subprocess where runtime monkeypatches don't exist,
so the flag must live on the class in its home module.
"""

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration as _Qwen3_5VL,
    Qwen3_5MoeForCausalLM as _Qwen3_5MoeForCausalLM,
)


class _HybridStateMixin:
    """Hybrid flag + GDN state classmethods (borrowed from the VL class).

    The mamba state shape/dtype/copy hooks only read hf_text_config, so the
    VL implementations are correct for the text-only model too.
    """

    is_hybrid = True
    get_mamba_state_dtype_from_config = _Qwen3_5VL.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = _Qwen3_5VL.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = _Qwen3_5VL.get_mamba_state_copy_func


class Qwen3_5ForCausalLM(_HybridStateMixin, _Qwen3_5ForCausalLM):
    pass


class Qwen3_5MoeForCausalLM(_HybridStateMixin, _Qwen3_5MoeForCausalLM):
    pass
