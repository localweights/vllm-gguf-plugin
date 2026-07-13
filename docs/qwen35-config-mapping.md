# Qwen3.6 (arch `qwen35` / `qwen35moe`) GGUF → HF config mapping

Chair-derived ground truth for WS1 D2 (2026-07-13), introspected against
transformers in the dev venv (`Qwen3_5TextConfig` / `Qwen3_5MoeConfig`) and the
committed metadata fixtures. This is the authoritative field map the loader patch
must reproduce; the T-4 test asserts every value below.

## Where the mapping actually lives (spec D2 correction)

The GGUF→HF config translation is NOT done in the plugin's `config_parser.py`. It
happens inside **`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`**:

1. `architecture = read_field(reader, "general.architecture")` → `"qwen35"` / `"qwen35moe"`.
2. A hardcoded if-chain normalizes some arches (`qwen3moe → qwen3_moe`, …). **There is
   no `qwen35` branch** → `updated_architecture` stays `"qwen35"`.
3. Guard: `if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture
   not in GGUF_SUPPORTED_ARCHITECTURES: raise ValueError(...)` → **raises for qwen35**.
4. Config fields are pulled via `GGUF_TO_TRANSFORMERS_MAPPING["config"][updated_architecture]`
   (a.k.a. `GGUF_CONFIG_MAPPING`) — **no `qwen35` entry**.
5. Tensor post-processing: `TENSOR_PROCESSORS.get(architecture, TensorProcessor)`.

So the fork must, at import time (monkeypatch, in-style with the plugin's existing
vLLM patching):
- add `qwen35 → qwen3_5_text`, `qwen35moe → qwen3_5_moe_text` arch normalization,
- add both to `GGUF_SUPPORTED_ARCHITECTURES`,
- add the `GGUF_CONFIG_MAPPING` entries below,
- apply the **num_hidden_layers = block_count − nextn_predict_layers** correction
  (the raw `block_count` includes the trailing MTP layer — see T-2 skip-list).

Target config: the TEXT config (`model_type = "qwen3_5_text"` / `"qwen3_5_moe_text"`).
`qwen3_5` / `qwen3_5_moe` are multimodal wrapper configs with a nested `text_config`;
the GGUF is language-model-only, so map straight to the text config.

## Field map (qwen35 / dense, 27B)

| GGUF metadata key | HF `Qwen3_5TextConfig` attr | 27B value |
|---|---|---|
| `general.architecture` | `model_type` (→ `qwen3_5_text`) | qwen35 |
| `qwen35.context_length` | `max_position_embeddings` | 262144 |
| `qwen35.block_count` − `qwen35.nextn_predict_layers` | `num_hidden_layers` | 65 − 1 = **64** |
| `qwen35.feed_forward_length` | `intermediate_size` | 17408 |
| `qwen35.embedding_length` | `hidden_size` | 5120 |
| `qwen35.rope.freq_base` | `rope_theta` | 10000000.0 |
| `qwen35.attention.head_count` | `num_attention_heads` | 24 |
| `qwen35.attention.head_count_kv` | `num_key_value_heads` | 4 |
| `qwen35.attention.key_length` | `head_dim` (full-attn) | 256 |
| `qwen35.attention.layer_norm_rms_epsilon` | `rms_norm_eps` | 1e-6 |
| `qwen35.ssm.conv_kernel` | `linear_conv_kernel_dim` | 4 |
| `qwen35.ssm.state_size` | `linear_key_head_dim` & `linear_value_head_dim` | 128 |
| `qwen35.ssm.group_count` | `linear_num_key_heads` | 16 |
| `qwen35.ssm.inner_size` ÷ `qwen35.ssm.state_size` | `linear_num_value_heads` | 6144/128 = **48** |
| `qwen35.full_attention_interval` | drives `layer_types` | 4 |
| `vocab_size` (from tokens / tensor) | `vocab_size` | — |

`layer_types` is built by the config itself from `full_attention_interval=4`:
`"full_attention" if (i+1) % 4 == 0 else "linear_attention"` over `num_hidden_layers`
→ **16 full-attn + 48 GDN(linear)** for 64 layers. Do NOT hand-populate it; pass
`full_attention_interval` through (or set `layer_types` identically) and let the
config derive it. `linear_num_value_heads = inner_size/state_size = 48` equals
`qwen35.ssm.time_step_rank` (48) — cross-check, they must agree.

Distinct head dims (do not conflate): `head_dim=256` is the FULL-ATTENTION head dim
(`attention.key_length`); `linear_{key,value}_head_dim=128` are the GDN dims
(`ssm.state_size`).

## MoE additions (qwen35moe, 35B — num_hidden_layers 41 − 1 = 40)

Same base map plus (from `qwen3_moe` template + `Qwen3_5MoeConfig`):
| GGUF key | HF attr |
|---|---|
| `qwen35moe.expert_count` | `num_experts` |
| `qwen35moe.expert_used_count` | `num_experts_per_tok` |
| `qwen35moe.expert_feed_forward_length` | `moe_intermediate_size` |
| `qwen35moe.expert_shared_feed_forward_length` | `shared_expert_intermediate_size` (if present) |

Builder MUST introspect `Qwen3_5MoeConfig` for the exact MoE attr names + any
shared-expert / norm_topk_prob fields and dump the actual `qwen35moe.*` keys from
`tests/fixtures/qwen36_35b_metadata.json` — the table above lists the expected pairs;
confirm names against the real config before wiring.
