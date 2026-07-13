# gguf-py Coverage Verdict for Qwen3.6 GDN-hybrid (qwen35/qwen35moe)

## Inspection

Programmatic inspection of the installed `gguf` package (via `gguf.constants` and
`gguf.tensor_mapping`) was performed against the Qwen3.6 artifacts:

### MODEL_ARCH enum (`gguf.constants.MODEL_ARCH`)

The following Qwen-family architecture identifiers exist:

| Member     | Value | Notes                          |
|------------|-------|--------------------------------|
| `QWEN`     | 25    | Original Qwen                  |
| `QWEN2`    | 26    | Qwen2                          |
| `QWEN2MOE` | 27    | Qwen2 MoE                      |
| `QWEN2VL`  | 28    | Qwen2-VL                       |
| `QWEN3`    | 29    | Qwen3                          |
| `QWEN3MOE` | 30    | Qwen3 MoE                      |
| `QWEN3NEXT`| 31    | Qwen3-Next                     |
| `QWEN3VL`  | 32    | Qwen3-VL                       |
| `QWEN3VLMOE`| 33   | Qwen3-VL MoE                   |
| **`QWEN35`**   | **34** | **Qwen3.5 — matches 27B GGUF** |
| **`QWEN35MOE`** | **35** | **Qwen3.5 MoE — matches 35B GGUF** |

### Tensor Name Maps (`gguf.tensor_mapping.get_tensor_name_map`)

Both `MODEL_ARCH.QWEN35` and `MODEL_ARCH.QWEN35MOE` are resolvable via
`get_tensor_name_map(arch, n_blocks)` and return populated `TensorNameMap` objects:

- **QWEN35** (65 blocks): 18,514 mapping entries. Includes SSM templates:
  `blk.{N}.ssm_conv1d`, `blk.{N}.ssm_dt`, `blk.{N}.ssm_a`, `blk.{N}.ssm_norm`,
  `blk.{N}.ssm_out`, `blk.{N}.ssm_alpha`, `blk.{N}.ssm_beta`, plus attention,
  feed-forward, and gating tensors.

- **QWEN35MOE** (65 blocks): 18,189 mapping entries. Same SSM template family
  plus MoE expert mappings.

Both maps include GDN-specific tensors (`ssm_*`) and the hybrid MTP/nextn
structure expected by Qwen3.6 GGUFs.

## Conclusion

gguf-py already includes the `QWEN35` and `QWEN35MOE` architecture entries with
complete tensor name mappings covering all GDN (SSM) hybrid block tensors. No
additional mappings need to be added to the fork for basic tensor-name resolution.

VERDICT: reuse