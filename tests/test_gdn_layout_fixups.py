# SPDX-License-Identifier: Apache-2.0
# GDN layout fixup tests — undo llama.cpp transforms at GGUF load time.
# CPU-only.

import math

import torch

# ── Helpers: forward (llama.cpp) and inverse (our adapter) reorder ─────

def _reorder_v_heads_fwd(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Forward: grouped → tiled (what llama.cpp does)."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)

def _reorder_v_heads_inverse(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Inverse: tiled → grouped (what we do at load time).

    Swap K and v_per_k roles: reshape into (v_per_k, K, hd), swap → (K, v_per_k, hd).
    """
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)

# Model dims from the 27B: K=16, V=48, v_per_k=3, head_dim=128
K = 16
V = 48
V_PER_K = 3
HD = 128

# ── Test 1: roundtrip ──────────────────────────────────────────────────

def test_reorder_inverse_roundtrip():
    """Forward then inverse returns original for 1D, 2D, and conv-shaped tensors."""
    # 1D: shape (48,) — like dt_bias / A_log (expanded to 2D for reshape)
    t1d = torch.arange(V, dtype=torch.float32)
    fwd1d = _reorder_v_heads_fwd(t1d.unsqueeze(-1), 0, K, V_PER_K, 1).squeeze(-1)
    inv1d = _reorder_v_heads_inverse(fwd1d.unsqueeze(-1), 0, K, V_PER_K, 1).squeeze(-1)
    assert torch.equal(t1d, inv1d), "1D roundtrip failed"

    # 2D: shape (V*128, 7) — like in_proj_z.weight (rows)
    t2d = torch.randn(V * HD, 7)
    fwd2d = _reorder_v_heads_fwd(t2d, 0, K, V_PER_K, HD)
    inv2d = _reorder_v_heads_inverse(fwd2d, 0, K, V_PER_K, HD)
    assert torch.allclose(t2d, inv2d, atol=1e-6), "2D row roundtrip failed"

    # Conv layout: rows = [q(K*128), k(K*128), v(V*128)] = (2048+2048+6144, 4)
    qk_size = K * HD * 2  # 4096
    v_size = V * HD  # 6144
    total_rows = qk_size + v_size  # 10240
    cols = 4
    t_conv = torch.arange(total_rows * cols, dtype=torch.float32).reshape(total_rows, cols)
    # Only v block is reordered in forward
    qk_fwd = t_conv[:qk_size]
    v_fwd = _reorder_v_heads_fwd(t_conv[qk_size:], 0, K, V_PER_K, HD)
    fwd_conv = torch.cat([qk_fwd, v_fwd], dim=0)
    # Inverse: only v block
    qk_inv = fwd_conv[:qk_size]
    v_inv = _reorder_v_heads_inverse(fwd_conv[qk_size:], 0, K, V_PER_K, HD)
    inv_conv = torch.cat([qk_inv, v_inv], dim=0)
    assert torch.equal(t_conv, inv_conv), "Conv layout roundtrip failed"

    # Column permutation roundtrip for out_proj
    t_col = torch.randn(5120, V * HD)  # weight (hidden, V*128)
    fwd_col = _reorder_v_heads_fwd(t_col, 1, K, V_PER_K, HD)
    inv_col = _reorder_v_heads_inverse(fwd_col, 1, K, V_PER_K, HD)
    assert torch.allclose(t_col, inv_col, atol=1e-6), "Column roundtrip failed"

# ── Test 2: A_log conversion ───────────────────────────────────────────

def test_a_log_conversion():
    """a = -exp(A_log) roundtrips through inverse fixup to A_log within 1e-6."""
    A_log = torch.tensor([-0.5, -1.2, 0.0, 2.3, -3.7], dtype=torch.float32)
    # llama.cpp stores: a = -exp(A_log)
    stored = -torch.exp(A_log)
    # Inverse: A_log = log(-a)
    recovered = torch.log(-stored)
    assert torch.allclose(A_log, recovered, atol=1e-6), \
        f"A_log roundtrip failed: {A_log} vs {recovered}"

# ── Test 3: norm -1 ────────────────────────────────────────────────────

def test_norm_minus_one():
    """Non-linear_attn norms get -1; ssm_norm (linear_attn.norm) is unchanged."""
    from vllm_gguf_plugin.weights_adapter.default import (
        _needs_norm_minus_one,
        _apply_norm_minus_one,
    )

    # Should subtract 1
    assert _needs_norm_minus_one("model.layers.0.input_layernorm.weight")
    assert _needs_norm_minus_one("model.layers.5.post_attention_layernorm.weight")
    assert _needs_norm_minus_one("model.layers.3.attention.q_norm.weight")
    assert _needs_norm_minus_one("model.layers.3.attention.k_norm.weight")
    assert _needs_norm_minus_one("model.norm.weight")
    assert _needs_norm_minus_one("mtp.layers.0.input_layernorm.weight")
    assert _needs_norm_minus_one("mtp.norm.weight")

    # Should NOT subtract 1
    assert not _needs_norm_minus_one("model.layers.0.linear_attn.norm.weight")
    assert not _needs_norm_minus_one("model.layers.10.linear_attn.norm.weight")
    assert not _needs_norm_minus_one("lm_head.weight")

    # Value test
    w = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float32)
    result = _apply_norm_minus_one(w, "model.layers.0.input_layernorm.weight")
    assert torch.allclose(result, torch.tensor([0.5, 1.0, -0.2], dtype=torch.float32))

    w2 = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float32)
    result2 = _apply_norm_minus_one(w2, "model.layers.0.linear_attn.norm.weight")
    assert torch.allclose(result2, w2)  # unchanged

# ── Test 4: QKV V-rows only (quant-safe row perm) ──────────────────────

def test_qkv_v_rows_only():
    """Build a fake quantized qkv tensor with distinct per-row markers;
    assert q/k rows untouched and v rows land in grouped order."""
    from vllm_gguf_plugin.weights_adapter.default import (
        _apply_gdn_row_fixup,
    )

    q_dim = K * HD  # 2048
    k_dim = K * HD  # 2048
    v_dim = V * HD  # 6144

    # Each row gets a unique marker value = row index
    fake_qkv = torch.arange(q_dim + k_dim + v_dim, dtype=torch.float32).unsqueeze(-1)

    # Apply fixup: only v block rows should be inverse-reordered
    result = _apply_gdn_row_fixup(fake_qkv, "in_proj_qkv", K, V, V_PER_K, HD)

    # q rows (0..2047) and k rows (2048..4095) should be untouched
    assert torch.equal(result[:q_dim], fake_qkv[:q_dim]), "Q rows were modified"
    assert torch.equal(result[q_dim:q_dim + k_dim], fake_qkv[q_dim:q_dim + k_dim]), \
        "K rows were modified"

    # v rows: the inverse-reordered values
    original_v = fake_qkv[q_dim + k_dim:]
    expected_v = _reorder_v_heads_inverse(original_v, 0, K, V_PER_K, HD)
    assert torch.equal(result[q_dim + k_dim:], expected_v), "V rows not inverse-reordered"

    # Verify that forward reorder of expected_v gives back the original_v
    fwd_check = _reorder_v_heads_fwd(expected_v, 0, K, V_PER_K, HD)
    assert torch.equal(fwd_check, original_v), "V rows don't roundtrip"

# ── Test 5: out_proj runtime column perm ───────────────────────────────

def test_out_proj_runtime_perm():
    """A layer with gguf_input_col_perm set: apply() output equals matmul
    with manually permuted x (CPU-only, F32 UNQUANTIZED)."""
    import unittest.mock as mock
    import vllm_gguf_plugin.quantization as quant_pkg
    import vllm_gguf_plugin.quantization.linear as linear_mod
    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
    import gguf
    from gguf import GGMLQuantizationType as WeightType

    col_perm = _reorder_v_heads_fwd(
        torch.arange(V * HD, dtype=torch.long).unsqueeze(0),
        1, K, V_PER_K, HD
    ).squeeze(0)

    input_dim = V * HD
    output_dim = 5120
    batch = 2
    seq = 3

    class _OutProjLayer:
        def __init__(self):
            self.gguf_input_col_perm = col_perm

    layer = _OutProjLayer()

    # Use float32 with bounded values to avoid overflow
    weight = (torch.randn(output_dim, input_dim, dtype=torch.float32) * 0.01)
    qweight_type_val = int(WeightType.F32)

    layer.qweight = weight
    layer.qweight_type = type("FakeType", (), {"weight_type": qweight_type_val})()
    layer.qweight.shard_id = []

    method = GGUFLinearMethod(quant_config=None)

    x = (torch.randn(batch * seq, input_dim, dtype=torch.float32) * 0.01)

    # Patch at package level (where `from . import` in apply() resolves to)
    # Use the raw _fused_mul_mat_gguf which has a CPU path for UNQUANTIZED.
    with mock.patch.object(
        quant_pkg, "fused_mul_mat_gguf", linear_mod._fused_mul_mat_gguf
    ):
        out = method.apply(layer, x)

    x_permuted = x.index_select(-1, col_perm)
    expected = x_permuted @ weight.T

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1.0, rtol=1e-3), \
        f"out_proj perm mismatch: max diff = {(out - expected).abs().max()}"