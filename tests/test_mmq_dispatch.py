"""Prefill dispatch for K-/standard-quant weights (2026-10-07).

On Ampere the legacy MMQ kernel (mul_mat_q5_K) was 46% of all prefill GPU time
for Qwen3.6/3.8-27B (attn qkv/v are Q5_K) at a 1024-token chunk. Above
_MMQ_MAX_M tokens those weights must take dequantize + cuBLAS like IQ types;
small batches keep MMQ.
"""
import torch
from gguf import GGMLQuantizationType as WT

import vllm_gguf_plugin.quantization.linear as lin


def _record(monkeypatch):
    calls = []

    def mmq(W, X, t, row):
        calls.append("mmq")
        return torch.zeros(X.shape[0], row, dtype=X.dtype)

    def mmvq(W, X, t, row):
        calls.append("mmvq")
        return torch.zeros(X.shape[0], row, dtype=X.dtype)

    def deq(W, t, m, n, dtype):
        calls.append("dequant")
        return torch.zeros(m, n, dtype=dtype)

    monkeypatch.setattr(lin.ops, "ggml_mul_mat_a8", mmq)
    monkeypatch.setattr(lin.ops, "ggml_mul_mat_vec_a8", mmvq)
    monkeypatch.setattr(lin.ops, "ggml_dequantize", deq)
    return calls


def _q5k(rows, cols):
    import gguf
    bs, ts = gguf.GGML_QUANT_SIZES[WT.Q5_K]
    return torch.zeros(rows, cols // bs * ts, dtype=torch.uint8)


def test_large_prefill_chunk_uses_dequant_not_mmq(monkeypatch):
    calls = _record(monkeypatch)
    x = torch.zeros(1024, 5120, dtype=torch.bfloat16)
    y = lin._fused_mul_mat_gguf(x, _q5k(1024, 5120), WT.Q5_K)
    assert y.shape == (1024, 1024)
    assert "mmq" not in calls and "dequant" in calls


def test_small_batch_keeps_mmq(monkeypatch):
    calls = _record(monkeypatch)
    x = torch.zeros(64, 5120, dtype=torch.bfloat16)
    lin._fused_mul_mat_gguf(x, _q5k(1024, 5120), WT.Q5_K)
    assert calls == ["mmq"]


def test_threshold_boundary(monkeypatch):
    calls = _record(monkeypatch)
    m = lin._MMQ_MAX_M
    lin._fused_mul_mat_gguf(torch.zeros(m, 5120, dtype=torch.bfloat16), _q5k(1024, 5120), WT.Q5_K)
    lin._fused_mul_mat_gguf(torch.zeros(m + 1, 5120, dtype=torch.bfloat16), _q5k(1024, 5120), WT.Q5_K)
    assert calls == ["mmq", "dequant"]
