# SPDX-License-Identifier: Apache-2.0
"""T-4: Qwen3.6 (qwen35 / qwen35moe) GGUF → HF config mapping tests.

Fixture-driven — loads the committed metadata dicts and asserts every value
from docs/qwen35-config-mapping.md.  Also verifies register() idempotency.
"""

import json
from pathlib import Path

import pytest
import transformers.modeling_gguf_pytorch_utils as gguf_utils

from vllm_gguf_plugin.qwen35_config import map_qwen35_config, register

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ── 27B dense (qwen35) ──────────────────────────────────────────────


class Test27BDense:
    """qwen35  — 65 blocks, 1 nextn → 64 hidden layers, interval=4."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.meta = _load_fixture("qwen36_27b_metadata.json")
        self.cfg = map_qwen35_config("qwen35", self.meta)

    def test_model_type(self):
        assert self.cfg.model_type in ("qwen3_5_text", "qwen3_5")

    def test_num_hidden_layers(self):
        # 65 block_count − 1 nextn_predict_layers = 64
        assert self.cfg.num_hidden_layers == 64

    def test_max_position_embeddings(self):
        assert self.cfg.max_position_embeddings == 262144

    def test_hidden_size(self):
        assert self.cfg.hidden_size == 5120

    def test_intermediate_size(self):
        assert self.cfg.intermediate_size == 17408

    def test_attention_heads(self):
        assert self.cfg.num_attention_heads == 24
        assert self.cfg.num_key_value_heads == 4

    def test_head_dim(self):
        # full-attention head dim = attention.key_length
        assert self.cfg.head_dim == 256

    def test_rope_theta(self):
        # rope_theta lives inside rope_parameters dict on this config
        assert self.cfg.rope_parameters["rope_theta"] == 10000000.0

    def test_linear_conv_kernel_dim(self):
        assert self.cfg.linear_conv_kernel_dim == 4

    def test_linear_key_value_head_dim(self):
        assert self.cfg.linear_key_head_dim == 128
        assert self.cfg.linear_value_head_dim == 128

    def test_linear_num_key_heads(self):
        assert self.cfg.linear_num_key_heads == 16

    def test_linear_num_value_heads(self):
        # inner_size 6144 // state_size 128 = 48
        assert self.cfg.linear_num_value_heads == 48

    def test_layer_types(self):
        lt = self.cfg.layer_types
        assert len(lt) == 64
        full = sum(1 for t in lt if t == "full_attention")
        linear = sum(1 for t in lt if t != "full_attention")
        assert full == 16
        assert linear == 48


# ── 35B MoE (qwen35moe) ────────────────────────────────────────────


class Test35BMoE:
    """qwen35moe  — 41 blocks, 1 nextn → 40 hidden layers, interval=4."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.meta = _load_fixture("qwen36_35b_metadata.json")
        self.cfg = map_qwen35_config("qwen35moe", self.meta)

    def test_model_type(self):
        assert self.cfg.model_type in ("qwen3_5_moe_text", "qwen3_5_moe")

    def test_num_hidden_layers(self):
        # 41 block_count − 1 nextn_predict_layers = 40
        assert self.cfg.num_hidden_layers == 40

    def test_hidden_size(self):
        assert self.cfg.hidden_size == 2048

    def test_attention_heads(self):
        assert self.cfg.num_attention_heads == 16
        assert self.cfg.num_key_value_heads == 2

    def test_moe_experts(self):
        assert self.cfg.num_experts == 256
        assert self.cfg.num_experts_per_tok == 8

    def test_linear_num_value_heads(self):
        # inner_size 4096 // state_size 128 = 32
        assert self.cfg.linear_num_value_heads == 32

    def test_layer_types(self):
        lt = self.cfg.layer_types
        assert len(lt) == 40
        full = sum(1 for t in lt if t == "full_attention")
        linear = sum(1 for t in lt if t != "full_attention")
        assert full == 10
        assert linear == 30


# ── registration ─────────────────────────────────────────────────────


class TestRegister:
    def test_qwen35_in_supported_architectures(self):
        register()
        assert "qwen35" in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES

    def test_qwen35moe_in_supported_architectures(self):
        register()
        assert "qwen35moe" in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES

    def test_idempotent(self):
        register()
        register()  # must not raise, must not duplicate

# ── real-load path (local only; needs the ~15GB GGUF, skipped in CI) ──────────

import os
import pytest

_Q4KM = "/mnt/scratch/gguf-models/Qwen3.6-27B-Q4_K_M.gguf"


@pytest.mark.skipif(not os.path.exists(_Q4KM), reason="canonical Q4_K_M gguf not present")
def test_real_gguf_config_load_via_wrapped_loader():
    """End-to-end: register() + the wrapped load_gguf_checkpoint produce the
    correct config directly from a real Qwen3.6-27B GGUF (not the fixture)."""
    import transformers.modeling_gguf_pytorch_utils as gu
    from vllm_gguf_plugin.qwen35_config import register

    register()
    register()  # idempotent — must not double-wrap
    res = gu.load_gguf_checkpoint(_Q4KM, return_tensors=False)
    cfg = res["config"]
    assert cfg.model_type == "qwen3_5_text"
    assert cfg.num_hidden_layers == 64
    assert cfg.hidden_size == 5120
    assert cfg.head_dim == 256
    assert cfg.linear_num_key_heads == 16
    assert cfg.linear_num_value_heads == 48
    assert cfg.layer_types.count("full_attention") == 16
    assert sum(1 for t in cfg.layer_types if t != "full_attention") == 48
