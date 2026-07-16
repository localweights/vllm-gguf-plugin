# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only tests: merged-linear shards stage on CPU, not GPU.

Verifies that _store_gguf_loaded_weight with shard_id keeps tensors on CPU,
and _create_padded_weight_param builds the final padded tensor from CPU
source shards (zero transient GPU allocations in the merged path).
"""

import torch

import vllm.model_executor.parameter as parameter_module
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
from vllm_gguf_plugin.quantization.params import (
    GGUFWeightParameter,
    _store_gguf_loaded_weight,
)


class _FakeLayer:
    """Minimal layer object with a qweight parameter."""

    def __init__(self, qweight):
        self.qweight = qweight

    def register_parameter(self, name, param):
        setattr(self, name, param)


# ── Test 1: _store_gguf_loaded_weight keeps shards on CPU ──────────────


def test_store_shards_on_cpu(monkeypatch):
    """All tensors in data_container after _store with shard_id are CPU."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    param = GGUFWeightParameter(
        data=torch.empty(0, dtype=torch.float32),
        weight_loader=lambda p, w, s=None: None,
        input_dim=1,
        output_dim=0,
        tensor_shape=(9, 8),
    )

    _store_gguf_loaded_weight(param, torch.ones((4, 8)), shard_id="q")
    _store_gguf_loaded_weight(param, torch.full((2, 6), 2.0), shard_id="k")
    _store_gguf_loaded_weight(param, torch.full((3, 8), 3.0), shard_id="v")

    assert len(param.data_container) == 3
    for t in param.data_container:
        assert t.device.type == "cpu", f"Expected CPU, got {t.device}"
    assert param.shard_id == ["q", "k", "v"]
    assert param.shard_id_map == {"q": 0, "k": 1, "v": 2}


# ── Test 2: _create_padded_weight_param on CPU shards ───────────────────


def test_padded_from_cpu_shards(monkeypatch):
    """_create_padded_weight_param concatenates CPU shards with zero-padding
    to max width.  Result is on CPU when CUDA is absent; shard_offset_map is
    correct; data_container emptied."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    param = GGUFWeightParameter(
        data=torch.empty(0, dtype=torch.float32),
        weight_loader=lambda p, w, s=None: None,
        input_dim=1,
        output_dim=0,
        tensor_shape=(9, 8),
    )

    shard_q = torch.full((4, 8), 1.0)
    shard_k = torch.full((2, 6), 2.0)
    shard_v = torch.full((3, 8), 3.0)

    _store_gguf_loaded_weight(param, shard_q, shard_id="q")
    _store_gguf_loaded_weight(param, shard_k, shard_id="k")
    _store_gguf_loaded_weight(param, shard_v, shard_id="v")

    layer = _FakeLayer(param)
    method = GGUFLinearMethod(quant_config=None)
    method._create_padded_weight_param(layer)

    result = layer.qweight.data
    assert result.shape == (9, 8), f"Expected (9, 8) got {result.shape}"

    # Row ranges with correct values (device is GPU if CUDA available, CPU otherwise)
    assert (result[0:4, 0:8] == 1.0).all(), "shard q mismatch"
    assert (result[4:6, 0:6] == 2.0).all(), "shard k values mismatch"
    # shard k width-padded columns
    assert (result[4:6, 6:8] == 0.0).all(), "shard k padding mismatch"
    assert (result[6:9, 0:8] == 3.0).all(), "shard v mismatch"

    offset_map = getattr(layer.qweight, "shard_offset_map", None)
    assert offset_map is not None
    assert offset_map["q"] == (0, 4, 8)
    assert offset_map["k"] == (4, 6, 6)
    assert offset_map["v"] == (6, 9, 8)

    assert len(layer.qweight.data_container) == 0, "container not emptied"


# ── Test 3: single-shard case — staging tensor moved to param.data ──────


def test_single_shard_staging(monkeypatch):
    """Single CPU shard: _create_padded_weight_param moves it to param.data
    and clears the container."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    param = GGUFWeightParameter(
        data=torch.empty(0, dtype=torch.float32),
        weight_loader=lambda p, w, s=None: None,
        input_dim=1,
        output_dim=0,
        tensor_shape=(4, 8),
    )

    _store_gguf_loaded_weight(param, torch.ones((4, 8)), shard_id=0)

    assert len(param.data_container) == 1
    assert param.data_container[0].device.type == "cpu"

    layer = _FakeLayer(param)
    method = GGUFLinearMethod(quant_config=None)
    method._create_padded_weight_param(layer)

    result = layer.qweight.data
    assert result.shape == (4, 8), f"Expected (4, 8) got {result.shape}"
    assert (result == 1.0).all(), "single shard values mismatch"
    assert len(layer.qweight.data_container) == 0, "container not emptied"
