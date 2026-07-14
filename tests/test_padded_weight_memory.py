# SPDX-License-Identifier: Apache-2.0
# Memory-efficient merged-layer padding tests for GGUF plugin.
# CPU-only — tests _create_padded_weight_param without GPU.

import torch

import vllm.model_executor.parameter as parameter_module
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
from vllm_gguf_plugin.quantization.params import GGUFWeightParameter


class _FakeLayer:
    """Minimal layer object with a qweight parameter."""

    def __init__(self, qweight):
        self.qweight = qweight

    def register_parameter(self, name, param):
        setattr(self, name, param)


class _DataContainerSpy(list):
    """List wrapper that records __setitem__ calls for verification."""

    def __init__(self, data):
        super().__init__(data)
        self.setitem_calls = []  # (index, value) pairs

    def __setitem__(self, index, value):
        self.setitem_calls.append((index, value))
        super().__setitem__(index, value)


def _make_qweight(shard_tensors, shard_ids, shard_id_map, weight_loader=lambda p, w, s=None: None):
    """Build a GGUFWeightParameter populated with shards (CPU)."""
    device = shard_tensors[0].device
    dummy_data = torch.empty(0, dtype=shard_tensors[0].dtype, device=device)
    qweight = GGUFWeightParameter(
        data=dummy_data,
        weight_loader=weight_loader,
        input_dim=1,
        output_dim=0,
        tensor_shape=(sum(t.size(0) for t in shard_tensors), shard_tensors[0].size(1)),
    )
    qweight.data_container = shard_tensors
    qweight.shard_id = shard_ids
    qweight.shard_id_map = shard_id_map
    return qweight


# ── Test 1: layout unchanged ────────────────────────────────────────────


def test_padded_layout_unchanged(monkeypatch):
    """Shards of differing widths are row-concatenated with zero-padding
    to max width, in _gguf_ordered_shard_ids order.  shard_offset_map
    must have the exact (start, end, size) tuples."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)

    shard_q = torch.full((4, 8), 1.0)
    shard_k = torch.full((2, 6), 2.0)
    shard_v = torch.full((3, 8), 3.0)

    shard_ids = ["q", "v", "k"]  # out of order — should sort to q,k,v
    shard_id_map = {"q": 0, "k": 1, "v": 2}
    shards = [shard_q, shard_k, shard_v]

    qweight = _make_qweight(shards, shard_ids, shard_id_map)
    layer = _FakeLayer(qweight)

    method = GGUFLinearMethod(quant_config=None)
    method._create_padded_weight_param(layer)

    result = layer.qweight.data  # CPU tensor
    assert result.shape == (9, 8), f"Expected (9, 8) got {result.shape}"

    # Rows 0-3: shard_q (value 1.0), no padding needed (width 8)
    assert (result[0:4, 0:8] == 1.0).all()
    # Rows 4-5: shard_k (value 2.0) in cols 0-5, cols 6-7 zero-padded
    assert (result[4:6, 0:6] == 2.0).all()
    assert (result[4:6, 6:8] == 0.0).all()
    # Rows 6-8: shard_v (value 3.0), no padding needed (width 8)
    assert (result[6:9, 0:8] == 3.0).all()

    # shard_offset_map
    offset_map = getattr(layer.qweight, "shard_offset_map", None)
    assert offset_map is not None
    assert offset_map["q"] == (0, 4, 8)
    assert offset_map["k"] == (4, 6, 6)
    assert offset_map["v"] == (6, 9, 8)


# ── Test 2: shards released during copy ─────────────────────────────────


def test_shards_released_during_copy(monkeypatch):
    """Each shard slot in data_container is set to None *after its copy*,
    not deferred to a bulk clear at the end of the loop."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)

    shard_a = torch.ones((4, 8))
    shard_b = torch.ones((2, 6))
    shard_c = torch.ones((3, 8))

    shard_ids = ["q", "k", "v"]
    shard_id_map = {"q": 0, "k": 1, "v": 2}
    shard_tensors = [shard_a, shard_b, shard_c]

    spy = _DataContainerSpy(shard_tensors)

    qweight = _make_qweight(shard_tensors, shard_ids, shard_id_map)
    qweight.data_container = spy  # swap in the spy

    layer = _FakeLayer(qweight)

    method = GGUFLinearMethod(quant_config=None)
    method._create_padded_weight_param(layer)

    # Check that setitem was called to set each slot to None
    none_indices = [idx for idx, val in spy.setitem_calls if val is None]
    assert 0 in none_indices, "shard 0 was not set to None"
    assert 1 in none_indices, "shard 1 was not set to None"
    assert 2 in none_indices, "shard 2 was not set to None"


# ── Test 3: single-shard no-op ──────────────────────────────────────────


def test_single_shard_noop(monkeypatch):
    """Container of length 1 — method is a no-op, leaves qweight unchanged."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)

    shard = torch.ones((4, 8))

    qweight = _make_qweight([shard], [0], {0: 0})
    original_qweight_id = id(qweight)
    layer = _FakeLayer(qweight)

    method = GGUFLinearMethod(quant_config=None)
    method._create_padded_weight_param(layer)

    assert id(layer.qweight) == original_qweight_id, "Single-shard path replaced qweight"