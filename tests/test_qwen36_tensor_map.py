# SPDX-License-Identifier: Apache-2.0
"""T-6: Qwen3.6 GGUF tensor-name map completion + MTP skip + typed error."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import gguf

from vllm_gguf_plugin.qwen35_config import map_qwen35_config
from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter
from vllm_gguf_plugin.weight_utils import partition_unmapped_gguf_tensors
from vllm_gguf_plugin.errors import GGUFUnmappedTensorError

_GGUF_PATH = "/mnt/scratch/gguf-models/Qwen3.6-27B-Q4_K_M.gguf"

# ── helpers ──────────────────────────────────────────────────────────


def _build_adapter_from_fixture(arch="qwen35", meta_path=None):
    """Build a GGUFWeightsAdapter with a config from fixture metadata."""
    if meta_path is None:
        meta_path = (
            Path(__file__).parent
            / "fixtures"
            / ("qwen36_27b_metadata.json" if arch == "qwen35" else "qwen36_35b_metadata.json")
        )
    meta = json.loads(meta_path.read_text())
    cfg = map_qwen35_config(arch, meta)
    adapter = GGUFWeightsAdapter(config=cfg)
    return adapter


def _build_model_config(adapter):
    """Build a vllm ModelConfig from the adapter's hf_config."""
    return SimpleNamespace(hf_config=adapter.config, trust_remote_code=False)


# ── real-GGUF tests (skip if file absent) ──────────────────────────


@pytest.mark.skipif(not os.path.exists(_GGUF_PATH), reason="Q4_K_M GGUF not present")
class TestBuildNameMapRealGGUF:
    """End-to-end tensor map building against the real 27B GGUF."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        meta = json.loads(
            (Path(__file__).parent / "fixtures" / "qwen36_27b_metadata.json").read_text()
        )
        cfg = map_qwen35_config("qwen35", meta)
        self.adapter = GGUFWeightsAdapter(config=cfg)

    def test_build_name_map_real_gguf(self):
        """Map must have >800 entries and blk.0.ssm_dt.bias must be present."""
        model_config = _build_model_config(self.adapter)
        name_map = self.adapter.build_name_map(model_config)
        assert len(name_map) > 800, f"Only {len(name_map)} entries mapped"
        assert "blk.0.ssm_dt.bias" in name_map, "dt_bias missing from map"

    def test_all_hf_params_mapped(self):
        """build_name_map must NOT raise (0 unmapped HF params)."""
        model_config = _build_model_config(self.adapter)
        # Should complete without RuntimeError
        name_map = self.adapter.build_name_map(model_config)
        # Every dt_bias should be present
        dt_bias_keys = [k for k in name_map if "ssm_dt.bias" in k]
        assert len(dt_bias_keys) == 48, f"Expected 48 dt_bias entries, got {len(dt_bias_keys)}"


# ── D4 partition unit tests ────────────────────────────────────────


class TestPartitionUnmappedGgufTensors:
    """Unit tests for the MTP-skip / fail-closed partition helper."""

    def test_mtp_skipped_and_others_raise(self):
        """MTP tensors (blk.N where N >= num_hidden_layers) are skipped;
        other unmapped tensors raise GGUFUnmappedTensorError."""
        gguf_names = {
            "mapped_tensor.weight",         # already mapped
            "blk.64.attn_q.weight",        # MTP — index 64 >= 64
            "blk.5.bogus.weight",          # unknown — should raise
        }
        mapped_keys = {"mapped_tensor.weight"}
        num_hidden_layers = 64
        arch = "qwen35"

        with pytest.raises(GGUFUnmappedTensorError) as exc_info:
            partition_unmapped_gguf_tensors(
                gguf_names, mapped_keys, num_hidden_layers, arch
            )
        assert "blk.5.bogus.weight" in str(exc_info.value)

    def test_mtp_only_no_error(self):
        """Only MTP tensors unmapped → no error, returns count of skipped."""
        gguf_names = {
            "mapped_tensor.weight",
            "blk.64.attn_q.weight",
            "blk.64.ssm_out.weight",
        }
        mapped_keys = {"mapped_tensor.weight"}
        num_hidden_layers = 64
        arch = "qwen35"

        skipped_count = partition_unmapped_gguf_tensors(
            gguf_names, mapped_keys, num_hidden_layers, arch
        )
        assert skipped_count == 2

    def test_non_qwen35_arch_raises_on_any_unmapped(self):
        """For non-qwen35/qwen35moe arches, ALL unmapped tensors raise."""
        gguf_names = {"mapped_tensor.weight", "blk.0.some_unknown.weight"}
        mapped_keys = {"mapped_tensor.weight"}
        num_hidden_layers = 64
        arch = "llama"

        with pytest.raises(GGUFUnmappedTensorError):
            partition_unmapped_gguf_tensors(
                gguf_names, mapped_keys, num_hidden_layers, arch
            )

    def test_no_unmapped_no_error(self):
        """Zero unmapped → returns 0 skipped, no error."""
        gguf_names = {"a.weight", "b.weight"}
        mapped_keys = {"a.weight", "b.weight"}
        num_hidden_layers = 64
        arch = "qwen35"

        skipped_count = partition_unmapped_gguf_tensors(
            gguf_names, mapped_keys, num_hidden_layers, arch
        )
        assert skipped_count == 0

    def test_mtp_boundary_exclusive(self):
        """blk.63 is NOT MTP for num_hidden_layers=64; only >=64 is MTP."""
        gguf_names = {"blk.63.attn_q.weight"}
        mapped_keys = set()
        num_hidden_layers = 64
        arch = "qwen35"

        with pytest.raises(GGUFUnmappedTensorError) as exc_info:
            partition_unmapped_gguf_tensors(
                gguf_names, mapped_keys, num_hidden_layers, arch
            )
        assert "blk.63.attn_q.weight" in str(exc_info.value)

    def test_qwen35moe_mtp_boundary(self):
        """For qwen35moe with 40 layers, blk.40+ is MTP."""
        gguf_names = {
            "blk.40.attn_q.weight",   # MTP — index 40 >= 40
            "blk.39.attn_q.weight",   # not MTP — should raise
        }
        mapped_keys = set()
        num_hidden_layers = 40
        arch = "qwen35moe"

        with pytest.raises(GGUFUnmappedTensorError) as exc_info:
            partition_unmapped_gguf_tensors(
                gguf_names, mapped_keys, num_hidden_layers, arch
            )
        assert "blk.39.attn_q.weight" in str(exc_info.value)
        assert "blk.40.attn_q.weight" not in str(exc_info.value)

# ── real-gguf regression: D4 partition must NOT raise (all remainder is MTP) ──
# Guards the ssm_a/A_log trailing-dot mapping bug chair caught: build the full
# map from a real GGUF, then run the fail-closed partition over ALL its tensors.

import json as _json
import types as _types
import os as _os

import pytest as _pytest

_Q4KM = "/mnt/scratch/gguf-models/Qwen3.6-27B-Q4_K_M.gguf"


@_pytest.mark.skipif(not _os.path.exists(_Q4KM), reason="canonical Q4_K_M gguf not present")
def test_real_gguf_all_tensors_mapped_or_mtp():
    import gguf
    from vllm_gguf_plugin.qwen35_config import map_qwen35_config, register
    from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter
    from vllm_gguf_plugin.weight_utils import partition_unmapped_gguf_tensors

    register()
    here = _os.path.dirname(__file__)
    meta = _json.load(open(_os.path.join(here, "fixtures", "qwen36_27b_metadata.json")))
    cfg = map_qwen35_config("qwen35", meta)
    mc = _types.SimpleNamespace(hf_config=cfg, trust_remote_code=False)
    name_map = GGUFWeightsAdapter(config=cfg).build_name_map(mc)

    names = {t.name for t in gguf.GGUFReader(_Q4KM).tensors}
    # must NOT raise GGUFUnmappedTensorError — every gguf tensor is mapped or MTP
    partition_unmapped_gguf_tensors(
        names, set(name_map), cfg.get_text_config().num_hidden_layers, "qwen35"
    )
    # and the GDN A_log tensor is mapped with the correct (no trailing dot) key
    assert "blk.0.ssm_a" in name_map
