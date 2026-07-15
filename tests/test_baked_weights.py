# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for baked weight cache (gguf→safetensors prebake)."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from vllm_gguf_plugin.baked_weights import (
    _bake_meta_path,
    bake_meta,
    bake_path,
    is_bake_valid,
    load_bake,
    save_bake,
)


def _make_fake_gguf(tmp_path: Path, mtime_ns: int | None = None) -> Path:
    p = tmp_path / "model.gguf"
    p.touch()
    if mtime_ns is not None:
        os.utime(p, ns=(mtime_ns, mtime_ns))
    return p


class TestBakeRoundTrip:
    """save_bake → is_bake_valid true → load_bake round-trips values/dtypes/names."""

    def test_round_trip_simple(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 8, dtype=torch.float16),
            "model.layers.0.self_attn.v_proj.weight": torch.ones(3, 5, dtype=torch.bfloat16),
        }
        save_bake(gguf, tensors)
        assert is_bake_valid(gguf)
        loaded = load_bake(gguf)
        assert set(loaded.keys()) == set(tensors.keys())
        for name in tensors:
            assert loaded[name].dtype == tensors[name].dtype
            assert loaded[name].shape == tensors[name].shape
            assert torch.equal(loaded[name], tensors[name])

    def test_round_trip_empty(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {})
        assert is_bake_valid(gguf)
        loaded = load_bake(gguf)
        assert loaded == {}

    def test_round_trip_quant_shapes(self, tmp_path):
        """Preserve packed-quant shapes (uint8 blocks)."""
        gguf = _make_fake_gguf(tmp_path)
        tensors = {
            "model.layers.0.mlp.gate_proj.qweight": torch.randint(0, 255, (16, 32), dtype=torch.uint8),
            "model.layers.0.mlp.gate_proj.scales": torch.randn(4, 4, dtype=torch.float16),
        }
        save_bake(gguf, tensors)
        loaded = load_bake(gguf)
        for name in tensors:
            assert loaded[name].dtype == tensors[name].dtype
            assert loaded[name].shape == tensors[name].shape
            assert torch.equal(loaded[name], tensors[name])


class TestBakeInvalidation:
    """Meta mismatch → invalid."""

    def test_mtime_change_invalidates(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path, mtime_ns=1_000_000_000)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        assert is_bake_valid(gguf)
        # Change mtime on the gguf file
        os.utime(gguf, ns=(2_000_000_000, 2_000_000_000))
        assert not is_bake_valid(gguf)

    def test_missing_meta_json_invalid(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        meta = _bake_meta_path(gguf)
        meta.unlink()
        assert not is_bake_valid(gguf)

    def test_corrupt_meta_json_invalid(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        meta = _bake_meta_path(gguf)
        meta.write_text("{corrupt")
        assert not is_bake_valid(gguf)
        # is_bake_valid must not raise
        _ = is_bake_valid(gguf)

    def test_missing_safetensors_invalid(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        bake_path(gguf).unlink()
        assert not is_bake_valid(gguf)

    def test_wrong_plugin_version_invalid(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        meta = _bake_meta_path(gguf)
        data = json.loads(meta.read_text())
        data["plugin_version"] = "0.0.3"
        meta.write_text(json.dumps(data))
        assert not is_bake_valid(gguf)

    def test_wrong_gguf_size_invalid(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        meta = _bake_meta_path(gguf)
        data = json.loads(meta.read_text())
        data["gguf_size"] = data["gguf_size"] + 1
        meta.write_text(json.dumps(data))
        assert not is_bake_valid(gguf)


class TestKillSwitch:
    """VLLM_GGUF_BAKED_CACHE=0 disables."""

    def test_kill_switch_disables(self, tmp_path):
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        assert is_bake_valid(gguf)
        with patch.dict(os.environ, {"VLLM_GGUF_BAKED_CACHE": "0"}):
            assert not is_bake_valid(gguf)

    def test_kill_switch_default_enabled(self, tmp_path):
        """Default (unset) must allow valid."""
        gguf = _make_fake_gguf(tmp_path)
        save_bake(gguf, {"w": torch.tensor([1.0])})
        # Ensure env is not set
        if "VLLM_GGUF_BAKED_CACHE" in os.environ:
            del os.environ["VLLM_GGUF_BAKED_CACHE"]
        assert is_bake_valid(gguf)


class TestCrashSafety:
    """save_bake writes meta LAST so crash mid-write never leaves valid bake."""

    def test_meta_written_after_safetensors(self, tmp_path, monkeypatch):
        """Assert meta mtime >= safetensors mtime after save_bake."""
        gguf = _make_fake_gguf(tmp_path)
        tensors = {"w": torch.tensor([42.0])}
        save_bake(gguf, tensors)
        st = bake_path(gguf)
        meta = _bake_meta_path(gguf)
        st_mtime = st.stat().st_mtime_ns
        meta_mtime = meta.stat().st_mtime_ns
        assert meta_mtime >= st_mtime, (
            f"Meta mtime {meta_mtime} < safetensors mtime {st_mtime}; "
            "meta should be written last"
        )

    def test_crash_mid_write_no_meta(self, tmp_path, monkeypatch):
        """If safetensors save crashes, no meta written → invalid."""
        gguf = _make_fake_gguf(tmp_path)
        with patch(
            "vllm_gguf_plugin.baked_weights.safetensors.torch.save_file",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError):
                save_bake(gguf, {"w": torch.tensor([1.0])})
        # No meta should exist
        meta = _bake_meta_path(gguf)
        assert not meta.exists(), "Meta should not exist after failed save"
        # Bake must be invalid
        assert not is_bake_valid(gguf)

