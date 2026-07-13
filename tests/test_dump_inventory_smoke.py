"""Smoke tests for committed GGUF inventory fixtures.

Loads the JSON fixtures (no GGUF file access) and asserts basic invariants
that prove the dump script produced real, well-formed output.
"""

import json
import pathlib

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load_inventory(stem: str) -> list:
    return json.loads((FIXTURES / f"{stem}_inventory.json").read_text())


def _load_metadata(stem: str) -> dict:
    return json.loads((FIXTURES / f"{stem}_metadata.json").read_text())


# ── 27B fixture ────────────────────────────────────────────────────────────


def test_27b_inventory_tensor_count():
    inv = _load_inventory("qwen36_27b")
    assert len(inv) > 500, f"Expected >500 tensors, got {len(inv)}"


def test_27b_inventory_entry_shape():
    inv = _load_inventory("qwen36_27b")
    for entry in inv:
        assert "name" in entry, f"Missing 'name' key in entry: {entry}"
        assert "dtype" in entry, f"Missing 'dtype' key in entry: {entry}"
        assert "shape" in entry, f"Missing 'shape' key in entry: {entry}"
        assert isinstance(entry["name"], str)
        assert isinstance(entry["dtype"], str)
        assert isinstance(entry["shape"], list)


def test_27b_architecture():
    meta = _load_metadata("qwen36_27b")
    assert meta["general.architecture"] == "qwen35"


def test_27b_has_ssm_tensors():
    inv = _load_inventory("qwen36_27b")
    ssm_names = [t["name"] for t in inv if "ssm" in t["name"]]
    assert len(ssm_names) > 0, "No SSM tensors found — GDN block missing"


# ── 35B fixture ────────────────────────────────────────────────────────────


def test_35b_inventory_tensor_count():
    inv = _load_inventory("qwen36_35b")
    assert len(inv) > 500, f"Expected >500 tensors, got {len(inv)}"


def test_35b_inventory_entry_shape():
    inv = _load_inventory("qwen36_35b")
    for entry in inv:
        assert "name" in entry
        assert "dtype" in entry
        assert "shape" in entry


def test_35b_architecture():
    meta = _load_metadata("qwen36_35b")
    assert meta["general.architecture"] == "qwen35moe"


def test_35b_has_ssm_tensors():
    inv = _load_inventory("qwen36_35b")
    ssm_names = [t["name"] for t in inv if "ssm" in t["name"]]
    assert len(ssm_names) > 0, "No SSM tensors found — GDN block missing"