"""WS1 D3/R3: assert the LinearBase verdicts for Qwen3.5 GDN modules.

If a vLLM refactor changes any of these classes, GGUF weight loading for the GDN
layers breaks — this test is the early-warning signal. Runs in the dev venv (needs vllm).
"""

import importlib.util
import pathlib

import pytest

pytest.importorskip("vllm")

_TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "introspect_gdn_modules.py"
_spec = importlib.util.spec_from_file_location("introspect_gdn_modules", _TOOL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_linearbase_params_are_linearbase():
    for name in ("conv1d", "in_proj_qkvz", "in_proj_ba", "out_proj"):
        assert _mod.linearbase_verdict(_mod.GDN_MODULE_CLASSES[name]), f"{name} must be LinearBase"


def test_dense_params_are_not_linearbase():
    for name in ("dt_bias", "A_log", "norm"):
        assert not _mod.linearbase_verdict(_mod.GDN_MODULE_CLASSES[name]), f"{name} must be dense"


def test_doc_generation(tmp_path):
    out = tmp_path / "gdn.md"
    doc = _mod.build_doc(None)
    out.write_text(doc)
    assert "conv1d" in doc and "LinearBase" in doc
    # every GDN param appears in the table
    for name in _mod.GDN_MODULE_CLASSES:
        assert f"`{name}`" in doc
