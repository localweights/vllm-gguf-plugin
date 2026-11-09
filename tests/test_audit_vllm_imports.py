# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``tools/audit_vllm_imports.py``.

The tool resolves every collected (module, symbol) pair in the *current*
interpreter.  These tests must not depend on a real vLLM install, and a fake
``vllm`` stub on ``sys.path`` is explicitly not allowed (it would prove nothing
about the real resolver), so the resolver entry point
``audit_vllm_imports.resolve_target`` is monkeypatched instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "audit_vllm_imports.py"

# One resolvable import (top level), one lazy duplicate of it, one bogus symbol
# and one import of an optional (not installed) package.
FAKE_PACKAGE_SOURCE = """\
from vllm.config import VllmConfig
from vllm.model_executor.definitely_missing import BogusSymbol
from vllm_omni.diffusion import OmniLoader


def _lazy_use():
    from vllm.config import VllmConfig as _config_cls

    return _config_cls
"""

# Monkeypatch targets + a module-level attribute chain, all collected as "attr".
FAKE_PATCH_SOURCE = """\
import vllm.config

patch("vllm.config.VllmConfig")


def _install():
    setattr(vllm.config, "VllmConfig", None)


vllm.config.VllmConfig
"""


def _load_tool():
    spec = importlib.util.spec_from_file_location("audit_vllm_imports", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool():
    return _load_tool()


def _make_package(tmp_path: Path, source: str) -> Path:
    package = tmp_path / "fake_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "module.py").write_text(source, encoding="utf-8")
    return package


def _fake_resolver(tool, calls):
    def resolve(module, symbol):
        calls.append((module, symbol))
        if module.startswith("vllm.config") and symbol in (None, "VllmConfig"):
            return tool.Resolution(tool.STATUS_OK)
        if module.startswith("vllm_omni"):
            return tool.Resolution(
                tool.STATUS_OPTIONAL_ABSENT, hint="optional root is not installed"
            )
        return tool.Resolution(
            tool.STATUS_MISSING_SYMBOL, hint="candidates: vllm.fake.module"
        )

    return resolve


@pytest.fixture
def calls():
    return []


@pytest.fixture
def tool_with_fake_resolver(tool, calls, monkeypatch):
    monkeypatch.setattr(tool, "resolve_target", _fake_resolver(tool, calls))
    return tool


def _rows(out: str) -> list[dict[str, str]]:
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        if cells[0] == "symbol" or set(cells[0]) <= set("-: "):
            continue
        rows.append(dict(zip(("symbol", "file_line", "kind", "status", "hint"), cells)))
    return rows


def _summary(out: str) -> str:
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[-1]


def test_report_statuses_kinds_and_summary(tool_with_fake_resolver, tmp_path, capsys):
    package = _make_package(tmp_path, FAKE_PACKAGE_SOURCE)

    assert tool_with_fake_resolver.main([str(package)]) == 0
    out = capsys.readouterr().out

    rows = _rows(out)
    assert len(rows) == 4, out
    by_symbol = {row["symbol"]: row for row in rows}

    resolvable = by_symbol["vllm.config.VllmConfig"]
    # Two rows for the same target: the top-level import and the lazy one.
    kinds = sorted(
        row["kind"] for row in rows if row["symbol"] == "vllm.config.VllmConfig"
    )
    assert kinds == ["lazy", "top"]
    assert all(
        row["status"] == tool_with_fake_resolver.STATUS_OK
        for row in rows
        if row["symbol"] == "vllm.config.VllmConfig"
    )
    assert resolvable["file_line"].startswith(str(package))

    bogus = by_symbol["vllm.model_executor.definitely_missing.BogusSymbol"]
    assert bogus["status"] == tool_with_fake_resolver.STATUS_MISSING_SYMBOL
    assert bogus["kind"] == "top"
    assert "vllm.fake.module" in bogus["hint"]

    optional = by_symbol["vllm_omni.diffusion.OmniLoader"]
    assert optional["status"] == tool_with_fake_resolver.STATUS_OPTIONAL_ABSENT

    assert _summary(out) == "SUMMARY ok=2 missing=1 optional=1"


def test_patch_and_setattr_targets_are_attr_kind(
    tool_with_fake_resolver, tmp_path, capsys
):
    package = _make_package(tmp_path, FAKE_PATCH_SOURCE)

    assert tool_with_fake_resolver.main([str(package)]) == 0
    out = capsys.readouterr().out

    rows = _rows(out)
    targets = [row for row in rows if row["symbol"] == "vllm.config.VllmConfig"]
    # patch("...") + setattr(vllm.config, "...") + module-level attribute chain.
    assert len(targets) == 3, out
    assert {row["kind"] for row in targets} == {"attr"}
    assert {row["status"] for row in targets} == {tool_with_fake_resolver.STATUS_OK}
    lines = sorted(int(row["file_line"].rsplit(":", 1)[1]) for row in targets)
    assert lines == [3, 7, 10]

    # The bare `import vllm.config` is checked as a module only.
    module_row = next(row for row in rows if row["symbol"] == "vllm.config")
    assert module_row["kind"] == "top"
    assert module_row["status"] == tool_with_fake_resolver.STATUS_OK

    assert _summary(out) == "SUMMARY ok=4 missing=0 optional=0"


def test_missing_package_dir_keeps_report_shape(tool, tmp_path, capsys):
    assert tool.main([str(tmp_path / "nope")]) == 0
    out = capsys.readouterr().out
    assert _rows(out) == []
    assert _summary(out) == "SUMMARY ok=0 missing=0 optional=0"
