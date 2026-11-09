#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit a plugin package's vLLM import surface.

Usage::

    python tools/audit_vllm_imports.py <package-dir>

Every ``*.py`` file below ``<package-dir>`` is walked with :mod:`ast` and each
reference into vLLM becomes one report row:

===========  ================================================================
kind         collected from
===========  ================================================================
``top``      ``from vllm... import X`` / ``import vllm...`` at module or class
             scope
``lazy``     the same imports inside a function body (lazy imports)
``attr``     monkeypatch targets -- ``setattr(vllm.a.b, "C", ...)`` and string
             targets passed to ``patch(...)`` / ``setattr(...)`` -- plus
             attribute chains ``vllm.a.b.C`` evaluated at module level (best
             effort)
===========  ================================================================

Each (module, symbol) pair is resolved in the *current* interpreter with
``importlib.import_module`` + ``getattr`` (a missing attribute is retried as a
submodule, so ``from pkg import submodule`` is not a false positive).  The
audited package itself is never imported, and nothing is written to
``sys.path``.

Statuses:

* ``OK`` -- resolved here.
* ``MISSING-MODULE`` -- the module itself does not exist.
* ``MISSING-SYMBOL`` -- the module imports, the name is not defined in it.
* ``IMPORT-ERROR(<type>)`` -- the module exists but raised while importing
  (missing dependency, GPU-only import, ...); not evidence of a fork break.
* ``OPTIONAL-ABSENT`` -- an optional root (``vllm_omni``) that is not
  installed.

Missing targets get a hint: up to three modules of the installed ``vllm``
tree whose files contain ``def <name>`` / ``class <name>`` / ``<name> =``.

Star imports carry no symbol to resolve and are skipped.  The report always
exits 0 -- it is a report, not a gate.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

STATUS_OK = "OK"
STATUS_MISSING_MODULE = "MISSING-MODULE"
STATUS_MISSING_SYMBOL = "MISSING-SYMBOL"
STATUS_IMPORT_ERROR = "IMPORT-ERROR"
STATUS_OPTIONAL_ABSENT = "OPTIONAL-ABSENT"

KIND_TOP = "top"
KIND_LAZY = "lazy"
KIND_ATTR = "attr"

# Packages that the audited code imports optionally (imported only when the
# deployment has them installed).  Extend this tuple, not the matching logic.
OPTIONAL_ROOTS = ("vllm_omni",)
_TARGET_ROOTS = ("vllm",) + OPTIONAL_ROOTS

# Callables whose string arguments name a dotted target to monkeypatch.
_PATCH_HELPERS = frozenset({"patch", "setattr"})

_HINT_LIMIT = 3


@dataclass
class Resolution:
    """Outcome of resolving one (module, symbol) pair."""

    status: str
    hint: str = ""


@dataclass
class Finding:
    """One vLLM reference found in the audited package."""

    symbol: str  # dotted target as reported, e.g. "vllm.config.VllmConfig"
    module: str  # module part handed to the resolver
    name: object  # symbol/attribute path handed to the resolver (None = module only)
    file: str  # display path, "<package-dir>/<relpath>"
    line: int
    kind: str
    chain: list = field(default_factory=list)  # attribute-chain parts, attr only


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _root_installed(root: str) -> bool:
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _import_module(name: str):
    """Import ``name``; return ``(module, None)`` or ``(None, exception)``."""
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # report the failure, never abort the audit
        return None, exc


def _failure_status(name: str, exc: BaseException) -> str:
    """Map an import exception to a status for the module called ``name``.

    A ``ModuleNotFoundError`` naming a *dependency* of an existing module is an
    import error, not a missing module.  ``exc.name`` is the module that could
    not be located, so it only means "missing module" when it is ``name`` or
    one of its parents.
    """
    if isinstance(exc, ModuleNotFoundError):
        missing = getattr(exc, "name", None) or ""
        if missing and (name == missing or name.startswith(missing + ".")):
            return STATUS_MISSING_MODULE
    return f"{STATUS_IMPORT_ERROR}({type(exc).__name__})"


def resolve_target(module: str, symbol=None) -> Resolution:
    """Resolve ``symbol`` (possibly dotted) inside ``module`` here and now.

    This is the single resolver entry point (monkeypatched by the tests).
    """
    root = module.split(".")[0]
    if root in OPTIONAL_ROOTS and not _root_installed(root):
        return Resolution(
            STATUS_OPTIONAL_ABSENT, hint=f"optional root '{root}' not installed"
        )

    obj, exc = _import_module(module)
    if obj is None:
        return Resolution(_failure_status(module, exc), hint=_hint(module, symbol))
    if not symbol:
        return Resolution(STATUS_OK)

    head, _, tail = symbol.partition(".")
    try:
        value = getattr(obj, head)
    except AttributeError:
        # ``from pkg import submodule`` -> the name lives as a submodule.
        submodule = module + "." + head
        sub, sub_exc = _import_module(submodule)
        if sub is not None:
            value = sub
        elif _failure_status(submodule, sub_exc) == STATUS_MISSING_MODULE:
            return Resolution(STATUS_MISSING_SYMBOL, hint=_hint(module, head))
        else:
            return Resolution(
                _failure_status(submodule, sub_exc), hint=_hint(module, head)
            )
    for part in tail.split(".") if tail else ():
        try:
            value = getattr(value, part)
        except AttributeError:
            return Resolution(STATUS_MISSING_SYMBOL, hint=_hint(module, part))
    return Resolution(STATUS_OK)


def resolve_chain(parts) -> Resolution:
    """Resolve ``vllm.a.b.C`` best effort: longest importable module prefix wins."""
    result = Resolution(STATUS_MISSING_MODULE)
    for split in range(len(parts) - 1, 0, -1):
        result = resolve_target(".".join(parts[:split]), ".".join(parts[split:]))
        if result.status != STATUS_MISSING_MODULE:
            return result
    return result


# ---------------------------------------------------------------------------
# hints
# ---------------------------------------------------------------------------

_VLLM_ROOT = None
_VLLM_ROOT_PROBED = False


def _vllm_root():
    """Directory of the installed ``vllm`` package (never imports it)."""
    global _VLLM_ROOT, _VLLM_ROOT_PROBED
    if not _VLLM_ROOT_PROBED:
        _VLLM_ROOT_PROBED = True
        try:
            spec = importlib.util.find_spec("vllm")
        except (ImportError, ValueError):
            spec = None
        locations = getattr(spec, "submodule_search_locations", None)
        if locations:
            _VLLM_ROOT = Path(list(locations)[0])
        elif spec is not None and spec.origin:
            _VLLM_ROOT = Path(spec.origin).parent
    return _VLLM_ROOT


def _module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["vllm", *parts])


def _search_installed_vllm(name: str, limit: int = _HINT_LIMIT) -> list:
    """Up to ``limit`` vllm modules that define ``name`` (def/class/assignment)."""
    root = _vllm_root()
    if root is None or not root.is_dir():
        return []
    escaped = re.escape(name)
    pattern = re.compile(
        rf"^\s*(?:(?:async\s+)?def|class)\s+{escaped}\b|^\s*{escaped}\s*(?::[^=\n]*)?=",
        re.MULTILINE,
    )
    hits = []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            hits.append(_module_name(root, path))
            if len(hits) >= limit:
                break
    return hits


def _hint(module: str, symbol) -> str:
    name = (symbol or "").split(".")[-1] or module.split(".")[-1]
    candidates = _search_installed_vllm(name)
    if candidates:
        return f"installed vllm defines {name} in: {', '.join(candidates)}"
    return f"no def/class/assignment of '{name}' in installed vllm"


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _is_target_module(module: str) -> bool:
    root = module.split(".")[0]
    if root not in _TARGET_ROOTS:
        return False
    return module == root or module.startswith(root + ".")


def _callee_name(func) -> object:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _string_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _root_chain(node):
    """``vllm.a.b`` -> ``["vllm", "a", "b"]``; ``None`` for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in _TARGET_ROOTS:
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _target(module: str, name) -> str:
    return module if not name else module + "." + name


class _Collector(ast.NodeVisitor):
    """Collects vLLM references from one module's AST."""

    def __init__(self, display_path: str):
        self.display_path = display_path
        self.findings = []
        self._depth = 0  # >0 while inside a function body
        self._stack = []
        self._seen = set()

    def collect(self, tree) -> list:
        self.visit(tree)
        return self.findings

    def generic_visit(self, node):
        self._stack.append(node)
        super().generic_visit(node)
        self._stack.pop()

    def _kind(self) -> str:
        return KIND_LAZY if self._depth else KIND_TOP

    def _add_import(self, module: str, name, line: int) -> None:
        self._record(
            Finding(
                symbol=_target(module, name),
                module=module,
                name=name,
                file=self.display_path,
                line=line,
                kind=self._kind(),
            )
        )

    def _add_attr(self, parts, line: int) -> None:
        if not parts:
            return
        self._record(
            Finding(
                symbol=".".join(parts),
                module=".".join(parts[:-1]),
                name=parts[-1],
                file=self.display_path,
                line=line,
                kind=KIND_ATTR,
                chain=list(parts),
            )
        )

    def _record(self, finding: Finding) -> None:
        key = (finding.symbol, finding.file, finding.line, finding.kind)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(finding)

    # -- imports ----------------------------------------------------------
    def visit_Import(self, node) -> None:
        for alias in node.names:
            if _is_target_module(alias.name):
                self._add_import(alias.name, None, node.lineno)

    def visit_ImportFrom(self, node) -> None:
        if node.level or not node.module or not _is_target_module(node.module):
            return
        for alias in node.names:
            if alias.name == "*":  # nothing concrete to resolve
                continue
            self._add_import(node.module, alias.name, node.lineno)

    # -- scopes -----------------------------------------------------------
    def visit_FunctionDef(self, node) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._visit_function(node)

    def _visit_function(self, node) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    # -- monkeypatch targets and attribute chains -------------------------
    def visit_Call(self, node) -> None:
        helper = _callee_name(node.func)
        if helper in _PATCH_HELPERS:
            self._collect_patch_target(node, helper)
        self.generic_visit(node)

    def _collect_patch_target(self, node, helper: str) -> None:
        # setattr(vllm.mod, "Name", ...) -> the object names the module, the
        # string names the symbol.
        if helper == "setattr" and len(node.args) >= 2:
            parts = _root_chain(node.args[0])
            name = _string_const(node.args[1])
            if parts and name:
                self._add_attr(parts + name.split("."), node.lineno)
                return
        # patch("vllm.a.b.C") / monkeypatch.setattr("vllm.a.b.C", ...).
        for argument in list(node.args) + [kw.value for kw in node.keywords]:
            target = _string_const(argument)
            if not target:
                continue
            parts = target.split(".")
            # At least root.module.symbol is needed; a bare module target has no
            # symbol to check.
            if len(parts) >= 3 and _is_target_module(target):
                self._add_attr(parts, node.lineno)

    def visit_Attribute(self, node) -> None:
        parent = self._stack[-1] if self._stack else None
        if isinstance(parent, ast.Attribute) and parent.value is node:
            return  # prefix of a longer chain, reported as a whole
        parts = _root_chain(node)
        if parts and len(parts) >= 2 and self._depth == 0:
            self._add_attr(parts, node.lineno)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# audit + report
# ---------------------------------------------------------------------------


def audit(package_dir: str) -> list:
    """Collect and resolve every vLLM reference below ``package_dir``."""
    root = Path(package_dir)
    display_base = Path(os.path.normpath(package_dir))
    findings = []
    for path in sorted(root.rglob("*.py")):
        display = str(display_base / path.relative_to(root))
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="replace"), filename=display
            )
        except (OSError, SyntaxError):
            continue  # unparseable file: nothing to say, never crash the report
        findings.extend(_Collector(display).collect(tree))
    findings.sort(key=lambda f: (f.file, f.line, f.symbol, f.kind))

    cache = {}
    rows = []
    for finding in findings:
        key = (
            ("chain", tuple(finding.chain))
            if finding.chain
            else ("import", finding.module, finding.name)
        )
        if key not in cache:
            cache[key] = (
                resolve_chain(finding.chain)
                if finding.chain
                else resolve_target(finding.module, finding.name)
            )
        rows.append((finding, cache[key]))
    return rows


def render(rows) -> str:
    """Markdown report; the final line is the SUMMARY instead of the counts."""
    lines = [
        "| symbol | file:line | kind | status | hint |",
        "| --- | --- | --- | --- | --- |",
    ]
    ok = missing = optional = 0
    for finding, resolution in rows:
        if resolution.status == STATUS_OK:
            ok += 1
        elif resolution.status == STATUS_OPTIONAL_ABSENT:
            optional += 1
        else:
            missing += 1
        hint = (resolution.hint or "-").replace("|", "/")
        lines.append(
            f"| {finding.symbol} | {finding.file}:{finding.line} "
            f"| {finding.kind} | {resolution.status} | {hint} |"
        )
    lines.append("")
    lines.append(f"SUMMARY ok={ok} missing={missing} optional={optional}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write(
            "usage: audit_vllm_imports.py <package-dir>  "
            "(audits the vLLM import surface of <package-dir>)\n"
        )
        return 0  # report tool: exit 0 even on bad usage
    package_dir = args[0]
    if not Path(package_dir).is_dir():
        sys.stderr.write(f"audit_vllm_imports: not a directory: {package_dir}\n")
        print(render([]))
        return 0
    print(render(audit(package_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
