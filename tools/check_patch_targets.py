#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that every upstream callable the plugin monkeypatches still exists.

Usage::

    python tools/check_patch_targets.py <package-dir>

The plugin fork survives vLLM upgrades by replacing/augmenting upstream
attributes (``EngineArgs.create_model_config = ...``,
``SpeculativeConfig.hf_config_override = ...``,
``_orig = SomeVllmClass.method``).  Each of those is a hard dependency on an
upstream *symbol*, and a rename upstream is a silent runtime break (worse than
an ImportError).  This tool finds them statically and resolves them in the
*current* interpreter:

* assignment to an attribute of an imported upstream object,
  ``EngineArgs.create_model_config = wrapper`` — checked (must exist: it is
  what we replace, and ``@wraps`` reads it);
* ``setattr(<vllm obj>, "name", ...)``;
* a read of an upstream attribute, e.g. ``_orig = Cls.method`` — checked (the
  wrapper needs the attribute to bind);
* sentinel/flag assignments whose value is a constant
  (``Cls._gguf_x_patched = True``) are NOT checked: the name is created by the
  plugin itself.

Only attribute assignments *without* a constant value are reported, so the
plugin's own marker attributes never show up as false failures.

Output is one markdown row per target::

    | target | file:line | exists | signature |

``exists`` is ``yes`` / ``NO`` / ``skip-absent`` (an optional root such as
``vllm_omni`` that is not installed here) / ``inert`` (upstream removed the
symbol on purpose and the plugin handles it — declared with a
``# patch-target: inert`` comment on the patch site or the line above it) /
``guarded`` (the symbol only exists on some supported vLLM versions and the
import is wrapped in ``try/except ImportError`` — declared with
``# patch-target: version-guarded``).

Statements inside an ``except ImportError`` handler are treated as version
*fallbacks*: a symbol that only the old vLLM location exports is expected to be
absent on the new one, so those reads are not reported as failures.

Exit status is 1 if any target that must exist does not, 0 otherwise.  The
report is written to stdout.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Roots whose symbols the plugin patches.  ``vllm_omni`` is optional: the
# plugin's diffusion integration is imported behind ``try/except ImportError``.
PATCH_ROOTS = ("vllm", "transformers", "vllm_omni")
OPTIONAL_ROOTS = ("vllm_omni",)

EXISTS_YES = "yes"
EXISTS_NO = "NO"
EXISTS_SKIP = "skip-absent"
EXISTS_INERT = "inert"
EXISTS_GUARDED = "guarded"

# Declares "upstream removed this on purpose; the plugin handles its absence".
INERT_MARKER = "patch-target: inert"
# Declares "this symbol only exists on some supported vLLM versions; the import
# site is wrapped in try/except ImportError".
GUARDED_MARKER = "patch-target: version-guarded"
# Names whose ``except`` body is a version fallback for a moved/removed symbol.
_FALLBACK_EXCEPTIONS = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)


@dataclass(frozen=True)
class Target:
    """One upstream attribute the plugin patches or reads."""

    path: str  # fully-qualified dotted path, e.g. vllm.config.VllmConfig.foo
    file: str
    line: int


# ---------------------------------------------------------------------------
# static collection
# ---------------------------------------------------------------------------


def _dotted(node) -> str | None:
    """Flatten ``a.b.c`` to ``"a.b.c"`` (None for anything else)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class _Scope:
    """Local name -> dotted upstream path (imports and simple aliases)."""

    def __init__(self, parent: dict[str, str] | None = None):
        self.names: dict[str, str] = dict(parent or {})

    def resolve(self, expr) -> str | None:
        """Resolve an expression to a dotted path through this scope."""
        dotted = _dotted(expr)
        if dotted is None:
            return None
        root, _, rest = dotted.partition(".")
        base = self.names.get(root)
        if base is None:
            return None
        return base + ("." + rest if rest else "")


def _root(path: str) -> str:
    return path.split(".", 1)[0]


def _is_patch_root(path: str) -> bool:
    return _root(path) in PATCH_ROOTS


def _is_constant(node) -> bool:
    """True for literal values (the plugin's own idempotency flags)."""
    return isinstance(node, ast.Constant)


def _collect_scope(body, scope: _Scope, file: str, out: list, fallback=False) -> None:
    """Walk a statement list, recording imports/aliases and patch targets.

    Only functions and classes open a new name scope; ``try``/``if``/``with``/
    ``for`` bodies run in the enclosing scope (that is what Python does), so an
    import inside ``try:`` is visible to later statements of the same function.

    ``fallback`` marks the body of an ``except ImportError`` handler: targets
    there are old-location fallbacks, not patch sites.
    """
    for stmt in body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if not _is_patch_root(alias.name):
                    continue
                if alias.asname:
                    scope.names[alias.asname] = alias.name
                else:
                    local = alias.name.split(".", 1)[0]
                    scope.names[local] = local
            continue

        if isinstance(stmt, ast.ImportFrom):
            if stmt.level or not stmt.module or not _is_patch_root(stmt.module):
                continue
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                scope.names[alias.asname or alias.name] = (
                    stmt.module + "." + alias.name
                )
            continue

        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            if not fallback:
                _collect_assign(stmt, scope, file, out)
            # Aliases (``P = mod.Cls``) must be visible to later statements.
            targets = (
                stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            )
            value = stmt.value
            resolved = scope.resolve(value) if value is not None else None
            if resolved is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        scope.names[target.id] = resolved
            continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if not fallback:
                _collect_setattr_call(stmt.value, stmt.lineno, scope, file, out)
            continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect_scope(
                stmt.body, _Scope(scope.names), file, out, fallback=fallback
            )
            continue

        if isinstance(stmt, ast.ExceptHandler):
            _collect_scope(
                stmt.body,
                scope,
                file,
                out,
                fallback=fallback or _is_import_fallback(stmt),
            )
            continue

        for field_name in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field_name, None)
            if isinstance(child, list):
                _collect_scope(child, scope, file, out, fallback=fallback)
        for handler in getattr(stmt, "handlers", []):
            _collect_scope(
                handler.body,
                scope,
                file,
                out,
                fallback=fallback or _is_import_fallback(handler),
            )


def _is_import_fallback(handler) -> bool:
    """True for ``except ImportError`` / bare``except`` / ``except Exception``."""
    node = handler.type
    if node is None:  # bare except
        return True
    names = []
    if isinstance(node, ast.Name):
        names = [node.id]
    elif isinstance(node, ast.Tuple):
        names = [elt.id for elt in node.elts if isinstance(elt, ast.Name)]
    return any(name in _FALLBACK_EXCEPTIONS for name in names)


def _collect_setattr_call(call, line: int, scope: _Scope, file: str, out: list) -> None:
    """Record ``setattr(<upstream obj>, "name", ...)`` as a patch target."""
    func = call.func
    if not (isinstance(func, ast.Name) and func.id == "setattr") or len(call.args) < 2:
        return
    base = scope.resolve(call.args[0])
    name = call.args[1]
    if base is None or not _is_patch_root(base):
        return
    if isinstance(name, ast.Constant) and isinstance(name.value, str):
        out.append(Target(base + "." + name.value, file, line))


def _collect_assign(stmt, scope: _Scope, file: str, out: list) -> None:
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    value = stmt.value

    for target in targets:
        if not isinstance(target, ast.Attribute):
            continue
        base = scope.resolve(target.value)
        if base is None or not _is_patch_root(base):
            continue
        if value is not None and _is_constant(value):
            continue  # plugin's own marker attribute, not an upstream symbol
        out.append(Target(base + "." + target.attr, file, stmt.lineno))

    # setattr(<vllm obj>, "name", ...) assigned through
    if not isinstance(value, ast.Call):
        # read of an upstream attribute (``_orig = Cls.method``)
        resolved = scope.resolve(value) if value is not None else None
        if resolved is not None and "." in resolved[len(_root(resolved)) :]:
            out.append(Target(resolved, file, stmt.lineno))
        return

    _collect_setattr_call(value, stmt.lineno, scope, file, out)


def collect(package_dir: str) -> list:
    """Statically collect every upstream patch target below ``package_dir``.

    When the same symbol is touched in several places (``_orig = Cls.m`` read +
    ``Cls.m = wrapper`` assignment), one row per symbol is returned: the first
    site that carries a ``# patch-target: ...`` declaration, else the first.
    """
    root = Path(package_dir)
    display_base = Path(os.path.normpath(package_dir))
    sites: dict = {}
    for path in sorted(root.rglob("*.py")):
        display = str(display_base / path.relative_to(root))
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="replace"), filename=display
            )
        except (OSError, SyntaxError):
            continue
        found: list = []
        _collect_scope(list(tree.body), _Scope(), display, found)
        for target in found:
            current = sites.get(target.path)
            if current is None or (
                _declared_marker(current) is None
                and _declared_marker(target) is not None
            ):
                sites[target.path] = target
    return sorted(sites.values(), key=lambda t: (t.path, t.file, t.line))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _root_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _import_module(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # a broken optional dependency must not abort
        return None, exc


def resolve(path: str):
    """Resolve ``path`` in this interpreter.

    Returns ``(exists, signature, detail)`` where ``exists`` is one of
    :data:`EXISTS_YES`, :data:`EXISTS_NO`, :data:`EXISTS_SKIP`.
    """
    if _root(path) in OPTIONAL_ROOTS and not _root_installed(_root(path)):
        return EXISTS_SKIP, "-", f"optional root '{_root(path)}' not installed"

    parts = path.split(".")
    obj = None
    consumed = 0
    for split in range(len(parts), 0, -1):
        module, exc = _import_module(".".join(parts[:split]))
        if module is not None:
            obj = module
            consumed = split
            break
    if obj is None:
        return EXISTS_NO, "-", "no importable module prefix"

    for part in parts[consumed:]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return EXISTS_NO, "-", f"missing attribute '{part}'"

    return EXISTS_YES, _signature(obj), "-"


def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj)).replace("|", "/").replace("\n", " ")
    except (TypeError, ValueError):
        return "-"


def _declared_marker(target: Target) -> str | None:
    """The ``# patch-target: ...`` declaration at the patch site, if any.

    The comment may sit on the patch line or on the line directly above it.
    Returns :data:`EXISTS_INERT`, :data:`EXISTS_GUARDED` or ``None``.
    """
    try:
        lines = Path(target.file).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None
    # Walk up from the patch line through its (possibly multi-line) comment
    # block; stop at the first line of actual code.
    for offset in range(0, 5):
        idx = target.line - 1 - offset
        if not (0 <= idx < len(lines)):
            break
        line = lines[idx].strip()
        if INERT_MARKER in line:
            return EXISTS_INERT
        if GUARDED_MARKER in line:
            return EXISTS_GUARDED
        if offset and line and not line.startswith("#"):
            break
    return None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def check(package_dir: str) -> tuple:
    """Resolve every collected target; return ``(rows, failures)``."""
    rows = []
    failures = 0
    for target in collect(package_dir):
        exists, signature, detail = resolve(target.path)
        marker = _declared_marker(target)
        if exists == EXISTS_NO and marker is not None:
            detail = (
                "declared inert (removed upstream, plugin handles it)"
                if marker == EXISTS_INERT
                else "declared version-guarded (absent on this vLLM, import is guarded)"
            )
            exists = marker
        if exists == EXISTS_NO:
            failures += 1
        rows.append((target, exists, signature, detail))
    return rows, failures


def render(rows) -> str:
    lines = [
        "| target | file:line | exists | signature |",
        "| --- | --- | --- | --- |",
    ]
    for target, exists, signature, _detail in rows:
        lines.append(
            f"| {target.path} | {target.file}:{target.line} "
            f"| {exists} | {signature} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write(
            "usage: check_patch_targets.py <package-dir>  "
            "(checks every monkeypatched upstream symbol)\n"
        )
        return 2
    package_dir = args[0]
    if not Path(package_dir).is_dir():
        sys.stderr.write(f"check_patch_targets: not a directory: {package_dir}\n")
        return 2

    rows, failures = check(package_dir)
    print(render(rows))
    skipped = sum(1 for _t, exists, _s, _d in rows if exists == EXISTS_SKIP)
    inert = sum(1 for _t, exists, _s, _d in rows if exists == EXISTS_INERT)
    guarded = sum(1 for _t, exists, _s, _d in rows if exists == EXISTS_GUARDED)
    print(
        f"\nSUMMARY targets={len(rows)} missing={failures} "
        f"skipped={skipped} inert={inert} guarded={guarded}"
    )
    for target, exists, _signature, detail in rows:
        if exists == EXISTS_NO:
            print(f"  MISSING {target.path} ({target.file}:{target.line}) — {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
