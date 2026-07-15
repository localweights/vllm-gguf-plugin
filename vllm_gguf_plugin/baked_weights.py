# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Baked weight cache — write restructured GGUF tensors to safetensors for
sub-100ms cold-start reload.

Usage (from loader.py)::

    from .baked_weights import bake_path, is_bake_valid, load_bake, save_bake

    bw_path = bake_path(gguf_path)
    if is_bake_valid(gguf_path):
        tensors = load_bake(gguf_path)
        # feed tensors to model.load_weights(...)
    else:
        # run normal GGUF parse path, capture tensors, then:
        save_bake(gguf_path, captured_tensors)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import safetensors.torch

# ---------------------------------------------------------------------------
# Package version — read from the installed wheel metadata once at module
# load time (changes only when the wheel is re-installed).
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version

    _PLUGIN_VERSION: str = _pkg_version("vllm-gguf-plugin")
except Exception:  # noqa: BLE001  # pragma: no cover
    _PLUGIN_VERSION = "0.0.0"

_BAKE_FORMAT = 1

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def bake_path(gguf_path: str | Path) -> Path:
    """Return the safetensors path for a baked copy of *gguf_path*."""
    return Path(str(gguf_path) + ".baked.safetensors")


def _bake_meta_path(gguf_path: str | Path) -> Path:
    return Path(str(gguf_path) + ".baked.meta.json")


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------


def bake_meta(gguf_path: str | Path) -> dict:
    """Build the meta dict that validates a bake for *gguf_path*."""
    st = Path(gguf_path).stat()
    return {
        "gguf_size": st.st_size,
        "gguf_mtime_ns": st.st_mtime_ns,
        "plugin_version": _PLUGIN_VERSION,
        "format": _BAKE_FORMAT,
    }


def _kill_switched() -> bool:
    """``VLLM_GGUF_BAKED_CACHE=0`` disables both save and load."""
    return os.environ.get("VLLM_GGUF_BAKED_CACHE", "1") == "0"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_bake_valid(gguf_path: str | Path) -> bool:
    """Return True when a valid bake exists for *gguf_path*.

    Both the safetensors file and the meta-json must exist, and the meta
    must exactly match the current :func:`bake_meta` output.
    """
    if _kill_switched():
        return False

    st_path = bake_path(gguf_path)
    meta_path = _bake_meta_path(gguf_path)

    if not st_path.is_file() or not meta_path.is_file():
        return False

    try:
        with open(meta_path) as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    try:
        current = bake_meta(gguf_path)
    except OSError:
        return False

    return existing == current


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_bake(
    gguf_path: str | Path,
    named_tensors: dict[str, "torch.Tensor"],
) -> None:
    """Write *named_tensors* to safetensors, then atomically write meta json.

    Tensors are moved to CPU and made contiguous before saving.  The meta
    json is written **last** (via tmp + rename) so that a crash during the
    safetensors write never leaves a valid-looking bake.

    No-op when the ``VLLM_GGUF_BAKED_CACHE=0`` kill-switch is set.
    """
    if _kill_switched():
        return

    tensors = {
        name: t.detach().cpu().contiguous() for name, t in named_tensors.items()
    }

    st_path = bake_path(gguf_path)
    safetensors.torch.save_file(tensors, str(st_path))

    meta = bake_meta(gguf_path)
    meta_path = _bake_meta_path(gguf_path)
    fd, tmp_name = tempfile.mkstemp(
        dir=meta_path.parent,
        prefix=meta_path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f)
        os.rename(tmp_name, str(meta_path))
    except BaseException:
        # Clean up the temp file if rename fails
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_bake(gguf_path: str | Path) -> dict[str, "torch.Tensor"]:
    """Load tensors from the baked safetensors file (CPU)."""
    st_path = bake_path(gguf_path)
    result = safetensors.torch.load_file(str(st_path), device="cpu")
    return result
