#!/usr/bin/env python3
"""Dump a GGUF file's tensor inventory and metadata to JSON fixtures.

Usage:
    python tools/dump_gguf_inventory.py <gguf_path> --name <stem> [--check]

Writes:
    tests/fixtures/<stem>_inventory.json  — sorted tensor list
    tests/fixtures/<stem>_metadata.json  — full metadata KV dict
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def numpy_to_native(obj):
    """Convert numpy types to plain Python scalars for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def serialise_value(val):
    """Recursively convert a metadata value to JSON-native types."""
    if isinstance(val, list):
        return [serialise_value(item) for item in val]
    return numpy_to_native(val)


def dump_inventory(reader: GGUFReader, stem: str, out_dir: Path):
    """Write tensor inventory and metadata JSON files."""
    # Tensor inventory: sorted list of {name, dtype, shape}
    tensors = sorted(
        [
            {
                "name": t.name,
                "dtype": t.tensor_type.name,
                "shape": [int(s) for s in t.shape],
            }
            for t in reader.tensors
        ],
        key=lambda t: t["name"],
    )

    inventory_path = out_dir / f"{stem}_inventory.json"
    with open(inventory_path, "w") as f:
        json.dump(tensors, f, indent=2, sort_keys=True)

    # Metadata: full KV dict with values converted to JSON-native types.
    # Large arrays (tokenizer vocab/merges/token-type — ~16MB of the raw dict)
    # are irrelevant to the config-parser/loader fixtures that consume this file,
    # so we omit their payload and keep only a length marker. Keeps the committed
    # fixture small while preserving that the key exists.
    ARRAY_OMIT_THRESHOLD = 256
    metadata = {}
    for key in reader.fields.keys():
        field = reader.get_field(key)
        try:
            val = field.contents() if field is not None else None
        except Exception:
            val = None
        sval = serialise_value(val)
        if isinstance(sval, list) and len(sval) > ARRAY_OMIT_THRESHOLD:
            sval = {"__omitted_array__": len(sval)}
        metadata[key] = sval

    metadata_path = out_dir / f"{stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return tensors, metadata


def check_invariants(tensors: list, metadata: dict, stem: str) -> bool:
    """Assert invariants and print summary. Return True on success."""
    tensor_count = len(tensors)
    arch = metadata.get("general.architecture", "UNKNOWN")

    ok = True
    if tensor_count <= 500:
        print(f"FAIL: {stem} has {tensor_count} tensors (expected > 500 — 27B~866, 35B-A3B~753)", file=sys.stderr)
        ok = False

    if arch not in {"qwen35", "qwen35moe"}:
        print(f"FAIL: {stem} arch={arch} (expected qwen35 or qwen35moe)", file=sys.stderr)
        ok = False

    print(f"INVENTORY {stem}: {tensor_count} tensors arch={arch}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Dump GGUF tensor inventory + metadata")
    parser.add_argument("gguf_path", help="Path to the GGUF file")
    parser.add_argument("--name", required=True, help="Stem name for output files")
    parser.add_argument("--check", action="store_true", help="Run invariant checks and exit non-zero on failure")
    args = parser.parse_args()

    gguf_path = args.gguf_path
    stem = args.name
    check = args.check

    # Resolve output dir relative to repo root (where the script is in tools/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    out_dir = repo_root / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {gguf_path} ...", flush=True)
    reader = GGUFReader(gguf_path)

    tensors, metadata = dump_inventory(reader, stem, out_dir)

    if check:
        ok = check_invariants(tensors, metadata, stem)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()