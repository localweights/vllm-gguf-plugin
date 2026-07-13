"""Verify the committed MTP/NextN skip-prefix list against the GGUF inventories.

The skip-list (tests/fixtures/mtp_skip_prefixes.json) is chair-derived from the D0
inventory fixtures. These tests are the D6 oracle T-6's loader must satisfy:
applying the skip prefixes must remove exactly the trailing MTP block and leave no
NextN tensor behind.
"""

import json
import pathlib

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
SKIP = json.loads((FIX / "mtp_skip_prefixes.json").read_text())


def _inventory(stem: str) -> list[str]:
    data = json.loads((FIX / f"{stem}_inventory.json").read_text())
    return [t["name"] for t in data]


CASES = [("qwen35", "qwen36_27b"), ("qwen35moe", "qwen36_35b")]


def _prefixes(arch: str) -> list[str]:
    return SKIP[arch]["skip_block_prefixes"]


def test_every_prefix_matches_at_least_one_tensor():
    for arch, stem in CASES:
        names = _inventory(stem)
        for p in _prefixes(arch):
            assert any(n.startswith(p) for n in names), f"{arch}: prefix {p!r} matches nothing"


def test_skip_count_matches_expected():
    for arch, stem in CASES:
        names = _inventory(stem)
        prefixes = _prefixes(arch)
        skipped = [n for n in names if any(n.startswith(p) for p in prefixes)]
        assert len(skipped) == SKIP[arch]["expected_skip_count"], (
            f"{arch}: skipped {len(skipped)} != expected {SKIP[arch]['expected_skip_count']}"
        )


def test_no_nextn_survives_the_skip():
    """After removing skipped names, zero remaining tensors are NextN/MTP tensors."""
    for arch, stem in CASES:
        names = _inventory(stem)
        prefixes = _prefixes(arch)
        remaining = [n for n in names if not any(n.startswith(p) for p in prefixes)]
        assert not [n for n in remaining if ".nextn." in n], f"{arch}: NextN tensor survived skip"


def test_num_hidden_layers_is_block_count_minus_nextn():
    for arch, stem in CASES:
        meta = json.loads((FIX / f"{stem}_metadata.json").read_text())
        bc = meta[f"{arch}.block_count"]
        nx = meta[f"{arch}.nextn_predict_layers"]
        assert SKIP[arch]["num_hidden_layers"] == bc - nx
        # the sole skip prefix is exactly the trailing MTP block
        assert _prefixes(arch) == [f"blk.{bc - nx}."]
