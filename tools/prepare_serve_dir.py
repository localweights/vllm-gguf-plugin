#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a vLLM serve directory for a config-less Qwen3.5/3.6 GGUF.

Canonical artifacts (e.g. Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn.gguf) ship no
config.json or tokenizer. vLLM needs both next to the gguf (pass the dir via
--hf-config-path). This bundles the T-7 hand-steps:

  1. symlink the gguf into <out_dir>
  2. generate config.json from gguf metadata via the plugin's
     map_qwen35_config (arch + registered CausalLM class)
  3. copy tokenizer files from a reference HF checkout of the same model

Usage:
  prepare_serve_dir.py <model.gguf> <out_dir> --tokenizer-dir <hf_model_dir>
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
)

_ARCH_CLASS = {
    "qwen35": "Qwen3_5ForCausalLM",
    "qwen35moe": "Qwen3_5MoeForCausalLM",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", help="path to the .gguf file")
    ap.add_argument("out_dir", help="serve directory to create/populate")
    ap.add_argument(
        "--tokenizer-dir",
        required=True,
        help="HF model dir to copy tokenizer files from (same base model)",
    )
    args = ap.parse_args()

    from vllm_gguf_plugin.qwen35_config import (
        _read_gguf_metadata,
        map_qwen35_config,
    )

    gguf_path = os.path.abspath(args.gguf)
    os.makedirs(args.out_dir, exist_ok=True)

    meta = _read_gguf_metadata(gguf_path)
    arch = meta.get("general.architecture")
    if arch not in _ARCH_CLASS:
        print(f"unsupported architecture {arch!r} (want qwen35/qwen35moe)")
        return 1

    link = os.path.join(args.out_dir, os.path.basename(gguf_path))
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(gguf_path, link)

    config = map_qwen35_config(arch, meta)
    config.architectures = [_ARCH_CLASS[arch]]
    config.save_pretrained(args.out_dir)

    copied = []
    for f in TOKENIZER_FILES:
        src = os.path.join(args.tokenizer_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out_dir, f))
            copied.append(f)
    if "tokenizer.json" not in copied and "vocab.json" not in copied:
        print(f"no tokenizer files found in {args.tokenizer_dir}")
        return 1

    print(f"serve dir ready: {args.out_dir}")
    print(f"  gguf: {os.path.basename(gguf_path)} (symlink)")
    print(f"  config.json: arch {_ARCH_CLASS[arch]}")
    print(f"  tokenizer: {', '.join(copied)}")
    print(f"serve with: vllm serve {link} --hf-config-path {args.out_dir} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
