# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import os
import re as _re
from collections.abc import Generator
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from vllm.logger import init_logger

logger = init_logger(__name__)


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    prefix_list = ["*.", "*-"]
    suffix_list = ["-*", ""]
    allow_patterns = [
        f"{prefix}{qt}{suffix}.gguf"
        for qt in (quant_type.upper(), quant_type.lower())
        for prefix, suffix in itertools.product(prefix_list, suffix_list)
    ]

    folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    local_files: list[str] = []
    for pattern in allow_patterns:
        local_files.extend(glob.glob(os.path.join(folder, pattern)))

    if not local_files:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]


def resolve_local_gguf(local_dir: str, quant_type: str) -> str:
    """Find a GGUF file matching *quant_type* in a local directory."""
    import glob as glob_mod

    patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob_mod.glob(os.path.join(local_dir, pat)))
    if not matches:
        raise ValueError(
            f"No GGUF file matching quant_type '{quant_type}' found in {local_dir}"
        )
    matches.sort(key=lambda x: (x.count("-"), x))
    return matches[0]


def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {tensor.name for tensor in reader.tensors}
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def get_gguf_weight_type_map(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]:
    reader = gguf.GGUFReader(gguf_file)
    return {
        gguf_to_hf_name_map[tensor.name]: tensor.tensor_type.name
        for tensor in reader.tensors
        if tensor.name in gguf_to_hf_name_map
    }


def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    yield from gguf_quant_weights_iterator_multi([gguf_file], gguf_to_hf_name_map)


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str], gguf_to_hf_name_map: dict[str, str] | None = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` for all tensors in *gguf_files*.

    When *gguf_to_hf_name_map* is ``None``, raw GGUF tensor names are used
    directly (useful when a caller will apply a :class:`WeightsMapper`
    afterwards).  When a mapping is provided, tensors not present in the map
    are skipped and names are translated accordingly.
    """
    _QUANT_TYPES = ("F32", "BF16", "F16")

    for gguf_file in gguf_files:
        reader = gguf.GGUFReader(gguf_file)
        for tensor in reader.tensors:
            if gguf_to_hf_name_map is not None:
                if tensor.name not in gguf_to_hf_name_map:
                    continue
                name = gguf_to_hf_name_map[tensor.name]
            else:
                name = tensor.name

            weight_type = tensor.tensor_type
            if weight_type.name not in _QUANT_TYPES:
                yield name.replace("weight", "qweight_type"), torch.tensor(weight_type)
                name = name.replace("weight", "qweight")

            weight = tensor.data
            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)
            yield name, param


def get_gguf_unquantized_params(gguf_files: list[str]) -> list[str]:
    _QUANT_TYPES = ("F32", "BF16", "F16")
    return list(
        {
            tensor.name
            for gguf_file in gguf_files
            for tensor in gguf.GGUFReader(gguf_file).tensors
            if tensor.tensor_type.name in _QUANT_TYPES
        }
    )
    # for gguf_file in gguf_files:
    #     reader = gguf.GGUFReader(gguf_file)
    #     for tensor in reader.tensors:
    #         if tensor.tensor_type.name in unquant_types:
    #             yield tensor.name.rsplit(".", 1)[0]


# D4 — fail-closed MTP skip for qwen35/qwen35moe
# Owner: prepare_loading() calls this after build_name_map;
#        success → skipped count logged, failure → GGUFUnmappedTensorError raised.

_BLK_IDX_RE = _re.compile(r"^blk\.(\d+)\.")


def partition_unmapped_gguf_tensors(
    gguf_names: set[str],
    mapped_keys: set[str],
    num_hidden_layers: int,
    arch: str,
) -> int:
    """Partition unmapped GGUF tensor names into MTP (skipped) vs error.

    For qwen35/qwen35moe arches, tensors in block indices >= num_hidden_layers
    belong to the MTP (NextN prediction) layer and are silently skipped with a
    log line.  Any other unmapped tensor raises GGUFUnmappedTensorError.

    For non-qwen35 arches, any unmapped tensor is an error.

    Returns the number of MTP tensors skipped.

    Raises GGUFUnmappedTensorError if non-MTP unmapped tensors exist.
    """
    from .errors import GGUFUnmappedTensorError

    unmapped = gguf_names - mapped_keys
    if not unmapped:
        return 0

    if arch not in ("qwen35", "qwen35moe"):
        raise GGUFUnmappedTensorError(unmapped)

    mtp_tensors: list[str] = []
    error_tensors: list[str] = []
    for name in unmapped:
        m = _BLK_IDX_RE.match(name)
        if m:
            idx = int(m.group(1))
            if idx >= num_hidden_layers:
                mtp_tensors.append(name)
                continue
        error_tensors.append(name)

    if error_tensors:
        raise GGUFUnmappedTensorError(error_tensors)

    if mtp_tensors:
        logger.info(
            "skipped %d MTP tensors (serving path does not use MTP)",
            len(mtp_tensors),
        )
    return len(mtp_tensors)