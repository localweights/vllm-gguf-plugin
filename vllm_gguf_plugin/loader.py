# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Generator, cast

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from .baked_weights import is_bake_valid, load_bake, save_bake
from .quantization import GGUFConfig
from .weight_utils import download_gguf, resolve_local_gguf
from .weights_adapter import get_weights_adapter

logger = init_logger(__name__)


def _register_gdn_out_proj_perm(model: nn.Module, adapter) -> None:
    """Attach the GDN out_proj input-column permutation to each out_proj layer.

    The GGUF stores out_proj with columns in llama.cpp's tiled V-head order; a
    column permutation of the quantized weight is impossible (quant blocks are
    wider than a head), so GGUFLinearMethod.apply() gathers the activation
    columns instead — it reads `gguf_input_col_perm` off the layer.
    """
    perm = getattr(adapter, "_gdn_out_proj_perm", None)
    if perm is None:
        return
    count = 0
    for name, module in model.named_modules():
        if name.endswith("linear_attn.out_proj"):
            device = next(
                (p.device for p in module.parameters(recurse=False)), None
            )
            module.gguf_input_col_perm = perm.to(
                device=device if device is not None else "cpu"
            )
            count += 1
    logger.info("Registered gguf_input_col_perm on %d out_proj layers", count)


def _mem_debug_iterator(weights):
    """Log cumulative CUDA memory per yielded tensor (VLLM_GGUF_MEM_DEBUG=1)."""
    count = 0
    last = 0.0
    for name, tensor in weights:
        count += 1
        alloc = torch.cuda.memory_allocated() / 2**30
        if count % 50 == 0 or alloc - last > 0.3 or tensor.numel() > 50_000_000:
            print(
                f"MEM-DEBUG #{count} {name} shape={tuple(tensor.shape)} "
                f"allocated={alloc:.2f} GiB (+{alloc - last:.2f})",
                flush=True,
            )
            last = alloc
        yield name, tensor


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model_weights or model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # local_dir:quant_type (e.g. /path/to/gguf-dir:Q8_0)
        if ":" in model_name_or_path:
            local_dir, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(local_dir):
                return resolve_local_gguf(local_dir, quant_type)
            # remote repo_id:quant_type
            return download_gguf(
                local_dir,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <local_dir>:<quant_type>, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _prepare_adapter(self, model_config: ModelConfig):
        local_model_path = self._prepare_weights(model_config)
        adapter = get_weights_adapter(model_config.hf_config)
        adapter.prepare_loading(local_model_path, model_config)
        return adapter

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter = self._prepare_adapter(model_config)
        gguf_path = adapter.load_spec.weights_source[0]

        if is_bake_valid(gguf_path):
            baked = load_bake(gguf_path)
            n_tensors = len(baked)
            total_gib = sum(
                t.numel() * t.element_size() for t in baked.values()
            ) / 2**30
            logger.info(
                "baked weight cache HIT (%s, %d tensors, %.2f GiB)",
                gguf_path, n_tensors, total_gib,
            )
            model.load_weights(baked.items())
        else:
            captured: dict[str, torch.Tensor] = {}

            def _capture(
                it,
            ) -> Generator[tuple[str, torch.Tensor], None, None]:
                for name, t in it:
                    captured[name] = t  # already on CPU from GGUF loader
                    yield name, t

            weights = _capture(adapter.prepare_weights(model_config))
            if os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1":
                weights = _mem_debug_iterator(weights)
            model.load_weights(weights)
            try:
                save_bake(gguf_path, captured)
                total_gib = sum(
                    t.numel() * t.element_size() for t in captured.values()
                ) / 2**30
                logger.info(
                    "baked weight cache SAVED (%s, %d tensors, %.2f GiB)",
                    gguf_path, len(captured), total_gib,
                )
            except Exception:
                logger.warning(
                    "Failed to save baked weight cache", exc_info=True
                )

        _register_gdn_out_proj_perm(model, adapter)
        if os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1":
            print(
                f"MEM-DEBUG post-load_weights allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB",
                flush=True,
            )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter = self._prepare_adapter(model_config)
        vllm_config.model_config.hf_config = model_config.hf_config
        logger.debug(
            "GGUF unquantized modules: %s", adapter.load_spec.unquantized_modules
        )
        vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(
            adapter.load_spec.unquantized_modules
        )

        target_device = torch.device(device_config.device)
        gguf_path = adapter.load_spec.weights_source[0]

        if is_bake_valid(gguf_path):
            baked = load_bake(gguf_path)
            n_tensors = len(baked)
            total_gib = sum(
                t.numel() * t.element_size() for t in baked.values()
            ) / 2**30
            logger.info(
                "baked weight cache HIT (%s, %d tensors, %.2f GiB)",
                gguf_path, n_tensors, total_gib,
            )
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, prefix=prefix)
                model.load_weights(baked.items())
                _register_gdn_out_proj_perm(model, adapter)
                process_weights_after_loading(model, model_config, target_device)
            return model

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)

            captured: dict[str, torch.Tensor] = {}

            def _capture(
                it,
            ) -> Generator[tuple[str, torch.Tensor], None, None]:
                for name, t in it:
                    captured[name] = t  # already on CPU from GGUF loader
                    yield name, t

            weights = _capture(adapter.prepare_weights(model_config))
            if os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1":
                weights = _mem_debug_iterator(weights)
            model.load_weights(weights)
            _register_gdn_out_proj_perm(model, adapter)
            if os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1":
                logger.info(
                    "MEM-DEBUG pre-process_weights allocated=%.2f GiB",
                    torch.cuda.memory_allocated() / 2**30,
                )
            process_weights_after_loading(model, model_config, target_device)
            try:
                save_bake(gguf_path, captured)
                total_gib = sum(
                    t.numel() * t.element_size() for t in captured.values()
                ) / 2**30
                logger.info(
                    "baked weight cache SAVED (%s, %d tensors, %.2f GiB)",
                    gguf_path, len(captured), total_gib,
                )
            except Exception:
                logger.warning(
                    "Failed to save baked weight cache", exc_info=True
                )

            if os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1":
                logger.info(
                    "MEM-DEBUG post-process_weights allocated=%.2f GiB",
                    torch.cuda.memory_allocated() / 2**30,
                )
        return model
