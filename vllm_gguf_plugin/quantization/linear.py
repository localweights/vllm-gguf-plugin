# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    # CUDA kernels below index raw data_ptrs and assume a contiguous x
    # (see fused_moe.py — non-contiguous compiled-graph views caused OOB).
    x = x.contiguous()
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        if shape[1] != x.shape[1]:
            print(
                f"MEM-DEBUG dequant-mismatch wtype={qweight_type} "
                f"qweight={tuple(qweight.shape)} dequant_shape={shape} "
                f"x={tuple(x.shape)}",
                flush=True,
            )
        # Chunk the dequant fallback along output rows: a full dequant of a
        # large layer (e.g. 34816x5120 bf16 = ~340 MB) as one transient OOMs
        # small cards during CUDA-graph capture. 8192-row chunks cap the
        # transient at ~80 MB with negligible overhead.
        _CHUNK = 4096
        if shape[0] > _CHUNK:
            outs = []
            for i in range(0, shape[0], _CHUNK):
                rows = min(_CHUNK, shape[0] - i)
                wchunk = ops.ggml_dequantize(
                    qweight.narrow(0, i, rows), qweight_type, rows, shape[1], x.dtype
                )
                outs.append(x @ wchunk.T)
            y = torch.cat(outs, dim=-1)
        else:
            weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
            y = x @ weight.T
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        import os

        _dbg = os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1"
        if _dbg:
            _before = torch.cuda.memory_allocated() / 2**30
        self._materialize_gguf_parameters(layer)
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)
        if _dbg:
            _after = torch.cuda.memory_allocated() / 2**30
            if tuple(layer.qweight.shape) == (16384, 3520) and not getattr(
                type(self), "_dbg_qkvz_printed", False
            ):
                type(self)._dbg_qkvz_printed = True
                print(
                    f"MEM-DEBUG qkvz-layer shard_id={layer.qweight.shard_id} "
                    f"types={layer.qweight_type.shard_weight_type} "
                    f"wtype={layer.qweight_type.weight_type} "
                    f"map={getattr(layer.qweight, 'shard_offset_map', None)}",
                    flush=True,
                )
            if _after - _before > 0.05:
                print(
                    f"MEM-DEBUG process_weights {type(layer).__name__} "
                    f"qshape={tuple(layer.qweight.shape)} {_before:.2f}->{_after:.2f} GiB "
                    f"(+{_after - _before:.2f})",
                    flush=True,
                )

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_qweight(layer)
        self._materialize_qweight_type(layer)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "qweight")

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "qweight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                # Release this shard's GPU memory immediately after copying.
                # Owner of the slot: _create_padded_weight_param (this method).
                # Removed on success: set to None here after copy.
                # Removed on failure: exception propagates, padded_data discarded.
                shard_nbytes = data_container[id_in_container].nbytes
                shard_refs = None
                import os as _os
                if _os.environ.get("VLLM_GGUF_MEM_DEBUG") == "1":
                    import sys as _sys
                    shard_refs = _sys.getrefcount(data_container[id_in_container])
                data_container[id_in_container] = None
                if shard_refs is not None:
                    print(
                        f"MEM-DEBUG shard-free idx={idx} bytes={shard_nbytes} "
                        f"refs-before-free={shard_refs} "
                        f"alloc={torch.cuda.memory_allocated() / 2**30:.3f} GiB",
                        flush=True,
                    )
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=qweight.weight_loader,
                input_dim=qweight.input_dim,
                output_dim=qweight.output_dim,
                tensor_shape=qweight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(qweight.shard_id_map)
            if hasattr(qweight, "ignore_warning"):
                padded_param.ignore_warning = qweight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            qweight.data_container.clear()
            qweight.shard_id.clear()
            qweight.shard_id_map.clear()
            if qweight.data.numel() > 0:
                qweight.data = torch.empty(
                    0, dtype=qweight.dtype, device=qweight.device
                )
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        # GDN out_proj column permutation: GGUF weight columns are in tiled
        # (ggml broadcast) order. Permute activations from grouped → tiled
        # order before the matmul. This is an activation-side gather (cheap)
        # rather than a column perm on the quantized weight (impossible).
        # The perm tensor is registered as gguf_input_col_perm on the layer
        # by the loader post-load hook.
        perm = getattr(layer, "gguf_input_col_perm", None)
        if perm is not None:
            x = x.index_select(-1, perm)

        shard_id = layer.qweight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            fallback_wtype = layer.qweight_type.weight_type
            shard_weight_types = [
                layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            offset_map = getattr(layer.qweight, "shard_offset_map", None)
            unpadded = offset_map is None or all(
                offset_map[idx][2] == qweight.shape[1] for idx in shard_id
            )
            # Whole-tensor fast path is only valid when every shard shares one
            # quant type AND none was width-padded — dequantizing a padded row
            # at a single type yields the wrong logical width (e.g. 6400 vs
            # 5120 on mixed-quant merged in_proj).
            if len(set(shard_weight_types)) == 1 and unpadded:
                out = fused_mul_mat_gguf_op(x, qweight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                try:
                    result.append(
                        fused_mul_mat_gguf_op(
                            x, qweight[start:end, :offset].contiguous(), qweight_type
                        )
                    )
                except RuntimeError:
                    print(
                        f"MEM-DEBUG apply-fail shard={idx} start={start} end={end} "
                        f"offset={offset} wtype={qweight_type} "
                        f"types={layer.qweight_type.shard_weight_type} "
                        f"fallback={fallback_wtype} qshape={tuple(qweight.shape)} "
                        f"x={tuple(x.shape)} map={layer.qweight.shard_offset_map}",
                        flush=True,
                    )
                    raise
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf_op(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out