#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TopKV2 Kernel/GEIR reference and PyTorch benchmark.

PyTorch ``torch.topk`` is the native GPU competitor where its dtype support
permits.  The CPU golden reproduces CUDA's threshold gather and
SmallBitonicSort for ``sorted=True`` and ``k<=32``.  For larger ``k``, equal
value tie indices are checked semantically; for ``sorted=False``, output order
is handled the same way.  ACLNN and E2E (torch.topk) pathways are covered by the Aclnn/Torch spec
    classes below; the repository does not deliver an ONNX pathway.
"""

import torch

__spec__ = {
    "top_k_v2": "TopKV2KernelSpec",
    "aclnnTopkV2": "TopKV2AclnnSpec",
    "torch.topk": "TopKV2TorchSpec",
}
__golden__ = {
    "kernel": {"top_k_v2": "top_k_v2_golden"},
    "aclnn": {"aclnnTopkV2": "aclnn_top_k_v2_golden"},
}
__input__ = {"kernel": {"top_k_v2": "top_k_v2_input"}}

_KERNEL_TOLERANCE = {
    dtype: {
        "standard": "stat_rel_err"
        if dtype in ("float16", "float32", "bfloat16")
        else "binary_equal"
    }
    for dtype in (
        "float16",
        "float32",
        "bfloat16",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
}

_NATIVE_UNSUPPORTED_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    )
    if dt is not None
)


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if "bfloat16" in str(x.dtype):
        return torch.from_numpy(x.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(x)


def _to_array(x):
    x = x.detach().cpu().contiguous()
    if x.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return x.view(torch.int16).numpy().view(bfloat16)
    return x.numpy()


def _k_value(k):
    """Accept a python scalar, numpy scalar/array or 0-d device tensor."""
    if isinstance(k, torch.Tensor):
        return int(k.reshape(-1)[0].item())
    if hasattr(k, "item"):
        return int(k.item())
    return int(k)


def _index_dtype(indices_dtype=3, output_dtypes=()):
    if output_dtypes and len(output_dtypes) > 1:
        dtype = output_dtypes[1]
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        return torch.int64 if "int64" in str(dtype) else torch.int32
    return torch.int64 if int(indices_dtype) == 9 else torch.int32


def _normalize_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is invalid for rank {ndim}")
    return axis


def _radix_keys(values):
    """Match CUDA TopKTypeConfig ordering for the supported dtypes."""
    if values.is_floating_point():
        width = values.element_size() * 8
        view_dtype = torch.int16 if width == 16 else torch.int32
        all_bits = (1 << width) - 1
        sign_bit = 1 << (width - 1)
        bits = values.contiguous().view(view_dtype).to(torch.int64) & all_bits
        keys = torch.where((bits & sign_bit) != 0, bits ^ all_bits, bits ^ sign_bit)
        return torch.where(torch.isnan(values), all_bits, keys)
    if values.dtype == torch.uint64:
        return values.view(torch.int64) ^ torch.iinfo(torch.int64).min
    return values.to(torch.int64)


def _value_better(lhs, rhs, largest):
    if lhs.is_floating_point():
        lhs_nan = bool(torch.isnan(lhs).item())
        rhs_nan = bool(torch.isnan(rhs).item())
        if largest:
            return (lhs_nan and not rhs_nan) or bool((lhs > rhs).item())
        return (rhs_nan and not lhs_nan) or bool((lhs < rhs).item())
    lhs_key = _radix_keys(lhs.reshape(1))[0]
    rhs_key = _radix_keys(rhs.reshape(1))[0]
    comparison = lhs_key > rhs_key if largest else lhs_key < rhs_key
    return bool(comparison.item())


def _gather_values(x, axis, indices):
    view_dtype = {
        torch.uint16: torch.int16,
        torch.uint32: torch.int32,
        torch.uint64: torch.int64,
    }.get(x.dtype)
    if view_dtype is None:
        return torch.gather(x, axis, indices)
    return torch.gather(x.view(view_dtype), axis, indices).view(x.dtype)


def _small_bitonic_sort(values, indices, largest):
    """Reproduce CUDA bitonicSort<32>, including invalid and equal swaps."""
    slots = 32
    count = values.numel()
    keys = torch.zeros(slots, dtype=values.dtype)
    result_indices = torch.zeros(slots, dtype=indices.dtype)
    valid = torch.zeros(slots, dtype=torch.bool)
    keys[:count] = values
    result_indices[:count] = indices
    valid[:count] = True

    size = 2
    while size < slots:
        stride = size // 2
        while stride > 0:
            for thread in range(slots // 2):
                pos = 2 * thread - (thread & (stride - 1))
                other = pos + stride
                direction = bool(thread & (size // 2))
                swap = (
                    _value_better(keys[pos], keys[other], largest) and bool(valid[pos])
                ) or not bool(valid[other])
                if swap == direction:
                    lhs_value, rhs_value = keys[pos].clone(), keys[other].clone()
                    lhs_index = result_indices[pos].clone()
                    rhs_index = result_indices[other].clone()
                    lhs_valid, rhs_valid = valid[pos].clone(), valid[other].clone()
                    keys[pos], keys[other] = rhs_value, lhs_value
                    result_indices[pos], result_indices[other] = rhs_index, lhs_index
                    valid[pos], valid[other] = rhs_valid, lhs_valid
            stride //= 2
        size *= 2

    stride = slots // 2
    while stride > 0:
        for thread in range(slots // 2):
            pos = 2 * thread - (thread & (stride - 1))
            other = pos + stride
            swap = (
                _value_better(keys[pos], keys[other], largest) and bool(valid[pos])
            ) or not bool(valid[other])
            if not swap:
                lhs_value, rhs_value = keys[pos].clone(), keys[other].clone()
                lhs_index = result_indices[pos].clone()
                rhs_index = result_indices[other].clone()
                lhs_valid, rhs_valid = valid[pos].clone(), valid[other].clone()
                keys[pos], keys[other] = rhs_value, lhs_value
                result_indices[pos], result_indices[other] = rhs_index, lhs_index
                valid[pos], valid[other] = rhs_valid, lhs_valid
        stride //= 2
    return keys[:count], result_indices[:count]


def _cuda_topk_row(row, k, largest, index_dtype):
    keys = _radix_keys(row)
    threshold_pos = len(keys) - k if largest else k - 1
    threshold = torch.kthvalue(keys, threshold_pos + 1).values
    strict = keys > threshold if largest else keys < threshold
    gathered = torch.cat(
        (torch.nonzero(strict).flatten(), torch.nonzero(keys == threshold).flatten())
    )[:k]
    values = _gather_values(row, 0, gathered).clone()
    indices = gathered.to(index_dtype)
    return _small_bitonic_sort(values, indices, largest)


def _cuda_small_topk(x, k, dim, largest, index_dtype):
    axis = _normalize_axis(dim, x.ndim)
    moved = torch.movedim(x, axis, -1).contiguous()
    rows = moved.reshape(-1, moved.shape[-1])
    if rows.shape[0] == 0:
        shape = list(x.shape)
        shape[axis] = k
        return torch.empty(shape, dtype=x.dtype), torch.empty(shape, dtype=index_dtype)
    outputs = [_cuda_topk_row(row, k, bool(largest), index_dtype) for row in rows]
    values = torch.stack([output[0] for output in outputs])
    indices = torch.stack([output[1] for output in outputs])
    moved_shape = (*moved.shape[:-1], k)
    return (
        torch.movedim(values.reshape(moved_shape), -1, axis).contiguous(),
        torch.movedim(indices.reshape(moved_shape), -1, axis).contiguous(),
    )


def _topk(x, k, dim, largest, sorted_output):
    axis = int(dim)
    if k < 0 or k > x.shape[axis]:
        raise ValueError(f"k={k} is outside [0, {x.shape[axis]}]")
    if x.dtype not in _NATIVE_UNSUPPORTED_DTYPES:
        return torch.topk(
            x, k, dim=axis, largest=bool(largest), sorted=bool(sorted_output)
        )
    values, indices = torch.sort(x, dim=axis, descending=bool(largest), stable=True)
    return values.narrow(axis, 0, k), indices.narrow(axis, 0, k)


def top_k_v2_golden(
    x,
    k,
    *,
    sorted=True,
    dim=-1,
    largest=True,
    indices_dtype=3,
    output_dtypes=(),
    **kwargs,
):
    tensor = _as_tensor(x)
    k_value = _k_value(k)
    index_dtype = _index_dtype(indices_dtype, output_dtypes)
    if bool(sorted) and 0 < k_value <= 32:
        values, indices = _cuda_small_topk(tensor, k_value, dim, largest, index_dtype)
    else:
        values, indices = _topk(tensor, k_value, dim, largest, sorted)
        indices = indices.to(index_dtype)
    return [_to_array(values), _to_array(indices)]


def top_k_v2_input(x, k, *, dim=-1, testcase_name="", **kwargs):
    """Inject deterministic tie and floating-point special-value inputs."""
    tensor = _as_tensor(x)
    axis = _normalize_axis(dim, tensor.ndim)
    name = str(testcase_name)
    patterns = {
        "duplicate": [7.0, 1.0, 7.0, 3.0, 7.0, 2.0, 8.0, 8.0],
        "all_equal": [3.0],
        "signed_zero": [-0.0, 0.0, 0.0, -0.0],
        "nan": [float("nan"), 3.0, float("nan"), 1.0, 2.0, 2.0, -1.0, 0.0],
        "infinity": [float("inf"), 3.0, -float("inf"), 0.0, 2.0, -2.0],
    }
    axis_values = None
    if name.endswith("_unique"):
        axis_values = torch.arange(tensor.shape[axis], dtype=torch.int64).to(
            tensor.dtype
        )
    else:
        for marker, values in patterns.items():
            if name.endswith(f"_{marker}"):
                pattern = torch.tensor(values, dtype=tensor.dtype)
                repeats = (tensor.shape[axis] + pattern.numel() - 1) // pattern.numel()
                axis_values = pattern.repeat(repeats)[: tensor.shape[axis]]
                break
    if axis_values is not None:
        shape = [1] * tensor.ndim
        shape[axis] = tensor.shape[axis]
        tensor.copy_(axis_values.reshape(shape).expand(tensor.shape))
    return x, k


def _equal_values(lhs, rhs):
    equal = lhs == rhs
    if lhs.is_floating_point():
        equal |= torch.isnan(lhs) & torch.isnan(rhs)
    return equal


def _exact_indices(compare_context):
    """Only the Kernel/GEIR golden reproduces CUDA bitonic index order;
    ACLNN/E2E goldens use torch.topk semantics, so ties are semantic."""
    api_name = str(getattr(compare_context, "api_name", "") or "")
    return not (api_name.startswith("aclnn") or api_name.startswith("torch."))


def top_k_v2_compare(
    npu_values,
    npu_indices,
    golden_values,
    golden_indices,
    *,
    compare_context,
):
    attrs = compare_context.attributes
    x = _as_tensor(compare_context.input_tensors[0])
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    input_tensors = list(compare_context.input_tensors)
    k_raw = input_tensors[1] if len(input_tensors) > 1 else attrs.get("k")
    k_value = _k_value(k_raw)
    values = _as_tensor(npu_values)
    expected = _as_tensor(golden_values)
    axis = _normalize_axis(attrs.get("dim", -1), x.ndim)
    sorted_output = bool(attrs.get("sorted", True))

    if sorted_output:
        value_equal = _equal_values(values, expected)
    else:
        value_equal = _equal_values(
            torch.sort(values, dim=axis).values,
            torch.sort(expected, dim=axis).values,
        )
    value_bad = int(torch.count_nonzero(~value_equal).item())
    value_total = values.numel()

    indices = _as_tensor(npu_indices).to(torch.int64)
    if indices.numel() == 0:
        invalid = gather_bad = duplicate_count = 0
    else:
        valid = (indices >= 0) & (indices < x.shape[axis])
        invalid = indices.numel() - int(torch.count_nonzero(valid).item())
        safe_indices = torch.clamp(indices, 0, x.shape[axis] - 1)
        gathered = _gather_values(x, axis, safe_indices)
        gather_bad = int(torch.count_nonzero(~_equal_values(gathered, values)).item())
        ordered = torch.sort(indices, dim=axis).values
        duplicate_count = (
            int(
                torch.count_nonzero(
                    ordered.narrow(axis, 1, ordered.shape[axis] - 1)
                    == ordered.narrow(axis, 0, ordered.shape[axis] - 1)
                ).item()
            )
            if ordered.shape[axis] > 1
            else 0
        )

    if sorted_output and 0 < k_value <= 32 and _exact_indices(compare_context):
        expected_indices = _as_tensor(golden_indices).to(torch.int64)
        golden_index_bad = int(torch.count_nonzero(indices != expected_indices).item())
        index_bad = golden_index_bad
    else:
        golden_index_bad = 0
        index_bad = invalid + gather_bad + duplicate_count
    index_total = indices.numel()
    value_precision = (
        100.0 if value_total == 0 else 100.0 * (value_total - value_bad) / value_total
    )
    index_precision = (
        100.0
        if index_total == 0
        else 100.0 * max(0, index_total - index_bad) / index_total
    )
    return [
        {
            "pass": value_bad == 0,
            "precision": value_precision,
            "error_info": None
            if value_bad == 0
            else f"{value_bad}/{value_total} value mismatches",
        },
        {
            "pass": index_bad == 0,
            "precision": index_precision,
            "error_info": None
            if index_bad == 0
            else (
                f"golden_index_mismatches={golden_index_bad}, "
                f"invalid_indices={invalid}, duplicate_indices={duplicate_count}, "
                f"gathered_value_mismatches={gather_bad}"
            ),
        },
    ]


class TopKV2ThirdParty:
    """Timed competitor: ``__call__`` contains only required TopK work."""

    def __init__(
        self, k, *, sorted=True, dim=-1, largest=True, indices_dtype=3, **kwargs
    ):
        self.k = _k_value(k)
        self.sorted = bool(sorted)
        self.dim = int(dim)
        self.largest = bool(largest)

    def __call__(self, x, **kwargs):
        return _topk(x, self.k, self.dim, self.largest, self.sorted)


class TopKV2KernelSpec:
    golden = staticmethod(top_k_v2_golden)
    customize_inputs = staticmethod(top_k_v2_input)
    compare = staticmethod(top_k_v2_compare)
    third_party = {"torch": TopKV2ThirdParty}
    tolerance = dict(_KERNEL_TOLERANCE)


def _aclnn_index_dtype(outputs, kwargs):
    """Resolve indices dtype: declared integer output tensor first, then attr (9 = int64)."""
    for tensor in reversed(outputs):
        if tensor is not None and hasattr(tensor, "dtype"):
            text = str(tensor.dtype)
            if "float" not in text and "bool" not in text and "complex" not in text:
                return torch.int64 if "int64" in text else torch.int32
    for name in ("indicesType", "indices_dtype"):
        value = kwargs.get(name)
        if value is not None and not hasattr(value, "dtype"):
            text = str(value)
            return torch.int64 if ("int64" in text or text == "9") else torch.int32
    return None


def _aclnn_top_k_v2(x, k, dim, largest, sorted_output, index_dtype):
    """ACLNN/E2E reference: plain torch.topk semantics; keeps torch.topk's
    native index dtype when the case does not declare one."""
    axis = -1 if dim is None else int(dim)
    tensor = _as_tensor(x)
    largest = True if largest is None else bool(largest)
    sorted_output = True if sorted_output is None else bool(sorted_output)
    values, indices = _topk(tensor, _k_value(k), axis, largest, sorted_output)
    if index_dtype is not None and indices.dtype != index_dtype:
        indices = indices.to(index_dtype)
    return values, indices


def aclnn_top_k_v2_golden(
    self, k, dim=-1, largest=1, sorted=1, valuesOut=None, indicesOut=None, **kwargs
):
    """ACLNN asset-registry form: positional layout follows the aclnnTopKV2
    C header (x, k, dim, largest, sorted, valuesOut, indicesOut)."""
    index_dtype = _aclnn_index_dtype((valuesOut, indicesOut), kwargs)
    values, indices = _aclnn_top_k_v2(self, k, dim, largest, sorted, index_dtype)
    return [values, indices]


def _aclnn_top_k_v2_golden(
    x, k, dim=-1, largest=True, sorted=True, *_outputs, **kwargs
):
    """TestSpec form: output tensors arrive positionally after the scalars."""
    index_dtype = _aclnn_index_dtype(_outputs, kwargs)
    values, indices = _aclnn_top_k_v2(x, k, dim, largest, sorted, index_dtype)
    return [values, indices]


class TopKV2AclnnThirdParty:
    """Timed torch competitor for the ACLNN/E2E pathways."""

    def __init__(
        self, k, *, dim=-1, largest=True, sorted=True, indices_dtype=3, **kwargs
    ):
        self.k = _k_value(k)
        self.dim = int(dim)
        self.largest = bool(largest)
        self.sorted = bool(sorted)

    def __call__(self, input=None, *args, x=None, **kwargs):
        """E2E pools name the tensor ``input`` (torch.topk signature);
        ACLNN pools deliver it as the first positional leftover (header param
        ``self``) or under the keyword ``x``."""
        tensor = input if input is not None else x
        return _topk(tensor, self.k, self.dim, self.largest, self.sorted)


class TopKV2AclnnSpec:
    golden = staticmethod(_aclnn_top_k_v2_golden)
    compare = staticmethod(top_k_v2_compare)
    third_party = {"torch": TopKV2AclnnThirdParty}
    tolerance = dict(_KERNEL_TOLERANCE)


class TopKV2TorchSpec(TopKV2AclnnSpec):
    """torch/torch_npu delivery path; argument semantics match ``torch.topk``."""
