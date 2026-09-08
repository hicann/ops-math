# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import ClassVar

import numpy as np
import torch

__spec__ = {"dense_bincount": "DenseBincountKernelSpec"}

_TOLERANCE = {"float32": {"standard": "cross_check", "level": "L1"}}


def _bool_attr(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _prepare_indices(input_tensor, size_tensor):
    size = int(size_tensor.reshape(-1)[0].item())
    rank = input_tensor.dim()
    rows = 1 if rank == 1 else input_tensor.shape[0]
    cols = input_tensor.numel() if rank == 1 else input_tensor.shape[1]
    bins = input_tensor.reshape(-1).to(torch.int64)
    row_indices = (
        torch.zeros_like(bins)
        if rank == 1
        else torch.arange(bins.numel(), device=bins.device, dtype=torch.int64) // cols
    )

    if rank == 2 and size > 0:
        negative = bins < 0
        row_delta = torch.where(
            negative, torch.div(bins, size, rounding_mode="floor"), 0
        )
        bins = torch.where(negative, torch.remainder(bins, size), bins)
        row_indices = row_indices + row_delta

    valid = (bins >= 0) & (bins < size) & (row_indices >= 0) & (row_indices < rows)
    linear_indices = row_indices[valid] * size + bins[valid] if size > 0 else bins[:0]
    return size, rows, linear_indices, valid


def _compute(input_tensor, size_tensor, weights_tensor, binary_output=False):
    size, rows, linear_indices, valid = _prepare_indices(input_tensor, size_tensor)
    output_shape = (size,) if input_tensor.dim() == 1 else (rows, size)
    if size <= 0:
        return [
            torch.zeros(
                output_shape, dtype=weights_tensor.dtype, device=input_tensor.device
            )
        ]

    use_binary = _bool_attr(binary_output)
    if use_binary or weights_tensor.numel() == 0:
        values = torch.ones(
            linear_indices.numel(),
            dtype=weights_tensor.dtype,
            device=input_tensor.device,
        )
    else:
        values = weights_tensor.reshape(-1)[valid]
    output = torch.bincount(
        linear_indices, weights=values, minlength=rows * size
    ).reshape(output_shape)
    if use_binary:
        output = (output != 0).to(output.dtype)
    return [output]


class _DenseBincountCompose:
    def __init__(self, binary_output=False, **kwargs):
        del kwargs
        self.binary_output = _bool_attr(binary_output)

    def __call__(self, input, size, weights, **kwargs):
        del kwargs
        bin_count = int(size.reshape(-1)[0].item())
        rows = 1 if input.dim() == 1 else input.shape[0]
        output_shape = (bin_count,) if input.dim() == 1 else (rows, bin_count)
        if bin_count <= 0:
            return [torch.zeros(output_shape, dtype=weights.dtype, device=input.device)]

        bins = input.reshape(-1).to(torch.int64)
        if input.dim() == 1:
            mapped_rows = torch.zeros_like(bins)
        else:
            cols = input.shape[1]
            mapped_rows = (
                torch.arange(bins.numel(), device=bins.device, dtype=torch.int64)
                // cols
            )
            negative = bins < 0
            signed_remainder = torch.fmod(bins, bin_count)
            trunc_quotient = torch.div(bins, bin_count, rounding_mode="trunc")
            row_delta = trunc_quotient - (signed_remainder != 0).to(torch.int64)
            bins = torch.where(
                negative,
                torch.where(
                    signed_remainder == 0,
                    torch.zeros_like(bins),
                    signed_remainder + bin_count,
                ),
                bins,
            )
            mapped_rows = mapped_rows + torch.where(
                negative, row_delta, torch.zeros_like(row_delta)
            )
        valid = (
            (bins >= 0) & (bins < bin_count) & (mapped_rows >= 0) & (mapped_rows < rows)
        )
        linear_indices = mapped_rows[valid] * bin_count + bins[valid]

        if self.binary_output or weights.numel() == 0:
            values = torch.ones(
                linear_indices.numel(), dtype=weights.dtype, device=input.device
            )
        else:
            values = weights.reshape(-1)[valid]
        output = torch.zeros(rows * bin_count, dtype=values.dtype, device=input.device)
        output.scatter_add_(0, linear_indices, values)
        if self.binary_output:
            output = (output != 0).to(output.dtype)
        return [output.reshape(output_shape)]


class DenseBincountKernelSpec:
    def golden(*inputs, **kwargs):
        tensors = [torch.from_numpy(np.ascontiguousarray(value)) for value in inputs]
        outputs = _compute(*tensors, binary_output=kwargs.get("binary_output", False))
        output_dtypes = kwargs.get("output_dtypes") or []
        output_dtypes = [
            dtype[0] if isinstance(dtype, (list, tuple)) else str(dtype)
            for dtype in output_dtypes
        ]
        return [
            output.numpy().astype(output_dtypes[index])
            if index < len(output_dtypes)
            else output.numpy()
            for index, output in enumerate(outputs)
        ]

    third_party: ClassVar[dict] = {"torch": _DenseBincountCompose}
    tolerance: ClassVar[dict] = _TOLERANCE


def dense_bincount_golden(input_tensor, size_tensor, weights_tensor, **kwargs):
    """Compatibility entry for lightweight repository probes; TTK uses __spec__."""
    return DenseBincountKernelSpec.golden(
        input_tensor, size_tensor, weights_tensor, **kwargs
    )


# No aclnn/e2e spec: DenseBincount has no aclnn interface or torch binding.
