#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__spec__ = {"mul_no_nan": "MulNoNanKernelSpec"}

__golden__ = {"kernel": {"mul_no_nan": "mul_no_nan_golden"}}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def mul_no_nan_golden(x1, x2, **kwargs):
    """
    Kernel golden for mul_no_nan.
    All the parameters follow @mul_no_nan_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Semantics: y = (x2 == 0) ? 0 : x1 * x2 (element-wise, NumPy broadcasting).
    The mask is on x2 ONLY, so wherever x2 == 0 the product is replaced by 0
    regardless of what that product would have been -- including NaN produced
    by x1 == inf (inf * 0) or by x1 == NaN. Conversely, where x2 != 0 the
    product is returned as is, so x1 == 0 with x2 == inf still yields NaN and
    NOT 0. This is the core differentiator vs plain Mul.
    """
    dtype = x1.dtype
    if dtype == np.int32:
        # Integer path: no NaN/Inf, kernel computes in native int32.
        prod = (x1 * x2).astype(np.int32)
        out = np.where(x2 == 0, np.int32(0), prod).astype(np.int32)
    else:
        # Float / half / bf16 path: lift to fp32 for the intermediate compute
        # to match the kernel's MulNoNanFloatCast template for fp16/bf16, and
        # to be a no-op for the native fp32 template. Cast back at the end.
        x1f = x1.astype(np.float32)
        x2f = x2.astype(np.float32)
        mask = x2f == np.float32(0.0)
        prod = x1f * x2f
        # np.where picks element-wise, so wherever mask is True we get 0 even
        # if prod[i] is NaN/Inf -- this is exactly the MulNoNan semantics.
        out = np.where(mask, np.float32(0.0), prod).astype(dtype)
    return out


class _MulNoNanCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        import torch  # the remote server executes this, the local golden stays numpy-only

        prod = torch.mul(x1, x2)
        return [torch.where(torch.eq(x2, 0), torch.zeros_like(prod), prod)]


class MulNoNanKernelSpec:
    """kernel spec: numpy golden + third-party reference + precision standard."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return [mul_no_nan_golden(x1, x2, **kwargs)]

    third_party = {"torch": _MulNoNanCompose}
    tolerance = _KERNEL_TOLERANCE
