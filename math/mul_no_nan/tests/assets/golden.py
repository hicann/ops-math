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

__spec__ = {
    "mul_no_nan": "MulNoNanKernelSpec",
    "tf.raw_ops.MulNoNan": "MulNoNanTensorFlowSpec",
}

__golden__ = {"kernel": {"mul_no_nan": "mul_no_nan_golden"}}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}

_TENSORFLOW_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
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


class _MulNoNanTfCompose:
    """TensorFlow reference executed by the isolated third-party provider."""

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        import tensorflow as tf

        # TensorFlow's native MulNoNan excludes int32, while the CANN kernel
        # supports it. Keep the named raw op for its supported floating dtypes
        # and use the equivalent TensorFlow primitive composition for int32.
        if x1.dtype.is_integer:
            product = tf.multiply(x1, x2)
            return [tf.where(tf.equal(x2, 0), tf.zeros_like(product), product)]
        return [tf.raw_ops.MulNoNan(x=x1, y=x2)]


class MulNoNanKernelSpec:
    """kernel spec: numpy golden + third-party reference + precision standard."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return [mul_no_nan_golden(x1, x2, **kwargs)]

    # Keep Torch first so existing unfiltered cross-check behavior is unchanged;
    # TensorFlow can be selected explicitly with ``--provider tf``.
    third_party = {"torch": _MulNoNanCompose, "tf": _MulNoNanTfCompose}
    tolerance = _KERNEL_TOLERANCE


def _to_numpy_array(value):
    """Convert a NumPy or framework tensor to an independent host array."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if hasattr(value, "numpy"):
        return np.array(value.numpy(), copy=True)
    return np.array(value, copy=True)


def _promote_reference_array(array):
    """Promote floating inputs so cross-check uses an independent CPU truth."""
    target_dtype = {
        "float16": np.float32,
        "bfloat16": np.float32,
        "float32": np.float64,
    }.get(array.dtype.name)
    return array.astype(target_dtype) if target_dtype is not None else array


def tensorflow_mul_no_nan_golden(x, y, name=None, **kwargs):
    """CPU golden for ``tf.raw_ops.MulNoNan``."""
    del name, kwargs
    x_array = _promote_reference_array(_to_numpy_array(x))
    y_array = _promote_reference_array(_to_numpy_array(y))
    with np.errstate(invalid="ignore", over="ignore"):
        product = np.multiply(x_array, y_array)
    return [np.where(y_array == 0, np.zeros((), dtype=product.dtype), product)]


class MulNoNanTensorFlowSpec:
    """TensorFlow E2E spec registered by the CSV ``api_name``."""

    golden = staticmethod(tensorflow_mul_no_nan_golden)
    third_party = {"tf": "tf.raw_ops.MulNoNan"}
    # tf.raw_ops.MulNoNan itself supports floating/complex tensors, not int32.
    # This CANN operator exposes only the three floating types from that overlap.
    tolerance = _TENSORFLOW_TOLERANCE
