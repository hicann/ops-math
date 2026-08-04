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
"""Golden plugin for FusedMulAddNL2loss（torch 竞品算子拼接实现）。

    y1 = x1 * x3 + x2          (elementwise, x3 标量广播)
    y2 = sum(x1^2 / 2)         (全量 reduce, 标量)

torch float64 计算后 cast 到输入 dtype（y1/y2 dtype 均与 x1 一致，对齐 910b 语义）。
"""

import numpy as np
import torch

__golden__ = {
    "kernel": {"fused_mul_addn_l2loss": "fused_mul_addn_l2loss_golden"},
    "e2e": {"aclnnFusedMulAddNL2loss": "fused_mul_addn_l2loss_golden"},
}


def _to_torch_f64(tensor):
    """输入归一为 torch float64（接受 numpy / torch tensor；ml_dtypes.bfloat16 等 numpy 扩展 dtype 先升 fp32）。"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().to(torch.float64)
    arr = np.asarray(tensor)
    if arr.dtype not in (
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.int16,
        np.int8,
        np.uint8,
    ):
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(torch.float64)


def _out_dtype(x1):
    if isinstance(x1, torch.Tensor):
        return x1.detach().cpu().numpy().dtype
    return np.asarray(x1).dtype


def fused_mul_addn_l2loss_golden(x1, x2, x3, *args, **kwargs):
    """Golden for fused_mul_addn_l2loss. Parameters follow op proto (x1, x2, x3)。

    kernel 模式传 3 个输入；aclnn(e2e) 模式传全部 5 个 tensor（含 y1/y2 占位），忽略多余参数。
    """
    del args, kwargs
    out_dtype = _out_dtype(x1)
    x1_t = _to_torch_f64(x1)
    x2_t = _to_torch_f64(x2)
    x3_t = _to_torch_f64(x3)

    y1 = (x1_t * x3_t + x2_t).numpy().astype(out_dtype)
    y2 = ((x1_t * x1_t).sum() * 0.5).numpy().astype(out_dtype)
    return [y1, y2.reshape(1)]  # y2 标量
