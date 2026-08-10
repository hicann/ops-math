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
import torch

__golden__ = {
    "kernel": {"add_v2": "add_v2_golden"},
}


def add_v2_golden(x1, x2, **kwargs):
    """
    Kernel golden for add_v2.
    All the parameters follow @add_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    # 仅注册同 dtype 组合，x1/x2 dtype 恒等，输出 dtype 与 x1 一致
    dtype = x1.dtype
    if str(x2.dtype) != str(dtype):
        raise ValueError(
            f"add_v2 only supports identical input dtypes, got x1={dtype}, x2={x2.dtype}"
        )
    # torch 无原生 bfloat16 numpy 视图，先升 float32 计算再还原
    if "bfloat16" in str(dtype):
        x1 = x1.astype("float32")
        x2 = x2.astype("float32")
    x = torch.from_numpy(x1)
    y = torch.from_numpy(x2)
    res = torch.add(x, y).numpy()
    if "bfloat16" in str(dtype):
        res = res.astype(dtype)

    return res
