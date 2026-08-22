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

"""BroadcastTo 算子 Kernel/GEIR 和 E2E 流程的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，需手动转 torch 计算后转回 numpy；
E2E 的 golden 直接收到 torch.Tensor，无需转换。
"""

__spec__ = {
    "broadcast_to": "BroadcastToKernelSpec",
    "torch.broadcast_to": "TorchBroadcastToSpec",
}

import numpy as np
import torch


def _parse_shape(shape):
    if isinstance(shape, np.ndarray):
        return shape.tolist()
    if isinstance(shape, torch.Tensor):
        return shape.tolist()
    return list(shape)


class BroadcastToKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor"""

    def golden(x, shape, **kwargs):
        x_t = torch.from_numpy(x)
        shape_val = _parse_shape(shape)
        return [torch.broadcast_to(x_t, tuple(shape_val)).contiguous().numpy()]

    third_party = {"torch": "torch.broadcast_to"}
    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "bool": {"standard": "binary_equal"},
        "hifloat8": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float8_e4m3fn": {"standard": "binary_equal"},
    }


class TorchBroadcastToSpec:
    """E2E 流程 — golden / third_party 均收到 torch.Tensor（已在设备上）"""

    def golden(x, shape, **kwargs):
        shape_val = _parse_shape(shape)
        return [torch.broadcast_to(x, tuple(shape_val)).contiguous()]

    third_party = {"torch": "torch.broadcast_to"}
    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "bool": {"standard": "binary_equal"},
        "hifloat8": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float8_e4m3fn": {"standard": "binary_equal"},
    }
