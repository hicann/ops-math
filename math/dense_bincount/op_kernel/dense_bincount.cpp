/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/dense_bincount_regbase.h"
#include "arch35/dense_bincount_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void dense_bincount(GM_ADDR input, GM_ADDR size, GM_ADDR weights, GM_ADDR output,
                                          GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DenseBincountTilingData);
    GET_TILING_DATA_WITH_STRUCT(DenseBincountTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    constexpr bool IS_1D = schMode >= 4;
    constexpr bool BINARY_OUTPUT = (schMode & 0x2) != 0;
    constexpr bool HAS_WEIGHTS = (schMode & 0x1) != 0;
    NsDenseBincount::DenseBincountRegbase<DTYPE_INPUT, DTYPE_SIZE, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS> op(tilingData,
                                                                                                         &pipe);
    op.Init(input, size, weights, output);
    op.Process();
}
