/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file random_standard_normal_v2.cpp
 * \brief RandomStandardNormalV2 Kernel 入口（Ascend950/Ascend350/Ascend960dt/Ascend960pr）
 */
// ascend950/ascend350(A5)与ascend960dt/ascend960pr(A6)芯片：共用arch35目录下的实现
#include "arch35/random_standard_normal_v2.h"

using namespace AscendC;
using namespace RandomStandardNormalV2;

#define RANDOM_STANDARD_NORMAL_V2_DEFAULT_TILING_KEY 100

extern "C" __global__ __aicore__ void random_standard_normal_v2(GM_ADDR shape, GM_ADDR offset, GM_ADDR y, GM_ADDR count,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    if (TILING_KEY_IS(RANDOM_STANDARD_NORMAL_V2_DEFAULT_TILING_KEY)) {
        RandomStandardNormalV2::RandomStandardNormalV2Op<DTYPE_Y, DTYPE_OFFSET> op(&pipe, &tilingData);
        op.Init(offset, y, count);
        op.Process();
    }
}
