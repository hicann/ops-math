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
 * \file stateless_bernoulli.cpp
 * \brief
 */

#define TILING_KEY_FP32 1001
#define TILING_KEY_FP16 1002
#define TILING_KEY_BF16 1003

#include "arch35/stateless_bernoulli.h"

KERNEL_API void stateless_bernoulli(
    GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (AscendC::IsSameType<DTYPE_Y, bool>::value) {
        if (TILING_KEY_IS(TILING_KEY_FP32)) {
            StatelessBernoulli::StatelessBernoulliKernel<float, int8_t> op;
            op.Init(shape, prob, y, workspace, &tilingData, &pipe);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(TILING_KEY_FP16)) {
            StatelessBernoulli::StatelessBernoulliKernel<half, int8_t> op;
            op.Init(shape, prob, y, workspace, &tilingData, &pipe);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(TILING_KEY_BF16)) {
            StatelessBernoulli::StatelessBernoulliKernel<bfloat16_t, int8_t> op;
            op.Init(shape, prob, y, workspace, &tilingData, &pipe);
            op.Process(&tilingData);
        }
    } else {
        if (TILING_KEY_IS(TILING_KEY_FP32)) {
            StatelessBernoulli::StatelessBernoulliKernel<float, DTYPE_Y> op;
            op.Init(shape, prob, y, workspace, &tilingData, &pipe);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(TILING_KEY_FP16)) {
            StatelessBernoulli::StatelessBernoulliKernel<half, DTYPE_Y> op;
            op.Init(shape, prob, y, workspace, &tilingData, &pipe);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(TILING_KEY_BF16)) {
            StatelessBernoulli::StatelessBernoulliKernel<bfloat16_t, DTYPE_Y> op;
            op.Init(shape, prob, y, workspace, &tilingData, &pipe);
            op.Process(&tilingData);
        }
    }
}