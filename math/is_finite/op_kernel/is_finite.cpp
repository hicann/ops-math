/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file is_finite.cpp
 * \brief
 */

#include "is_finite.h"

using namespace IsFiniteNs;

// kernel function
extern "C" __global__ __aicore__ void is_finite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = nullptr;

    const int16_t HALF_TYPE_MASK = 0x7c00;      // 0111 1100 0000 0000
    const int32_t FLOAT_TYPE_MASK = 0x7f800000; // 0111 1111 1000 0000 0000 0000 0000 0000
    const int16_t BF16_TYPE_MASK = 0x7f80;      // 0111 1111 1000 0000

    if (TILING_KEY_IS(1)) {
        IsFiniteNs::IsFinite<half, HALF_TYPE_MASK> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        IsFiniteNs::IsFinite<float, BF16_TYPE_MASK> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        IsFiniteNs::IsFinite<half, BF16_TYPE_MASK> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}