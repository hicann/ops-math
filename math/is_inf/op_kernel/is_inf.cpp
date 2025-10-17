/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file is_inf.cpp
 * \brief
 */

#include "is_inf.h"

using namespace IsInfNS;

// kernel function
extern "C" __global__ __aicore__ void is_inf(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = nullptr;

    const int16_t F16_INF_NUM = 0x7c00;
    const int16_t BF16_INF_NUM = 0x7f80;
    const int32_t FLOAT_INF_NUM = 0x7f800000;
    const int16_t SIGN_MASK = 0x7fff;

    if (TILING_KEY_IS(1)) {
        IsInf<half, SIGN_MASK, F16_INF_NUM> op;
        op.Init(inputs, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        IsInf<float, SIGN_MASK, FLOAT_INF_NUM> op;
        op.Init(inputs, outputs, userWS, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(3)) {
        IsInf<half, SIGN_MASK, BF16_INF_NUM> op;
        op.Init(inputs, outputs, userWS, &tilingData);
        op.Process();
#endif
    }
}
