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
 * \file test_reflection_pad3d_grad.h
 * \brief
 */

#ifndef TEST_REFLECTION_PAD3D_GRAD_H
#define TEST_REFLECTION_PAD3D_GRAD_H

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__
#pragma pack(1)

struct ReflectionPad3dGradTilingDataInfo {
    uint32_t batch;
    uint32_t channel;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t alignDepth;
    uint32_t alignHeight;
    uint32_t alignWidth;
    uint32_t outDepth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t alignOutDepth;
    uint32_t alignOutHeight;
    uint32_t alignOutWidth;
    uint32_t dPad1;
    uint32_t dPad2;
    uint32_t hPad1;
    uint32_t hPad2;
    uint32_t wPad1;
    uint32_t wPad2;
    uint32_t blockNum;
    uint32_t ubFactorElement;
    uint32_t ncPerCore;
    uint32_t tailNC;
    uint32_t tilingKey;
};

#pragma pack()

inline void InitReflectionPad3dGradTilingData(uint8_t* tiling, ReflectionPad3dGradTilingDataInfo* data)
{
    memcpy(data, tiling, sizeof(ReflectionPad3dGradTilingDataInfo));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    ReflectionPad3dGradTilingDataInfo tilingData;  \
    InitReflectionPad3dGradTilingData(tilingArg, &tilingData)
#endif