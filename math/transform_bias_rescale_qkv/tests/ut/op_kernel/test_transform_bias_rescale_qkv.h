/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FATRELU_MUL_H
#define TEST_FATRELU_MUL_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define DT_FP16 half
#define DT_FP32 float
#define DTYPE_QKV DT_FP32

#define __CCE_UT_TEST__

#define __aicore__

struct TransformBiasRescaleQkvTilingData {
    int64_t qkvShapeSize = 1728;
    uint32_t needCoreNum = 1;
    int64_t batch = 3;
    int64_t token = 4;
    int64_t dimension = 144;
    int64_t numHeads = 3;
    int64_t dimPerHead = 16;
    int64_t maxEleNumUB = 12 * 1024;
};

inline void ITransformBiasRescaleQkvTilingData(uint8_t* tiling, TransformBiasRescaleQkvTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(TransformBiasRescaleQkvTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    TransformBiasRescaleQkvTilingData tilingData;  \
    ITransformBiasRescaleQkvTilingData(tilingPointer, &tilingData)
#endif
