/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANNDEV_TEST_STACK_BALL_QUERY_H
#define CANNDEV_TEST_STACK_BALL_QUERY_H

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct StackBallQueryTilingDataInfo {
    int32_t batchSize;
    int32_t totalLengthCenterXyz;
    int32_t totalLengthXyz;
    int32_t totalIdxLength;
    int32_t coreNum;
    int32_t centerXyzPerCore;
    int32_t tailCenterXyzPerCore;
    float maxRadius;
    int32_t sampleNum;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, element) \
    (tilingData).element = tilingDataPointer->element

#define GET_TILING_DATA(tilingData, tilingPointer)                                    \
    StackBallQueryTilingDataInfo tilingData;                                          \
    INIT_TILING_DATA(StackBallQueryTilingDataInfo, tilingDataPointer, tilingPointer); \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, batchSize);               \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, totalLengthCenterXyz);    \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, totalLengthXyz);          \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, totalIdxLength);          \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, coreNum);                 \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, centerXyzPerCore);        \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, tailCenterXyzPerCore);    \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, maxRadius);               \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, sampleNum);

#endif // CANNDEV_TEST_STACK_BALL_QUERY_H
