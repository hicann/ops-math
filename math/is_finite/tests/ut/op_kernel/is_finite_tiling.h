/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _REPEAT_INTERLEAVE_GRAD_TILING_H_
#define _REPEAT_INTERLEAVE_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct IsFiniteTilingData {
    uint32_t usableUbSize;
    uint32_t needCoreNum;
    uint64_t totalDataCount;
    uint64_t perCoreDataCount;
    uint64_t tailDataCoreNum;
    uint64_t lastCoreDataCount;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                          \
    IsFiniteTilingData tilingData;                                          \
    INIT_TILING_DATA(IsFiniteTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).totalDataCount = tilingDataPointer->totalDataCount;        \
    (tilingData).usableUbSize = tilingDataPointer->usableUbSize;            \
    (tilingData).needCoreNum = tilingDataPointer->needCoreNum;              \
    (tilingData).perCoreDataCount = tilingDataPointer->perCoreDataCount;    \
    (tilingData).tailDataCoreNum = tilingDataPointer->tailDataCoreNum;      \
    (tilingData).lastCoreDataCount = tilingDataPointer->lastCoreDataCount;
#endif