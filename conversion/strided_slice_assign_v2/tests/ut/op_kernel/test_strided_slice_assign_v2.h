/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_strided_slice_assign_v2.h
 * \brief
 */

#ifndef _STRIDED_SLICE_ASSIGN_V2_TILING_H_
#define _STRIDED_SLICE_ASSIGN_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct StridedSliceAssignV2TilingDataInfo {
    int64_t dimNum;
    int64_t varDim[8] = {0};
    int64_t inputValueDim[8] = {0};
    int64_t begin[8] = {0};
    int64_t strides[8] = {0};
    int64_t varCumShape[8] = {0};
    int64_t inputCumShape[8] = {0};
};

#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitTilingData(const __gm__ uint8_t *tiling, StridedSliceAssignV2TilingDataInfo *constData) {
    const __gm__ int64_t *src = (const __gm__ int64_t *)tiling;
    int64_t *dst = (int64_t *)constData;
    for (auto i = 0; i < sizeof(StridedSliceAssignV2TilingDataInfo) / sizeof(int64_t); i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, StridedSliceAssignV2TilingDataInfo *constData)
{
    memcpy(constData, tiling, sizeof(StridedSliceAssignV2TilingDataInfo));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg)     \
    StridedSliceAssignV2TilingDataInfo tilingData; \
    InitTilingData(tilingArg, &tilingData)

#define DTYPE_VAR half

#endif