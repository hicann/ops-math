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
 * \file test_unfold_grad.h
 * \brief
 */

#ifndef _FAST_OP_TEST_UNFOLD_GRAD_H_
#define _FAST_OP_TEST_UNFOLD_GRAD_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define __CCE_UT_TEST__

#define __aicore__

struct UnfoldGradTilingDataInfo {
    int64_t batchNum;
    int64_t batchNumPerCore;
    int64_t batchNumTailCore;
    int64_t maxBatchNum4Ub = 0;
    int64_t useCoreNum = 1;
    int64_t ubSizeT1;
    int64_t ubSizeT2;
    int64_t outputNumPerCore;
    int64_t inputNumPerCore;
    int64_t iterationNumPerCore;
    int64_t handleNUMOnceIterationPerCore;
    int64_t tasksOnceMaxPerCore;
    int64_t inputSizeLength;
    int64_t rowAvailableLengthSrc;
    int64_t lowestCommonMultiple;
    int64_t colOnceMaxPerUB;
    int64_t tailColLength;

    int64_t typeSizeT1;
    int64_t typeSizeT2 = 4;
    int64_t width = 8;
    int64_t gradOutSizeDim;
    int64_t inputSizeLastDim;
    int64_t dim;
    int64_t size;
    int64_t step;
    int64_t loop;
    int64_t tail;
};

inline void InitUnfoldGradTilingData(uint8_t* tiling, UnfoldGradTilingDataInfo* const_data)
{
    memcpy(const_data, tiling, sizeof(UnfoldGradTilingDataInfo));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    UnfoldGradTilingDataInfo tilingData;           \
    InitUnfoldGradTilingData(tilingPointer, &tilingData)
#endif