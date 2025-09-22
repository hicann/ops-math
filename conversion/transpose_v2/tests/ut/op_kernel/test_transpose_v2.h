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
 * \file test_transpose_v2.h
 * \brief
 */

#ifndef TEST_TRANSPOSE_V2_H
#define TEST_TRANSPOSE_V2_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct TransposeV2TilingDataInfo {
    uint64_t dim0Len{1};
    uint64_t dim1Len{0};
    uint64_t dim2Len{0};
    uint64_t dim3Len{1};
    uint64_t dim3LenAlign{1};
    uint64_t tasksPerCore{0};
    uint64_t tailCore{0};
    uint64_t dim1OnceMax{0};
    uint64_t dim2OnceMax{0};
    uint32_t doubleBuffer{2};
    uint32_t subMode{0};

    int64_t tasksTail{0};
    int64_t inputH{0};
    int64_t inputW{0};
    int64_t inputH16Align{0};
    int64_t inputWAlign{0};
    int64_t hOnce{0};
    int64_t tasksOnceMax{0};
    int64_t repeatH{0};
    int64_t transLoop{0};
};

inline void InitTransposeV2TilingData(uint8_t* tiling, TransposeV2TilingDataInfo* const_data)
{
    memcpy(const_data, tiling, sizeof(TransposeV2TilingDataInfo));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    TransposeV2TilingDataInfo tilingData;          \
    InitTransposeV2TilingData(tilingPointer, &tilingData)
#endif
