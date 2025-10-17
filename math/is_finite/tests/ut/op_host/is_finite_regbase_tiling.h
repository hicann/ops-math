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

#ifndef _GE_IS_FINITE_REGBASE_TILING_H_
#define _GE_IS_FINITE_REGBASE_TILING_H_

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct EleBaseTilingData {
    uint64_t scheMode;
    int64_t dim0;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t ubFormer;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
};

struct IsFiniteRegbaseTilingData {
    EleBaseTilingData baseTiling;
};

#pragma pack()

inline void InitTilingData(uint8_t* tiling, IsFiniteRegbaseTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(IsFiniteRegbaseTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    IsFiniteRegbaseTilingData tiling_data;       \
    InitTilingData(tiling_arg, &tiling_data)
#endif // _GE_IS_FINITE_REGBASE_TILING_H_