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

#ifndef _TEST_HANS_H_
#define _TEST_HANS_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

#define __aicore__

struct HansEncodeTilingData {
    int64_t processCoreDim;
    int64_t processLoopPerCore;
    int64_t processLoopLastCore;
    int64_t fixedLengthPerCore;
    int64_t fixedLengthLastCore;
    int64_t varLength;
    bool statistic;
    bool reshuff;
};

inline void IHansEncodeTilingData(uint8_t* tiling, HansEncodeTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(HansEncodeTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    HansEncodeTilingData tiling_data;            \
    IHansEncodeTilingData(tiling_arg, &tiling_data)
#endif