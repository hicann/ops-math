/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_HANS_H_
#define _TEST_HANS_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

#define __aicore__


struct HansDecodeTilingData {
    int64_t fixedByteSize;
    int64_t mantissaByteSize;
    int64_t processLoopLastCore;
    int64_t recoverExpByteSize;
    int64_t recoverByteSize;
    bool reshuff;
};

inline void IHansDecodeTilingData(uint8_t* tiling, HansDecodeTilingData* const_data) {
    memcpy(const_data, tiling, sizeof(HansDecodeTilingData));
}


#define GET_TILING_DATA(tiling_data, tiling_arg) \
HansDecodeTilingData tiling_data; \
IHansDecodeTilingData(tiling_arg, &tiling_data)


#endif