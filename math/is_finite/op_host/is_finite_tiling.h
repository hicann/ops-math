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
 * \file is_finite_tiling_def.h
 * \brief
 */

#ifndef IS_FINITE_TILING_DEF_H
#define IS_FINITE_TILING_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {
struct IsFiniteCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};

BEGIN_TILING_DATA_DEF(IsFiniteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usableUbSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalDataCount);
TILING_DATA_FIELD_DEF(uint64_t, perCoreDataCount);
TILING_DATA_FIELD_DEF(uint64_t, tailDataCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreDataCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsFinite, IsFiniteTilingData)
} // namespace optiling

#endif // IS_FINITE_TILING_DEF_H
