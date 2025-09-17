/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file angle_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ANGLE_V2_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ANGLE_V2_H
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AngleV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(uint32_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, formerLength);
TILING_DATA_FIELD_DEF(uint32_t, tailLength);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);
TILING_DATA_FIELD_DEF(uint32_t, totalLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, dataPerRepeat);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AngleV2, AngleV2TilingData)
struct AngleV2CompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ANGLE_V2_H