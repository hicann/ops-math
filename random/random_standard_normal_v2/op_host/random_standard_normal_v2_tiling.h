/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file random_standard_normal_v2_tiling.h
 * \brief
 */
#ifndef Random_Standard_Normal_V2_TILING_H
#define Random_Standard_Normal_V2_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
struct RandomStandardNormalV2CompileInfo {
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

} // namespace optiling

#endif // Random_Standard_Normal_V2_TILING_H
