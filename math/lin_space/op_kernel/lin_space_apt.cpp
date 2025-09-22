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
 * \file lin_space_apt.cpp
 * \brief
 */

#define DOUBLE_CAST_TILING_KEY 1002

#include "v35/lin_space_double_cast.h"

using namespace LinSpace;

extern "C" __global__ __aicore__ void lin_space(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(LinSpaceRegbaseTilingData, linSpaceTilingData, tiling);
    const LinSpaceRegbaseTilingData* __restrict tilingData = &linSpaceTilingData;

    TPipe pipe;
    if(TILING_KEY_IS(DOUBLE_CAST_TILING_KEY)) {
        LinSpace::LinSpaceDoubleCast<float, DTYPE_OUTPUT> op;
        op.Init(start, stop, num, output, userWs, tilingData, &pipe);
        op.Process(tilingData);
    }
}
