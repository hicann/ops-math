/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad_apt.cpp
 * \brief pad kernel
 */
#include "../pad_v3/arch35/pad_constant.h"
#include "../pad_v3/arch35/pad_slice.h"
#include "../pad_v3/arch35/pad_tilingkey.h"

using namespace PadV3;

extern "C" __global__ __aicore__ void pad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_NONE_TILING;
    if (TILING_KEY_IS(CONSTANT_CUT_LAST_DIM_BRANCH)) { // 30000
        PadV3::LaunchKernelPadWithHugeWidth<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM2)) { // 30021
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM3)) { // 30031
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM4)) { // 30041
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM2) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM3) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) { // 30002
        PadV3::LaunchKernelPadGather<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM2) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM3) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM4)) { // 30002
        PadV3::LaunchKernelPadScatter<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_SIMT_BRANCH)) { // 20000
        PadV3::LaunchKernelPadSimt<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CONSTANT_SIMT_BIG_SIZE_BRANCH)) { // 20001
        PadV3::LaunchKernelPadSimtHuge<DTYPE_X>(x, paddings, y, tiling);
    } else {
        RunPadSliceProcess(x, y, tiling);
    }
}
