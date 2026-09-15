/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mirror_pad.cpp
 * \brief mirror_pad
 */
#include "../pad_v3/arch35/pad_mirror.h"
#include "../pad_v3/arch35/pad_slice.h"
#include "../pad_v3/arch35/pad_tilingkey.h"

using namespace PadV3;

extern "C" __global__ __aicore__ void mirror_pad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_NONE_TILING;

    if (TILING_KEY_IS(REFLECT_SIMT_BRANCH)) { // 21000
        PadV3::LaunchKernelPadMirrorSimt<DTYPE_X, REFLECT_SIMT_BRANCH>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_SIMT_BIG_SIZE_BRANCH)) { // 21001
        PadV3::LaunchKernelPadMirrorSimtHuge<DTYPE_X, REFLECT_SIMT_BIG_SIZE_BRANCH>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_CUT_LAST_DIM_BRANCH)) { // 31010
        PadV3::LaunchKernelPadMirrorWithHugeWidth<DTYPE_X, REFLECT_CUT_LAST_DIM_BRANCH>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_BIG_LAST_DIM_BRANCH_DIM2)) { // 31021
        PadV3::LaunchKernelPadMirrorWithNormalWidth<DTYPE_X, REFLECT_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_BIG_LAST_DIM_BRANCH_DIM3)) { // 31031
        PadV3::LaunchKernelPadMirrorWithNormalWidth<DTYPE_X, REFLECT_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_BIG_LAST_DIM_BRANCH_DIM4)) { // 31041
        PadV3::LaunchKernelPadMirrorWithNormalWidth<DTYPE_X, REFLECT_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_SMALL_LAST_DIM_GATHER_BRANCH_DIM2)) { // 31022
        PadV3::LaunchKernelPadMirrorGather<DTYPE_X, REFLECT_SMALL_LAST_DIM_GATHER_BRANCH_DIM2>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_SMALL_LAST_DIM_GATHER_BRANCH_DIM3)) { // 31032
        PadV3::LaunchKernelPadMirrorGather<DTYPE_X, REFLECT_SMALL_LAST_DIM_GATHER_BRANCH_DIM3>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(REFLECT_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) { // 31042
        PadV3::LaunchKernelPadMirrorGather<DTYPE_X, REFLECT_SMALL_LAST_DIM_GATHER_BRANCH_DIM4>(x, paddings, y, tiling);
    }

    else if (TILING_KEY_IS(SYMMETRIC_SIMT_BRANCH)) { // 22000
        PadV3::LaunchKernelPadMirrorSimt<DTYPE_X, SYMMETRIC_SIMT_BRANCH>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_SIMT_BIG_SIZE_BRANCH)) { // 22001
        PadV3::LaunchKernelPadMirrorSimtHuge<DTYPE_X, SYMMETRIC_SIMT_BIG_SIZE_BRANCH>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_CUT_LAST_DIM_BRANCH)) { // 32010
        PadV3::LaunchKernelPadMirrorWithHugeWidth<DTYPE_X, SYMMETRIC_CUT_LAST_DIM_BRANCH>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_BIG_LAST_DIM_BRANCH_DIM2)) { // 32021
        PadV3::LaunchKernelPadMirrorWithNormalWidth<DTYPE_X, SYMMETRIC_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y,
                                                                                                 tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_BIG_LAST_DIM_BRANCH_DIM3)) { // 32031
        PadV3::LaunchKernelPadMirrorWithNormalWidth<DTYPE_X, SYMMETRIC_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y,
                                                                                                 tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_BIG_LAST_DIM_BRANCH_DIM4)) { // 32041
        PadV3::LaunchKernelPadMirrorWithNormalWidth<DTYPE_X, SYMMETRIC_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y,
                                                                                                 tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH_DIM2)) { // 32022
        PadV3::LaunchKernelPadMirrorGather<DTYPE_X, SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH_DIM2>(x, paddings, y,
                                                                                                 tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH_DIM3)) { // 32032
        PadV3::LaunchKernelPadMirrorGather<DTYPE_X, SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH_DIM3>(x, paddings, y,
                                                                                                 tiling);
    } else if (TILING_KEY_IS(SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) { // 32042
        PadV3::LaunchKernelPadMirrorGather<DTYPE_X, SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH_DIM4>(x, paddings, y,
                                                                                                 tiling);
    } else {
        RunPadSliceProcess(x, y, tiling);
    }
}
