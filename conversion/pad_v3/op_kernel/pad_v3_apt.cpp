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
 * \file pad_v3.cpp
 * \brief pad_v3 kernel
 */

#include "./arch35/pad_constant.h"
#include "./arch35/pad_edge.h"
#include "./arch35/pad_mirror.h"
#include "./arch35/pad_slice.h"
#include "./arch35/pad_tilingkey.h"
#include "./arch35/pad_circular.h"

using namespace PadV3;

#define EDGE_SIMT_BRANCH 23000
#define EDGE_SIMT_BIG_SIZE_BRANCH 23001
#define EDGE_CUT_LAST_DIM_BRANCH 33010
#define EDGE_BIG_LAST_DIM_BRANCH_DIM2 33021
#define EDGE_BIG_LAST_DIM_BRANCH_DIM3 33031
#define EDGE_BIG_LAST_DIM_BRANCH_DIM4 33041
#define EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM2 33022
#define EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM3 33032
#define EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM4 33042

extern "C" __global__ __aicore__ void pad_v3(GM_ADDR x, GM_ADDR paddings, GM_ADDR constValues, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;
    if (TILING_KEY_IS(CONSTANT_CUT_LAST_DIM_BRANCH)) { // 30000
        PadV3::LaunchKernelPadWithHugeWidth<DTYPE_X>(x, paddings, y, tiling, constValues);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM2)) { // 30021
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y, tiling,
                                                                                          constValues);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM3)) { // 30031
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y, tiling,
                                                                                          constValues);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM4)) { // 30041
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y, tiling,
                                                                                          constValues);
    } else if (TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM2) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM3) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) { // 30002
        PadV3::LaunchKernelPadGather<DTYPE_X>(x, paddings, y, tiling, constValues);
    } else if (TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM2) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM3) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM4)) { // 30002
        PadV3::LaunchKernelPadScatter<DTYPE_X>(x, paddings, y, tiling, constValues);
    } else if (TILING_KEY_IS(CONSTANT_SIMT_BRANCH)) { // 20000
        PadV3::LaunchKernelPadSimt<DTYPE_X>(x, paddings, y, tiling, constValues);
    } else if (TILING_KEY_IS(CONSTANT_SIMT_BIG_SIZE_BRANCH)) { // 20001
        PadV3::LaunchKernelPadSimtHuge<DTYPE_X>(x, paddings, y, tiling, constValues);
    }

    else if (TILING_KEY_IS(EDGE_SIMT_BRANCH)) { // 23000
        PadV3::LaunchKernelPadEdgeSimt<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_SIMT_BIG_SIZE_BRANCH)) { // 23001
        PadV3::LaunchKernelPadEdgeSimtHuge<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_CUT_LAST_DIM_BRANCH)) { // 33010
        PadV3::LaunchKernelPadEdgeWithHugeWidth<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_BIG_LAST_DIM_BRANCH_DIM2)) { // 33021
        PadV3::LaunchKernelPadEdgeWithNormalWidth<DTYPE_X, EDGE_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_BIG_LAST_DIM_BRANCH_DIM3)) { // 33031
        PadV3::LaunchKernelPadEdgeWithNormalWidth<DTYPE_X, EDGE_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_BIG_LAST_DIM_BRANCH_DIM4)) { // 33041
        PadV3::LaunchKernelPadEdgeWithNormalWidth<DTYPE_X, EDGE_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM2)) { // 33022
        PadV3::LaunchKernelPadEdgeGather<DTYPE_X, EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM2>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM3)) { // 33032
        PadV3::LaunchKernelPadEdgeGather<DTYPE_X, EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM3>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) { // 33042
        PadV3::LaunchKernelPadEdgeGather<DTYPE_X, EDGE_SMALL_LAST_DIM_GATHER_BRANCH_DIM4>(x, paddings, y, tiling);
    }

    else if (TILING_KEY_IS(REFLECT_SIMT_BRANCH)) { // 21000
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

    else if (TILING_KEY_IS(CIRCULAR_SIMT_BRANCH)) { // 24000
        PadV3::LaunchKernelPadCircularSimt<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CIRCULAR_SIMT_BIG_SIZE_BRANCH)) { // 24001
        PadV3::LaunchKernelPadCircularSimtHuge<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CIRCULAR_CUT_LAST_DIM_BRANCH)) { // 34010
        PadV3::LaunchKernelPadCircularWithHugeWidth<DTYPE_X>(x, paddings, y, tiling);
    } else if (TILING_KEY_IS(CIRCULAR_BIG_LAST_DIM_BRANCH_DIM2)) { // 34021
        PadV3::LaunchKernelPadCircularWithNormalWidth<DTYPE_X, CIRCULAR_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y,
                                                                                                  tiling);
    } else if (TILING_KEY_IS(CIRCULAR_BIG_LAST_DIM_BRANCH_DIM3)) { // 34031
        PadV3::LaunchKernelPadCircularWithNormalWidth<DTYPE_X, CIRCULAR_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y,
                                                                                                  tiling);
    } else if (TILING_KEY_IS(CIRCULAR_BIG_LAST_DIM_BRANCH_DIM4)) { // 34041
        PadV3::LaunchKernelPadCircularWithNormalWidth<DTYPE_X, CIRCULAR_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y,
                                                                                                  tiling);
    } else if (TILING_KEY_IS(CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH_DIM2)) { // 34022
        PadV3::LaunchKernelPadCircularGather<DTYPE_X, CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH_DIM2>(x, paddings, y,
                                                                                                  tiling);
    } else if (TILING_KEY_IS(CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH_DIM3)) { // 34032
        PadV3::LaunchKernelPadCircularGather<DTYPE_X, CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH_DIM3>(x, paddings, y,
                                                                                                  tiling);
    } else if (TILING_KEY_IS(CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) { // 34042
        PadV3::LaunchKernelPadCircularGather<DTYPE_X, CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH_DIM4>(x, paddings, y,
                                                                                                  tiling);
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
