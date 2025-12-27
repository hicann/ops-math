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
 * \file as_strided.cpp
 * \brief as_strided
 */

#include <cstdint>
#include "./arch35/as_strided_move_align.h"
#include "./arch35/as_strided_dual_cut.h"
#include "./arch35/as_strided.h"

using namespace AsStrided;

#define AS_STRIDED_B8 1
#define AS_STRIDED_B16 2
#define AS_STRIDED_B32 4
#define AS_STRIDED_B64 8
#define AS_STRIDED_MOVE_ALIGN_B8 101
#define AS_STRIDED_MOVE_ALIGN_B16 102
#define AS_STRIDED_MOVE_ALIGN_B32 104
#define AS_STRIDED_MOVE_ALIGN_B64 108
#define AS_STRIDED_DUAL_CUT 200

extern "C" __global__ __aicore__ void as_strided(
    GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride, GM_ADDR storageOffset, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(AS_STRIDED_B8)) {
        KernelAsStrided<int8_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_B16)) {
        KernelAsStrided<half> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_B32)) {
        KernelAsStrided<float> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_B64)) {
        KernelAsStrided<int64_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B8)) {
        KernelAsStridedMoveAlign<int8_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B16)) {
        KernelAsStridedMoveAlign<half> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B32)) {
        KernelAsStridedMoveAlign<float> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B64)) {
        KernelAsStridedMoveAlign<int64_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_DUAL_CUT)) {
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            KernelAsStridedDualCut<int8_t, AS_STRIDED_DUAL_CUT> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();
        } else {
            KernelAsStridedDualCut<DTYPE_X, AS_STRIDED_DUAL_CUT> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();
        }
    }
}