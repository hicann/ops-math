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
 * \file pad_v3_grad.cpp
 * \brief pad_v3_grad kernel
 */

#include "arch35/pad_v3_grad_struct.h"
#include "arch35/pad_v3_grad_tilingkey.h"
#include "arch35/pad_v3_grad_simt.h"

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

using namespace AscendC;

template <uint8_t modeName, bool isBigShape>
__global__ __aicore__ void pad_v3_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PadV3GradACTilingData);
    if constexpr (modeName == TPL_MODE_CONSTANT) {
        PadV3Grad::LaunchConstantKernelSIMT<DTYPE_X, isBigShape>(x, y, tiling);
    } else if constexpr (modeName == TPL_MODE_EDGE) {
        PadV3Grad::LaunchEdgeKernelSIMT<DTYPE_X, isBigShape>(x, y, tiling);
    } else if constexpr (modeName == TPL_MODE_REFLECT || modeName == TPL_MODE_SYMMETRIC) {
        PadV3Grad::LaunchMirrorKernelSIMT<DTYPE_X, isBigShape, modeName>(x, y, tiling);
    } else if constexpr (modeName == TPL_MODE_CIRCULAR) {
        PadV3Grad::LaunchCircularKernelSIMT<DTYPE_X, isBigShape>(x, y, tiling);
    }
}