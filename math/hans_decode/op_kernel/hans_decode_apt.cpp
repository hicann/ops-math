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
 * \file hans_decode_apt.cpp
 * \brief Ascend 950 entry for HansDecode.
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#ifdef __CCE_UT_TEST__
#include "../../hans_encode/op_kernel/hans_format.h"
#else
#include "../hans_encode/hans_format.h"
#endif
#include "arch35/hans_decode_simt.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void hans_decode(GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf,
                                                  GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    SetSysWorkspace(workspace);

    if (TILING_KEY_IS(2)) {
        HansDecodeArch35::HansDecodeSimt<2> op;
        op.Process(mantissa, fixed, var, pdf, output, &tilingData);
    } else if (TILING_KEY_IS(4)) {
        HansDecodeArch35::HansDecodeSimt<4> op;
        op.Process(mantissa, fixed, var, pdf, output, &tilingData);
    }
}
