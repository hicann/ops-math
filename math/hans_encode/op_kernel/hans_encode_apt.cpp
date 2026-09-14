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
 * \file hans_encode_apt.cpp
 * \brief Ascend 950 entry for HansEncode.
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/hans_encode_simt.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void hans_encode(GM_ADDR input, GM_ADDR pdf, GM_ADDR pdfRef, GM_ADDR mantissa,
                                                  GM_ADDR fixed, GM_ADDR var, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    (void)pdfRef;

    if (TILING_KEY_IS(2)) {
        HansEncodeArch35::HansEncodeSimt<2> op;
        op.Process(input, pdf, mantissa, fixed, var, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(4)) {
        HansEncodeArch35::HansEncodeSimt<4> op;
        op.Process(input, pdf, mantissa, fixed, var, userWorkspace, &tilingData);
    }
}
