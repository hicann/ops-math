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
 * \file unfold_grad.cpp
 * \brief
 */

#include "unfold_grad_final_axe.h"
#include "unfold_grad_final_second_axe.h"

extern "C" __global__ __aicore__ void unfold_grad(
    GM_ADDR grad_out, GM_ADDR input_sizes, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
#if ORIG_DTYPE_GRAD_OUT == DT_FLOAT32
    if (TILING_KEY_IS(211)) { // fp32
        UnfoldGradFinalAxe<float, float, false> op(&pipe);
        op.InitFinalAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(212)) { // fp32
        UnfoldGradFinalSecondAxe<float, float, false> op(&pipe);
        op.InitFinalSecondAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(311)) { // fp32
        UnfoldGradFinalAxe<float, float, false> op(&pipe);
        op.InitFinalAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(312)) { // fp32
        UnfoldGradFinalSecondAxe<float, float, false> op(&pipe);
        op.InitFinalSecondAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    }
#elif ORIG_DTYPE_GRAD_OUT == DT_FLOAT16
    if (TILING_KEY_IS(221)) { // fp16
        UnfoldGradFinalAxe<half, float, true> op(&pipe);
        op.InitFinalAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(222)) { // fp16
        UnfoldGradFinalSecondAxe<half, float, true> op(&pipe);
        op.InitFinalSecondAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(321)) { // fp16
        UnfoldGradFinalAxe<half, float, true> op(&pipe);
        op.InitFinalAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(322)) { // fp16
        UnfoldGradFinalSecondAxe<half, float, true> op(&pipe);
        op.InitFinalSecondAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    }
#elif ORIG_DTYPE_GRAD_OUT == DT_BF16
    if (TILING_KEY_IS(231)) { // bf16
        UnfoldGradFinalAxe<bfloat16_t, float, true> op(&pipe);
        op.InitFinalAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(232)) { // bf16
        UnfoldGradFinalSecondAxe<bfloat16_t, float, true> op(&pipe);
        op.InitFinalSecondAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(331)) { // bf16
        UnfoldGradFinalAxe<bfloat16_t, float, true> op(&pipe);
        op.InitFinalAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(332)) { // bf16
        UnfoldGradFinalSecondAxe<bfloat16_t, float, true> op(&pipe);
        op.InitFinalSecondAxe(grad_out, grad_in, usrWorkspace, &tiling_data);
        op.Process();
    }
#endif
}