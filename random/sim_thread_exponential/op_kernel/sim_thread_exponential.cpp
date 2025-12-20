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
 * \file sim_thread_exponential.cpp
 * \brief
 */

#include "sim_thread_exponential.h"

extern "C" __global__ __aicore__ void sim_thread_exponential(GM_ADDR self, GM_ADDR self_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);    // outputGM大小如何申请？

    if (TILING_KEY_IS(3)) {
        SimThreadExponential<float> op(&pipe);
        op.Init(self, self_ref, usrWorkspace, &tiling_data);
        op.Process();
    }else if (TILING_KEY_IS(1)) {
        SimThreadExponential<half> op(&pipe);
        op.Init(self, self_ref, usrWorkspace, &tiling_data);
        op.Process();
    }else if (TILING_KEY_IS(2)) {
        SimThreadExponential<bfloat16_t> op(&pipe);
        op.Init(self, self_ref, usrWorkspace, &tiling_data);
        op.Process();
    }
}