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
 * \file stateless_drop_out_gen_mask.cpp
 * \brief
 */

#define PT_FP32_TILING_KEY 1001
#define PT_FP16_TILING_KEY 1002
#define PT_BF16_TILING_KEY 1003
#define TF_FP32_TILING_KEY 1004
#define TF_FP16_TILING_KEY 1005
#define TF_BF16_TILING_KEY 1006

#include "arch35/stateless_drop_out_gen_mask_tf.h"
#include "arch35/stateless_drop_out_gen_mask_pt.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void stateless_drop_out_gen_mask(
    GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR seed1, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (TILING_KEY_IS(PT_FP32_TILING_KEY)) {
        StatelessDropOutGenMask::StatelessDropOutGenMaskPt<float> op;
        op.Init(shape, prob, y, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(PT_FP16_TILING_KEY)) {
        StatelessDropOutGenMask::StatelessDropOutGenMaskPt<half> op;
        op.Init(shape, prob, y, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(PT_BF16_TILING_KEY)) {
        StatelessDropOutGenMask::StatelessDropOutGenMaskPt<bfloat16_t> op;
        op.Init(shape, prob, y, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TF_FP32_TILING_KEY)) {
        StatelessDropOutGenMask::StatelessDropOutGenMaskTf<float> op;
        op.Init(shape, prob, y, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TF_FP16_TILING_KEY)) {
        StatelessDropOutGenMask::StatelessDropOutGenMaskTf<half> op;
        op.Init(shape, prob, y, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TF_BF16_TILING_KEY)) {
        StatelessDropOutGenMask::StatelessDropOutGenMaskTf<bfloat16_t> op;
        op.Init(shape, prob, y, userWS, &tilingData, &pipe);
        op.Process();
    }
}
