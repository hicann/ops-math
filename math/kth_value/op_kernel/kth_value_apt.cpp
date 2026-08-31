/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/kth_value_kernel_dispatch.h"

using namespace AscendC;

template <uint64_t schId, uint64_t isInt32>
__global__ __aicore__ void kth_value(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(KthValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(KthValueTilingData, tilingData, tiling);
    TPipe pipe;
    KthValue::Dispatch<false, schId, isInt32>(x, y1, y2, workspace, &tilingData, &pipe);
}
