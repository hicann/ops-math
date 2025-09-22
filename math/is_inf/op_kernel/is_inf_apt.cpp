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
 * \file is_inf.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "v35/is_inf_dag.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace Ops::Base;
using namespace AscendC;

extern "C" __global__ __aicore__ void is_inf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
        GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
        ElementwiseSch<0UL, IsInfOp::IsInfDAG<half>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(102UL)) {
        REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
        GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
        ElementwiseSch<0UL, IsInfOp::IsInfDAG<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(103UL)) {
        REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
        GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
        ElementwiseSch<0UL, IsInfOp::IsInfDAG<float>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}