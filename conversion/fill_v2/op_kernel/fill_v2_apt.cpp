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
 * \file fill_v2_apt.cpp
 * \brief fill_v2 kernel
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/fill_v2_dag.h"
#include "arch35/fill_v2_tiling_key.h"
#include "arch35/fill_v2_tilingdata.h"
#include "atvoss/elewise/elewise_sch_16b.h"

using namespace Ops::Base;
using namespace AscendC;

template <uint64_t dType>
__global__ __aicore__ void fill_v2(GM_ADDR dims, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FillV2TilingData);
    GET_TILING_DATA_PTR_WITH_STRUCT(FillV2TilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int64_t localValue = tilingData->value;
    if constexpr (dType == FILL_V2_TPL_FP16) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<half>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<half, 0>(*(half*)(&localValue));
        sch.Init(y);
        sch.Process();
    } else if constexpr (dType == FILL_V2_TPL_FP32) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<float>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<float, 0>(*(float*)(&localValue));
        sch.Init(y);
        sch.Process();
    } else if constexpr (dType == FILL_V2_TPL_DOUBLE) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<int64_t>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<int64_t, 0>(localValue);
        sch.Init(y);
        sch.Process();
    } else if constexpr (dType == FILL_V2_TPL_INT8) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<int8_t>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<int8_t, 0>(*(int8_t*)(&localValue));
        sch.Init(y);
        sch.Process();
    } else if constexpr (dType == FILL_V2_TPL_INT16) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<int16_t>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<int16_t, 0>(*(int16_t*)(&localValue));
        sch.Init(y);
        sch.Process();
    } else if constexpr (dType == FILL_V2_TPL_INT32) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<int32_t>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<int32_t, 0>(*(int32_t*)(&localValue));
        sch.Init(y);
        sch.Process();
    } else if constexpr (dType == FILL_V2_TPL_INT64) {
        ElementwiseSch16B<FILL_V2_TPL_SCH_MODE_1, FillV2Op::FillV2DAG<int64_t>::OpDag> sch(&(tilingData->baseTiling));
        sch.template SetVar<int64_t, 0>(localValue);
        sch.Init(y);
        sch.Process();
    }
    return;
}
