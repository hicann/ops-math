/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file round.cpp
 * \brief 
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/round_dag.h"
#include "arch35/round_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/round_tiling_struct.h"

using namespace AscendC;
using namespace RoundOp;

template <uint64_t schMode, uint64_t dType, typename DtypeX>
__global__ __aicore__ void RoundKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr bool notIntDtype = (std::is_same<DtypeX, half>::value) ||
                                 (std::is_same<DtypeX, float>::value) ||
                                 (std::is_same<DtypeX, bfloat16_t>::value);
    constexpr bool roundZero = (dType == static_cast<uint64_t>(ROUND_TPL_ZERO)) && (notIntDtype);
    constexpr bool roundPositive = (dType == static_cast<uint64_t>(ROUND_TPL_POSITIVE_DECIMALS)) && (notIntDtype);
    constexpr bool roundNegative = (dType == static_cast<uint64_t>(ROUND_TPL_NEGATIVE_DECIMALS)) && (notIntDtype);
    constexpr bool roundNan = (dType == static_cast<uint64_t>(ROUND_TPL_NAN_DECIMALS)) && (notIntDtype);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RoundTilingData);
    GET_TILING_DATA_WITH_STRUCT(RoundTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (dType == static_cast<uint64_t>(ROUND_TPL_INT32)) {
        ElementwiseSch<schMode, typename RoundDag::RoundInt<int>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (roundZero) {
        ElementwiseSch<schMode, typename RoundDag::RoundZero<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (roundPositive) {
        ElementwiseSch<schMode, typename RoundDag::RoundPositiveDecimals<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.decimals);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (roundNegative) {
        ElementwiseSch<schMode, typename RoundDag::RoundNegativeDecimals<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.decimals);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (roundNan) {
        ElementwiseSch<schMode, typename RoundDag::RoundNan<DtypeX>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(y);
        sch.Process();
    }
    return;
}

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void round(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    return RoundKernel<schMode, dType, DTYPE_X>(x, y, workspace, tiling);
}