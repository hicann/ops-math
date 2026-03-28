/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file reduce_nansum_arch32.cpp
 * \brief ReduceNansum Kernel 入口（arch32 架构）
 *
 * 模板参数说明（与 reduce_nansum_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X: 数据类型，由 ASCENDC_TPL_DATATYPE_DECL 定义
 *   - SCH_MODE: 调度模式（0=AR全载, 1=AR-ColSplit, 2=ARA全载, 3=ARA-RowSplit）
 */

#include "common/reduce_nansum.h"

template <typename D_T_X, int SCH_MODE>
__global__ __aicore__ void reduce_nansum(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReduceNansumTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceNansumTilingData, tilingData, tiling);

    if constexpr (SCH_MODE == 0) {
        // AR 全载
        NsReduceNansum::ReduceNansumArFullload<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (SCH_MODE == 1) {
        // AR ColSplit
        NsReduceNansum::ReduceNansumArColsplit<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (SCH_MODE == 2) {
        // ARA 全载
        NsReduceNansum::ReduceNansumAraFullload<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (SCH_MODE == 3) {
        // ARA RowSplit
        NsReduceNansum::ReduceNansumAraRowsplit<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
