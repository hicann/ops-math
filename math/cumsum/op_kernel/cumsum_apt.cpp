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
 * \file cumsum_apt.cpp
 * \brief calculate the prefix accumulation sum of tensor specified dimension
 */
#include "arch35/cumsum_4_int.h"
#include "arch35/cumsum_core_ss_oneway_ss.h"
#include "arch35/cumsum_core_ss_twoway_ss.h"
#include "arch35/cumsum_core_ss_ub_ss_oneway_ss.h"
#include "arch35/cumsum_core_ss_ub_ss_twoway_ss.h"
#include "arch35/cumsum_oneway_ss.h"
#include "arch35/cumsum_twoway_ss.h"
#include "arch35/cumsum_ub_ss_oneway_ss.h"
#include "arch35/cumsum_ub_ss_twoway_ss.h"
#include "arch35/cumsum_base/cumsum_bi_twoway_ar.h"

using namespace AscendC;
using namespace Cumsum;
using namespace Cum;

#define CUMSUM_ONEWAY_SS_TILING_KEY 1001
#define CUMSUM_TWOWAY_SS_TILING_KEY 1002
#define CUMSUM_UB_SS_ONEWAY_SS_TILING_KEY 1011
#define CUMSUM_UB_SS_TWOWAY_SS_TILING_KEY 1012
#define CUMSUM_CORE_SS_ONEWAY_SS_TILING_KEY 1101
#define CUMSUM_CORE_SS_TWOWAY_SS_TILING_KEY 1102
#define CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_TILING_KEY 1111
#define CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_TILING_KEY 1112
// batch 一致性（deterministic_level==3，AR/lenN==1）：复用现有类，BI 语义由 tilingData 驱动
#define CUMSUM_BI_UB_SS_ONEWAY_SS_TILING_KEY 3011
#define CUMSUM_BI_UB_SS_TWOWAY_SS_TILING_KEY 3012
#define CUMSUM_BI_CORE_SS_TWOWAY_SS_TILING_KEY 3112
#define CUM_NO_SPLIT 11000
#define CUM_AR_SPLIT 11001
#define CUM_WITH_GROUP 11002

extern "C" __aicore__ inline void cumsumSimd(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    using PromoteType = __cumsumType::GetPromoteType<DTYPE_X>::T;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(CUMSUM_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumOnewaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumTwowaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }

    } else if (TILING_KEY_IS(CUMSUM_UB_SS_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumUbSsOnewaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_UB_SS_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumUbSsTwowaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_ONEWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsOnewaySs<DTYPE_X, PromoteType, CumsumOnewaySklansky<DTYPE_X, PromoteType>> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_TWOWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsTwowaySs<DTYPE_X, PromoteType, CumsumTwowaySklansky<DTYPE_X, PromoteType>> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsUbSsOnewaySs<DTYPE_X, PromoteType,
                                     CumsumUbSklansky<DTYPE_X, PromoteType, CumsumOnewaySklansky<DTYPE_X, PromoteType>>>
                op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsUbSsTwowaySs<DTYPE_X, PromoteType,
                                     CumsumUbSklansky<DTYPE_X, PromoteType, CumsumTwowaySklansky<DTYPE_X, PromoteType>>>
                op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_BI_UB_SS_ONEWAY_SS_TILING_KEY)) {
        // BI_UB_SS ONEWAY：小 R（foldCount=1）场景，复用 1011 类零改动
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumUbSsOnewaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_BI_UB_SS_TWOWAY_SS_TILING_KEY)) {
        // BI_UB_SS TWOWAY（3012 紧凑特化接管 N=1 场景）：R 不分核、行分核，
        // 全行 memo Fenwick 即规范树（附录 B 定理 1）；N=1 紧凑布局（连续段 + 经典
        // VEC 桥 + 层优先表驱动 VF 加法树），Ub 层(memo/BetweenUb/行分核)经
        // IsTwowayInner trait 整层复用。防御性 lenN 检查：非 1 回退通用类。
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            /* 紧凑接管全属性（lenN==1 恒走紧凑）。 */
            if (tilingData->lenN == 1) {
                Cumsum::CumsumBiUbSsTwowayArSs<DTYPE_X, PromoteType> op(pipe);
                op.Init(x, y, tilingData, workspace);
                op.Process();
            } else {
                CumsumUbSsTwowaySs<DTYPE_X, PromoteType> op(pipe);
                op.Init(x, y, tilingData, workspace);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(CUMSUM_BI_CORE_SS_TWOWAY_SS_TILING_KEY)) {
        // BI_CORE_SS：固定 chunk 分核（blockIdx = m*K + k，M·K<=coreNum 由路由保证），
        // chunk 内 memo + 核间多轮传播，两级 Fenwick ≡ 统一树（附录 B 定理 2），复用 1112 类
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                      std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_BI_CORE_SS_TWOWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsUbSsTwowaySs<DTYPE_X, PromoteType,
                                     CumsumUbSklansky<DTYPE_X, PromoteType, CumsumTwowaySklansky<DTYPE_X, PromoteType>>>
                op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    }
}

extern "C" __aicore__ inline void cumsumSimdInt(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(CUM_NO_SPLIT)) {
        GET_TILING_DATA_WITH_STRUCT(Cum4IntTilingData, tilingDataInt, tiling);
        if constexpr (std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
                      std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
                      std::is_same<DTYPE_X, uint64_t>::value) {
            CumNoSplit<DTYPE_X> op;
            op.Init(x, y, &tilingDataInt, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUM_AR_SPLIT)) {
        GET_TILING_DATA_WITH_STRUCT(Cum4IntTilingData, tilingDataInt, tiling);
        if constexpr (std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
                      std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
                      std::is_same<DTYPE_X, uint64_t>::value) {
            CumSplitAR<DTYPE_X> op;
            op.Init(x, y, &tilingDataInt, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUM_WITH_GROUP)) {
        GET_TILING_DATA_WITH_STRUCT(Cum4IntTilingData, tilingDataInt, tiling);
        KERNEL_TASK_TYPE(CUM_WITH_GROUP, KERNEL_TYPE_MIX_AIV_1_0);
        if constexpr (std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
                      std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
                      std::is_same<DTYPE_X, uint64_t>::value) {
            CumWithGroup<DTYPE_X> op;
            op.Init(x, y, &tilingDataInt, &pipe);
            op.Process();
        }
    }
}

extern "C" __global__ __aicore__ void cumsum(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
                  std::is_same<DTYPE_X, bfloat16_t>::value) {
        cumsumSimd(x, axis, y, workspace, tiling);
    } else if constexpr (std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
                         std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
                         std::is_same<DTYPE_X, uint64_t>::value) {
        cumsumSimdInt(x, axis, y, workspace, tiling);
    }
}
