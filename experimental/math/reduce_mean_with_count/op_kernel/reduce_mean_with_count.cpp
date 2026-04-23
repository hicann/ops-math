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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file reduce_mean_with_count.cpp
 * @brief ReduceMeanWithCount kernel entry (arch35)
 *
 * Template parameters (corresponding to ASCENDC_TPL_ARGS_DECL in tiling_key.h):
 *   - D_T_X: data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - REDUCE_MODE: reduce mode (0=AR full-load, 1=AR col-split, 2=ARA full-load)
 */

#include "reduce_mean_with_count.h"

template <typename D_T_X, int REDUCE_MODE>
__global__ __aicore__ void reduce_mean_with_count(GM_ADDR input, GM_ADDR meanResult,
                                                   GM_ADDR countResult, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    ENABLE_PRINTF();
    REGISTER_TILING_DEFAULT(ReduceMeanWithCountTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceMeanWithCountTilingData, tilingData, tiling);
    NsReduceMeanWithCount::ReduceMeanWithCount<D_T_X, REDUCE_MODE> op;
    op.Init(input, meanResult, countResult, &tilingData);
    op.Process();
}
