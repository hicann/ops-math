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
 * \file population_count.cpp
 * \brief PopulationCount kernel entry (arch35)
 *
 * Template params match ASCENDC_TPL_ARGS_DECL in population_count_tiling_key.h:
 *   - D_T_X:       input dtype
 *   - BUFFER_MODE: 0 single / 1 double
 */

#include "population_count.h"

template <typename D_T_X, uint32_t BUFFER_MODE>
__global__ __aicore__ void population_count(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // Enable in-kernel printf for debug (no runtime overhead when no printf is invoked).
    ENABLE_PRINTF();

    REGISTER_TILING_DEFAULT(PopulationCountTilingData);
    GET_TILING_DATA_WITH_STRUCT(PopulationCountTilingData, tilingData, tiling);

    // Empty tensor guard (defense-in-depth; L2 already short-circuits)
    if (tilingData.totalNum == 0) {
        return;
    }

    NsPopulationCount::PopulationCount<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
