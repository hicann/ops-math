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

/*!
 * \file asin_grad.cpp
 * \brief AsinGrad Kernel entry (arch35)
 *
 * Template parameters (matching asin_grad_tiling_key.h ASCENDC_TPL_ARGS_DECL):
 *   - D_T: Data type, defined by ASCENDC_TPL_DATATYPE_DECL
 *   - BUFFER_MODE: Buffer mode (0=single, 1=double), defined by ASCENDC_TPL_UINT_DECL
 */

#include "asin_grad.h"

template <typename D_T, int BUFFER_MODE>
__global__ __aicore__ void asin_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    ENABLE_PRINTF();
    REGISTER_TILING_DEFAULT(AsinGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinGradTilingData, tilingData, tiling);

    if constexpr (std::is_same_v<D_T, bfloat16_t>) {
        NsAsinGrad::AsinGrad<D_T, float, BUFFER_MODE> op;
        op.Init(dy, x, dx, &tilingData);
        op.Process();
    } else {
        NsAsinGrad::AsinGrad<D_T, D_T, BUFFER_MODE> op;
        op.Init(dy, x, dx, &tilingData);
        op.Process();
    }
}
