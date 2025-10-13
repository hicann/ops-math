/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_d.cpp
 * \brief
 */
#include "lin_space_d.h"

using namespace LinSpaceD;

// kernel function
extern "C" __global__ __aicore__ void lin_space_d(GM_ADDR start, GM_ADDR end, GM_ADDR num, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }
    
    REGISTER_TILING_DEFAULT(LinSpaceDTilingDataa);
    GET_TILING_DATA_WITH_STRUCT(LinSpaceDTilingData, tilingData, tiling);
    if (
        (DTYPE_START == ge::DT_UINT8 || DTYPE_START == ge::DT_BF16 ||
         DTYPE_START == ge::DT_FLOAT16 || DTYPE_START == ge::DT_FLOAT || 
         DTYPE_START == ge::DT_INT8 || DTYPE_START == ge::DT_INT16 || DTYPE_START == ge::DT_INT32) &&
        (DTYPE_END == ge::DT_UINT8  || DTYPE_END == ge::DT_BF16 ||
         DTYPE_END == ge::DT_FLOAT16 || DTYPE_END == ge::DT_FLOAT || 
         DTYPE_END == ge::DT_INT8 || DTYPE_END == ge::DT_INT16 || DTYPE_END == ge::DT_INT32)
    ) {
        KernelLinSpaceD<DTYPE_START, DTYPE_END> op;
        op.Init(start, end, z, &tilingData);
        op.Process();
    }
}