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
 * \file drop_out_v3_infershape.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t DropOutV3_IN_X_IDX = 0;
static constexpr size_t DropOutV3_OUT_Y_IDX = 0;
static constexpr size_t DropOutV3_OUT_MASK_IDX = 1;

ge::graphStatus InferDataTypeForDropOutV3(gert::InferDataTypeContext* context)
{
    auto xType = context->GetInputDataType(DropOutV3_IN_X_IDX);
    context->SetOutputDataType(DropOutV3_OUT_Y_IDX, xType);
    context->SetOutputDataType(DropOutV3_OUT_MASK_IDX, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(DropOutV3).InferDataType(InferDataTypeForDropOutV3);
} // namespace ops