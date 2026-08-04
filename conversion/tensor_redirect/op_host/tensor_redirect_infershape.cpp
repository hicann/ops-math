/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_redirect_infershape.cpp
 * \brief
 */
#include "infershape_elewise_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t TENSOR_REDIRECT_IN_X_IDX = 0;
static constexpr size_t TENSOR_REDIRECT_OUT_Y_IDX = 0;

static ge::graphStatus InferDataType4TensorRedirect(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(TENSOR_REDIRECT_OUT_Y_IDX, context->GetInputDataType(TENSOR_REDIRECT_IN_X_IDX));
    return ge::GRAPH_SUCCESS;
}

// output_x 与 x 同 shape/dtype（same-as-input）
IMPL_OP_INFERSHAPE(TensorRedirect)
    .InferShape(Ops::Base::InferShape4Elewise)
    .InferDataType(InferDataType4TensorRedirect);
} // namespace ops
