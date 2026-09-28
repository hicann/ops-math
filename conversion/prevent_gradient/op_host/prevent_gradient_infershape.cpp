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
 * \file prevent_gradient_infershape.cpp
 * \brief Runtime shape and data-type inference for PreventGradient.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "log/log.h"

namespace ops {
const std::string OP_NAME = "PreventGradient";
static ge::graphStatus InferShapeForPreventGradient(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE(OP_NAME, "InferShape context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (context->GetComputeNodeInputNum() != 1U || context->GetComputeNodeOutputNum() != 1U) {
        OP_LOGE(OP_NAME, "Only support 1 input and 1 output, but got input num %zu, output num %zu.",
                context->GetComputeNodeInputNum(), context->GetComputeNodeOutputNum());
        return ge::GRAPH_FAILED;
    }
    const auto* input_shape = context->GetInputShape(0U);
    auto* output_shape = context->GetOutputShape(0U);
    if (input_shape == nullptr || output_shape == nullptr) {
        OP_LOGE(OP_NAME, "Input shape or output shape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForPreventGradient(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE(OP_NAME, "InferDataType context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(0U, context->GetInputDataType(0U));
}

IMPL_OP_INFERSHAPE(PreventGradient)
    .InferShape(InferShapeForPreventGradient)
    .InferDataType(InferDataTypeForPreventGradient);
} // namespace ops
