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
 * \file const_infershape.cpp
 * \brief InferShape and InferDataType implementation for Const operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Const(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "get attrs failed");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor* value_tensor = attrs->GetTensor(0);
    if (value_tensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "get value attr failed");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "get output shape failed");
        return ge::GRAPH_FAILED;
    }

    *outputShape = value_tensor->GetOriginShape();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Const(gert::InferDataTypeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor* value_tensor = attrs->GetTensor(0);
    if (value_tensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    return context->SetOutputDataType(0, value_tensor->GetDataType());
}

IMPL_OP_INFERSHAPE(Const).InferShape(InferShape4Const).InferDataType(InferDataType4Const);

} // namespace ops
