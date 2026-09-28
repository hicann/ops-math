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
 * \file constant_infershape.cpp
 * \brief InferShape and InferDataType implementation for Constant operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Constant(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto valueTensor = attrs->GetTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueTensor);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = valueTensor->GetOriginShape();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Constant(gert::InferDataTypeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto valueTensor = attrs->GetTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueTensor);
    return context->SetOutputDataType(0, valueTensor->GetDataType());
}

IMPL_OP_INFERSHAPE(Constant).InferShape(InferShape4Constant).InferDataType(InferDataType4Constant);

} // namespace ops
