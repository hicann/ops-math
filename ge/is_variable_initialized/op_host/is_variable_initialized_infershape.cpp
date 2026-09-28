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
 * \file is_variable_initialized_infershape.cpp
 * \brief InferShape and InferDataType implementation for IsVariableInitialized operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4IsVariableInitialized(gert::InferShapeContext* context)
{
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4IsVariableInitialized(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, ge::DT_BOOL);
}

IMPL_OP_INFERSHAPE(IsVariableInitialized)
    .InferShape(InferShape4IsVariableInitialized)
    .InferDataType(InferDataType4IsVariableInitialized);

} // namespace ops
