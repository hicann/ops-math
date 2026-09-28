/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file var_is_initialized_op_infershape.cpp
 * \brief InferShape and InferDataType implementation for VarIsInitializedOp operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

// RT1.0 中取输入 desc 后改写为标量 DT_BOOL 传给输出（输出与输入 shape/dtype 无关，
// 恒为 bool 标量）；RT2.0 直接写输出通道，无需读取输入
static ge::graphStatus InferShape4VarIsInitializedOp(gert::InferShapeContext* context)
{
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4VarIsInitializedOp(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, ge::DT_BOOL);
}

IMPL_OP_INFERSHAPE(VarIsInitializedOp)
    .InferShape(InferShape4VarIsInitializedOp)
    .InferDataType(InferDataType4VarIsInitializedOp);

} // namespace ops
