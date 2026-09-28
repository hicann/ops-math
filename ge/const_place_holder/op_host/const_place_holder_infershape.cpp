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
 * \file const_place_holder_infershape.cpp
 * \brief InferShape and InferDataType implementation for ConstPlaceHolder operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t kAttrShapeIndex = 0U;
constexpr size_t kAttrDtypeIndex = 5U;
} // namespace

namespace ops {

static ge::graphStatus InferShape4ConstPlaceHolder(gert::InferShapeContext* context)
{
    auto outShape = context->GetOutputShape(0U);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::TypedContinuousVector<int64_t>* shapeAttr = attrs->GetListInt(kAttrShapeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeAttr);
    outShape->SetDimNum(shapeAttr->GetSize());
    for (size_t i = 0U; i < shapeAttr->GetSize(); ++i) {
        outShape->SetDim(i, shapeAttr->GetData()[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ConstPlaceHolder(gert::InferDataTypeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dtype = attrs->GetAttrPointer<ge::DataType>(kAttrDtypeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtype);
    return context->SetOutputDataType(0U, *dtype);
}

IMPL_OP_INFERSHAPE(ConstPlaceHolder)
    .InferShape(InferShape4ConstPlaceHolder)
    .InferDataType(InferDataType4ConstPlaceHolder);

} // namespace ops
