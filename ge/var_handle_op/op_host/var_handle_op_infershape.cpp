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
 * \file var_handle_op_infershape.cpp
 * \brief InferShape and InferDataType implementation for VarHandleOp operator.
 */
#include <vector>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/ct_infer_shape_context.h"
#include "graph/inference_context.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t kAttrDtypeIndex = 2U;
constexpr size_t kAttrShapeIndex = 3U;
} // namespace

namespace ops {

// 输出为标量 DT_RESOURCE，shape/dtype 属性描述 handle 所指向变量的形状与类型，
// 经 InferenceContext::SetOutputHandleShapesAndTypes 传递给下游（框架 peer 传播机制，
// 供 Assign/ReadVariableOp 类算子经 GetInputHandleShapesAndTypes 读取）。
// 本函数仅在编译期调用：CtInferShapeContext 按 CT 布局读取 inputs_num+1 槽的
// InferenceContext（exe runtime 布局该槽位为 InferShapeFunc，不适用于本算子）。
static ge::graphStatus InferShape4VarHandleOp(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // dtype 为必填属性：缺失时推导失败（与 RT1.0 GetAttr 失败返回 GRAPH_FAILED 一致）
    auto dtype = attrs->GetAttrPointer<ge::DataType>(kAttrDtypeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtype);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(0U);

    auto* inferContext = static_cast<gert::CtInferShapeContext*>(context)->GetInferenceContext();
    if (inferContext != nullptr) {
        // shape 属性缺省值为 UNKNOWN_SHAPE({-1})，未设置时按未知 shape 传递
        std::vector<int64_t> dims = ge::UNKNOWN_SHAPE;
        const auto* shapeAttr = attrs->GetListInt(kAttrShapeIndex);
        if ((shapeAttr != nullptr) && (shapeAttr->GetSize() > 0U)) {
            dims.assign(shapeAttr->GetData(), shapeAttr->GetData() + shapeAttr->GetSize());
        }
        std::vector<ge::ShapeAndType> handleShapesAndTypes;
        handleShapesAndTypes.reserve(1U);
        handleShapesAndTypes.emplace_back(ge::Shape(dims), *dtype);
        std::vector<std::vector<ge::ShapeAndType>> shapesAndTypes(2U);
        shapesAndTypes[0] = handleShapesAndTypes;
        inferContext->SetOutputHandleShapesAndTypes(shapesAndTypes);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4VarHandleOp(gert::InferDataTypeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dtype = attrs->GetAttrPointer<ge::DataType>(kAttrDtypeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtype);
    return context->SetOutputDataType(0, ge::DT_RESOURCE);
}

IMPL_OP_INFERSHAPE(VarHandleOp).InferShape(InferShape4VarHandleOp).InferDataType(InferDataType4VarHandleOp);

} // namespace ops
