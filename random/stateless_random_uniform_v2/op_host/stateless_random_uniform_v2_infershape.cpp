/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file stateless_random_uniform_v2_infershape.cpp
 * \brief
 */

#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"
#include "util/const_util.h"

namespace {
constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kOutputIndex1 = 1U;
constexpr size_t kOutputIndex2 = 2U;
constexpr size_t kOutputIndex3 = 3U;
constexpr size_t kInputIndex0 = 0U;
constexpr size_t kInputIndex1 = 1U;
constexpr size_t kInputIndex2 = 2U;
constexpr size_t kInputIndex3 = 3U;
constexpr size_t kInputIndex4 = 4U;
constexpr size_t kInputIndex5 = 5U;
constexpr size_t kVectorDimNum = 1U;
constexpr size_t kMatrixDimNum = 2U;
constexpr size_t kCubeDimNum = 3U;
constexpr size_t kFirstDimIndex = 0U;
constexpr size_t kSecondDimIndex = 1U;
constexpr size_t kThirdDimIndex = 2U;
constexpr size_t kFourDimIndex = 3U;
constexpr size_t kFirstAttrIdx = 0U;
constexpr size_t kThirdAttrIdx = 2U;
constexpr int64_t kDimValue1 = 1LL;
constexpr int64_t kDimValue2 = 2LL;
constexpr int64_t kDimValue3 = 3LL;
constexpr int64_t kDimValue4 = 4LL;
constexpr int64_t kSeedValue = 2LL;
constexpr int64_t kCounterValue = 2LL;
} // namespace

using namespace ge;

namespace ops {

static graphStatus RandomOpCommonInferShapeByDependIndex(
    gert::InferShapeContext* context, const size_t depend_shape_index = 0)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    OP_LOGI(context->GetNodeName(), "Infer shape by depend index begin. index is %zu", depend_shape_index);
    const gert::Tensor* x_shape_tensor = context->GetInputTensor(depend_shape_index);
    gert::Shape* output_shape = context->GetOutputShape(kOutputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    gert::Shape const_shape;
    if (!Ops::Base::GetConstIntToShape(context, depend_shape_index, const_shape)) {
        OP_LOGI(context->GetNodeName(), "Get const of size unsuccessful, will do dynamic infershape.");
        const gert::Shape* x_shape = context->GetInputShape(depend_shape_index);
        OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
        if (x_shape->GetDim(kFirstDimIndex) >= 0LL) {
            OP_LOGI(context->GetNodeName(), "Set the output shape all -1.");
            Ops::Base::SetUnknownShape(x_shape->GetDim(kFirstDimIndex), *output_shape);
        } else {
            OP_LOGI(context->GetNodeName(), "Set the output shape = -2.");
            Ops::Base::SetUnknownRank(*output_shape);
        }
    }
    ge::DataType shape_dtype = x_shape_tensor->GetDataType();
    OP_CHECK_IF(
        (shape_dtype != DT_INT32 && shape_dtype != DT_INT64), OP_LOGI(context->GetNodeName(), "Shape type must in [int32, int64]. But get %s", Ops::Base::ToString(shape_dtype).c_str()),
        return GRAPH_FAILED);
    *output_shape = const_shape;
    OP_LOGI(context->GetNodeName(), "Infer shape by depend index end.");
    return GRAPH_SUCCESS;
}

static graphStatus StatelessTruncatedNormalV2InferShapeCheck(const gert::InferShapeContext* context)
{
    const gert::Shape* key_shape = context->GetInputShape(kInputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, key_shape);
    OP_CHECK_IF(
        (key_shape->GetDimNum() != kVectorDimNum || key_shape->GetDim(0U) != 1LL), OP_LOGI(context->GetNodeName(), "Input key shape must be 1D with 1 elements, but is[%u].", key_shape->GetDimNum()),
        return GRAPH_FAILED);
    const gert::Shape* counter_shape = context->GetInputShape(kInputIndex2);
    OP_CHECK_NULL_WITH_CONTEXT(context, counter_shape);
    OP_CHECK_IF(
        (counter_shape->GetDimNum() != kVectorDimNum || counter_shape->GetDim(0U) != kCounterValue), OP_LOGI(context->GetNodeName(), "Input counter shape must be 1D with 2 elements, but is[%uD].", counter_shape->GetDimNum()),
        return GRAPH_FAILED);
    const gert::Shape* alg_shape = context->GetInputShape(kInputIndex3);
    OP_CHECK_NULL_WITH_CONTEXT(context, alg_shape);

    OP_CHECK_IF(
        (!alg_shape->IsScalar()), OP_LOGI(context->GetNodeName(), "Input alg must be scalar, but is[%uD].", alg_shape->GetDimNum()),
        return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus StatelessTruncatedNormalV2InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    if (StatelessTruncatedNormalV2InferShapeCheck(context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return RandomOpCommonInferShapeByDependIndex(context);
}

IMPL_OP_INFERSHAPE(StatelessRandomUniformV2)
    .InputsDataDependency({kInputIndex0})
    .InferShape(StatelessTruncatedNormalV2InferShape);

} // namespace ops