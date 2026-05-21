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
 * \file reshape_infershape.cpp
 * \brief
 */

#include <limits>
#include <vector>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputX = 0U;
constexpr size_t kInputShape = 1U;
constexpr size_t kOutputY = 0U;
constexpr size_t kAttrAxis = 0U;
constexpr size_t kAttrNumAxes = 1U;
constexpr size_t kAttrAllowZero = 2U;
constexpr size_t kNoInferIndex = std::numeric_limits<size_t>::max();
constexpr int64_t kDefaultAxis = 0;
constexpr int64_t kDefaultNumAxes = -1;
constexpr int64_t kDefaultAllowZero = 0;

struct ReshapeAttrs {
    int64_t axis = kDefaultAxis;
    int64_t num_axes = kDefaultNumAxes;
    int64_t allow_zero = kDefaultAllowZero;
};

struct ReshapeAxisRange {
    int64_t start_axis = 0;
    int64_t end_axis = 0;
};

int64_t GetIntAttrOrDefault(const gert::RuntimeAttrs* attrs, size_t index, int64_t default_value)
{
    if (attrs == nullptr || attrs->GetAttrNum() <= index) {
        return default_value;
    }
    const int64_t* value = attrs->GetInt(index);
    return value == nullptr ? default_value : *value;
}

bool SafeMultiply(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }
    if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

ReshapeAttrs GetReshapeAttrs(const gert::RuntimeAttrs* attrs)
{
    ReshapeAttrs reshape_attrs;
    reshape_attrs.axis = GetIntAttrOrDefault(attrs, kAttrAxis, kDefaultAxis);
    reshape_attrs.num_axes = GetIntAttrOrDefault(attrs, kAttrNumAxes, kDefaultNumAxes);
    reshape_attrs.allow_zero = GetIntAttrOrDefault(attrs, kAttrAllowZero, kDefaultAllowZero);
    return reshape_attrs;
}

void CollectInputDims(const gert::Shape* x_shape, std::vector<int64_t>& input_dims)
{
    input_dims.clear();
    input_dims.reserve(x_shape->GetDimNum());
    for (size_t idx = 0U; idx < x_shape->GetDimNum(); ++idx) {
        input_dims.push_back(x_shape->GetDim(idx));
    }
}

bool TryGetElementCount(const std::vector<int64_t>& dims, int64_t& count)
{
    count = 1;
    for (const auto dim : dims) {
        if (dim < 0) {
            return false;
        }
        if (!SafeMultiply(count, dim, count)) {
            return false;
        }
    }
    return true;
}

template <typename T>
ge::graphStatus ReadRequestedDims(
    const gert::InferShapeContext* context, const gert::Tensor* shape_tensor, std::vector<int64_t>& requested_dims)
{
    const auto reshape_rank = static_cast<size_t>(shape_tensor->GetShapeSize());
    const T* shape_data = shape_tensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_data);
    requested_dims.reserve(reshape_rank);
    for (size_t idx = 0U; idx < reshape_rank; ++idx) {
        requested_dims.push_back(static_cast<int64_t>(shape_data[idx]));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetRequestedDims(
    const gert::InferShapeContext* context, const gert::Tensor* shape_tensor, std::vector<int64_t>& requested_dims)
{
    requested_dims.clear();
    switch (shape_tensor->GetDataType()) {
        case ge::DT_INT32:
            return ReadRequestedDims<int32_t>(context, shape_tensor, requested_dims);
        case ge::DT_INT64:
            return ReadRequestedDims<int64_t>(context, shape_tensor, requested_dims);
        default:
            OP_LOGE(
                context->GetNodeName(), "shape input dtype must be int32 or int64, but got %d.",
                shape_tensor->GetDataType());
            return ge::GRAPH_FAILED;
    }
}

ge::graphStatus ResolveAxisRange(
    const gert::InferShapeContext* context, int64_t input_rank, const ReshapeAttrs& attrs, ReshapeAxisRange& axis_range)
{
    axis_range.start_axis = attrs.axis >= 0 ? attrs.axis : attrs.axis + input_rank + 1;
    if (axis_range.start_axis < 0 || axis_range.start_axis > input_rank) {
        OP_LOGE(
            context->GetNodeName(), "axis %ld is out of range [%ld, %ld].", attrs.axis, -1 - input_rank,
            input_rank);
        return ge::GRAPH_FAILED;
    }
    if (attrs.num_axes < -1) {
        OP_LOGE(context->GetNodeName(), "num_axes %ld must be greater than or equal to -1.", attrs.num_axes);
        return ge::GRAPH_FAILED;
    }

    axis_range.end_axis = attrs.num_axes == -1 ? input_rank : axis_range.start_axis + attrs.num_axes;
    if (axis_range.end_axis > input_rank) {
        OP_LOGE(
            context->GetNodeName(), "num_axes %ld is invalid for axis %ld and rank %ld.", attrs.num_axes,
            attrs.axis, input_rank);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void InitOutputDims(
    const std::vector<int64_t>& input_dims, const std::vector<int64_t>& requested_dims,
    const ReshapeAxisRange& axis_range, std::vector<int64_t>& output_dims)
{
    output_dims.clear();
    output_dims.reserve(
        static_cast<size_t>(axis_range.start_axis) + requested_dims.size() +
        static_cast<size_t>(static_cast<int64_t>(input_dims.size()) - axis_range.end_axis));
    for (int64_t idx = 0; idx < axis_range.start_axis; ++idx) {
        output_dims.push_back(input_dims[static_cast<size_t>(idx)]);
    }
}

ge::graphStatus NormalizeRequestedDim(
    const gert::InferShapeContext* context, const std::vector<int64_t>& input_dims, const ReshapeAxisRange& axis_range,
    int64_t allow_zero, size_t requested_index, int64_t& dim)
{
    if (dim < -1) {
        OP_LOGE(context->GetNodeName(), "shape dim %zu must be -1 or non-negative, but got %ld.", requested_index, dim);
        return ge::GRAPH_FAILED;
    }
    if (dim == 0 && allow_zero == 0) {
        const int64_t copy_index = axis_range.start_axis + static_cast<int64_t>(requested_index);
        if (copy_index < 0 || copy_index >= static_cast<int64_t>(input_dims.size())) {
            OP_LOGE(context->GetNodeName(), "shape dim %zu cannot copy from input axis %ld.", requested_index, copy_index);
            return ge::GRAPH_FAILED;
        }
        dim = input_dims[static_cast<size_t>(copy_index)];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AppendRequestedDims(
    const gert::InferShapeContext* context, const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& requested_dims, const ReshapeAxisRange& axis_range, int64_t allow_zero,
    std::vector<int64_t>& output_dims, size_t& infer_index)
{
    infer_index = kNoInferIndex;
    for (size_t idx = 0U; idx < requested_dims.size(); ++idx) {
        int64_t dim = requested_dims[idx];
        if (dim == -1) {
            if (infer_index != kNoInferIndex) {
                OP_LOGE(context->GetNodeName(), "Only one dim in shape may be -1.");
                return ge::GRAPH_FAILED;
            }
            infer_index = output_dims.size();
            output_dims.push_back(1);
            continue;
        }
        if (NormalizeRequestedDim(context, input_dims, axis_range, allow_zero, idx, dim) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        output_dims.push_back(dim);
    }
    return ge::GRAPH_SUCCESS;
}

void AppendTrailingDims(
    const std::vector<int64_t>& input_dims, const ReshapeAxisRange& axis_range, std::vector<int64_t>& output_dims)
{
    for (int64_t idx = axis_range.end_axis; idx < static_cast<int64_t>(input_dims.size()); ++idx) {
        output_dims.push_back(input_dims[static_cast<size_t>(idx)]);
    }
}

ge::graphStatus ResolveInferDim(
    const gert::InferShapeContext* context, int64_t input_count, std::vector<int64_t>& output_dims, size_t infer_index)
{
    std::vector<int64_t> known_output_dims = output_dims;
    known_output_dims[infer_index] = 1;

    int64_t known_output_count = 0;
    const bool known_output_count_ready = TryGetElementCount(known_output_dims, known_output_count);
    if (!known_output_count_ready) {
        output_dims[infer_index] = ge::UNKNOWN_DIM;
        return ge::GRAPH_SUCCESS;
    }
    if (input_count == 0) {
        output_dims[infer_index] = 0;
        return ge::GRAPH_SUCCESS;
    }
    if (known_output_count == 0 || input_count % known_output_count != 0) {
        OP_LOGE(
            context->GetNodeName(), "input shape size %ld cannot be divided by %ld.", input_count,
            known_output_count);
        return ge::GRAPH_FAILED;
    }

    output_dims[infer_index] = input_count / known_output_count;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateOutputCount(
    const gert::InferShapeContext* context, int64_t input_count, const std::vector<int64_t>& output_dims)
{
    int64_t output_count = 0;
    if (TryGetElementCount(output_dims, output_count) && output_count != input_count) {
        OP_LOGE(
            context->GetNodeName(), "output shape size %ld is not equal to input shape size %ld.", output_count,
            input_count);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FinalizeOutputDims(
    const gert::InferShapeContext* context, const std::vector<int64_t>& input_dims, std::vector<int64_t>& output_dims,
    size_t infer_index)
{
    int64_t input_count = 0;
    const bool input_count_known = TryGetElementCount(input_dims, input_count);
    if (!input_count_known) {
        if (infer_index != kNoInferIndex) {
            output_dims[infer_index] = ge::UNKNOWN_DIM;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (infer_index != kNoInferIndex) {
        return ResolveInferDim(context, input_count, output_dims, infer_index);
    }
    return ValidateOutputCount(context, input_count, output_dims);
}

void WriteOutputDims(gert::Shape* y_shape, const std::vector<int64_t>& output_dims)
{
    y_shape->SetDimNum(output_dims.size());
    for (size_t idx = 0U; idx < output_dims.size(); ++idx) {
        y_shape->SetDim(idx, output_dims[idx]);
    }
}

ge::graphStatus InferShapeForReshape(gert::InferShapeContext* context)
{
    const auto* x_shape = context->GetInputShape(kInputX);
    const auto* shape_tensor = context->GetInputTensor(kInputShape);
    auto* y_shape = context->GetOutputShape(kOutputY);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    std::vector<int64_t> input_dims;
    CollectInputDims(x_shape, input_dims);

    std::vector<int64_t> requested_dims;
    if (GetRequestedDims(context, shape_tensor, requested_dims) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const ReshapeAttrs attrs = GetReshapeAttrs(context->GetAttrs());
    ReshapeAxisRange axis_range;
    if (ResolveAxisRange(context, static_cast<int64_t>(input_dims.size()), attrs, axis_range) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> output_dims;
    InitOutputDims(input_dims, requested_dims, axis_range, output_dims);

    size_t infer_index = kNoInferIndex;
    if (AppendRequestedDims(
            context, input_dims, requested_dims, axis_range, attrs.allow_zero, output_dims, infer_index) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    AppendTrailingDims(input_dims, axis_range, output_dims);

    if (FinalizeOutputDims(context, input_dims, output_dims, infer_index) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    WriteOutputDims(y_shape, output_dims);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static int64_t g_allow_zero = kDefaultAllowZero;
IMPL_OP_INFERSHAPE(Reshape)
    .InferShape(InferShapeForReshape)
    .InputsDataDependency({kInputShape})
    .PrivateAttr("allowzero", g_allow_zero);
} // namespace ops