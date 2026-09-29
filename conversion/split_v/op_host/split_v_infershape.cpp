/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"
#include "op_api/op_util.h"
#include "util/const_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static constexpr int32_t SPLIT_V_IDX_IN_X = 0;
static constexpr int32_t SPLIT_V_IDX_IN_SIZE_SPLITS = 1;
static constexpr int32_t SPLIT_V_IDX_IN_SPLIT_DIM = 2;
static constexpr int32_t SPLIT_V_ATTR_NUM_SPLIT = 0;

static graphStatus UpdateDynamicShape(gert::InferShapeContext* context, const gert::Shape* x_shape,
                                      const int64_t num_split)
{
    for (int64_t i = 0; i < num_split; i++) {
        gert::Shape* out_shape_dynamic = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_dynamic);
        *out_shape_dynamic = *x_shape;
    }
    return GRAPH_SUCCESS;
}

static graphStatus UpdatetAllUnknownDim(gert::InferShapeContext* context, const int64_t num_split, const int64_t rank)
{
    for (int64_t i = 0; i < num_split; i++) {
        gert::Shape* out_shape_dynamic = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_dynamic);
        SetUnknownShape(rank, *out_shape_dynamic);
    }
    return GRAPH_SUCCESS;
}

static graphStatus UpdateSplitDimUnknown(gert::InferShapeContext* context, const gert::Shape* x_shape,
                                         const int64_t num_split, const int64_t split_dim)
{
    for (int64_t i = 0; i < num_split; i++) {
        gert::Shape* out_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
        *out_shape = *x_shape;
        out_shape->SetDim(split_dim, -1);
    }
    return GRAPH_SUCCESS;
}

template <typename T>
graphStatus CalcSplitVOut(gert::InferShapeContext* context, const gert::Tensor* size_splits_vec)
{
    const gert::Shape* x_shape = context->GetInputShape(SPLIT_V_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* numSplit = attrs->GetAttrPointer<int64_t>(SPLIT_V_ATTR_NUM_SPLIT);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplit);
    int64_t num_split = *numSplit;

    OP_CHECK_IF(num_split <= 0,
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "num_split", std::to_string(num_split).c_str(),
                                          "greater than 0"),
                return GRAPH_FAILED);

    OP_CHECK_IF(IsUnknownRank(*x_shape),
                OP_LOGD(context->GetNodeName(), "input x is unknown rank, will set all output the same as input."),
                return UpdateDynamicShape(context, x_shape, num_split));

    int64_t split_dim = 0;
    if (!GetConstInt(context, SPLIT_V_IDX_IN_SPLIT_DIM, split_dim)) {
        OP_LOGD(context->GetNodeName(), "get split_dim unsuccessful, will set output to -1.");
        const int64_t input_rank = x_shape->GetDimNum();
        return UpdatetAllUnknownDim(context, num_split, input_rank);
    }

    int64_t x_dim_num = x_shape->GetDimNum();
    OP_CHECK_IF(!IsDimValid(x_dim_num, split_dim),
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "split_dim", std::to_string(split_dim).c_str(),
                                          ConcatString("[-", x_dim_num, ", ", x_dim_num, ")").c_str()),
                return GRAPH_FAILED);
    split_dim = split_dim < 0 ? split_dim + x_dim_num : split_dim;

    // size_splits is not const: only the dim on split_dim is unknown
    const T* split_size_value = (size_splits_vec == nullptr) ? nullptr : size_splits_vec->GetData<T>();
    if (split_size_value == nullptr) {
        OP_LOGD(context->GetNodeName(), "get size_splits value unsuccessful, will set dim on split_dim to -1.");
        return UpdateSplitDimUnknown(context, x_shape, num_split, split_dim);
    }

    int64_t size_splits_size = static_cast<int64_t>(size_splits_vec->GetShapeSize());
    OP_CHECK_IF(size_splits_size != num_split,
                OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(context->GetNodeName(), SPLIT_V_IDX_IN_SIZE_SPLITS,
                                                     std::to_string(size_splits_size).c_str(),
                                                     std::to_string(num_split).c_str()),
                return GRAPH_FAILED);

    int64_t dynamic_value_idx = -1;
    int64_t split_size_value_sum = 0;
    int64_t dynamic_value_num = 0;

    for (int64_t i = 0; i < size_splits_size; i++) {
        if (split_size_value[i] == -1) {
            dynamic_value_num = dynamic_value_num + 1;
            OP_CHECK_IF(dynamic_value_num > 1,
                        OP_LOGE(context->GetNodeName(), "value of split_size can only have one -1"),
                        return GRAPH_FAILED);

            dynamic_value_idx = i;
        } else {
            split_size_value_sum = split_size_value_sum + split_size_value[i];
        }
    }

    // update dynamic output
    for (int64_t i = 0; i < num_split; i++) {
        gert::Shape* out_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
        *out_shape = *x_shape;
        out_shape->SetDim(split_dim, split_size_value[i]);
    }

    if (dynamic_value_idx != -1) {
        gert::Shape* out_shape = context->GetOutputShape(dynamic_value_idx);
        if (x_shape->GetDim(split_dim) == -1) {
            out_shape->SetDim(split_dim, -1);
        } else {
            out_shape->SetDim(split_dim, x_shape->GetDim(split_dim) - split_size_value_sum);
        }
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferShape4SplitV(gert::InferShapeContext* context)
{
    const gert::Tensor* size_splits_desc = context->GetInputTensor(SPLIT_V_IDX_IN_SIZE_SPLITS);
    if (size_splits_desc == nullptr) {
        // size_splits is not const, degrade the dim on split_dim to -1
        OP_CHECK_IF(
            CalcSplitVOut<int64_t>(context, nullptr) == GRAPH_FAILED,
            OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v (size_splits not const)"),
            return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    DataType size_splits_dtype = size_splits_desc->GetDataType();
    if (size_splits_dtype == DT_INT32) {
        OP_CHECK_IF(CalcSplitVOut<int32_t>(context, size_splits_desc) == GRAPH_FAILED,
                    OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v (int32)"),
                    return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(CalcSplitVOut<int64_t>(context, size_splits_desc) == GRAPH_FAILED,
                    OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v (int64)"),
                    return GRAPH_FAILED);
    }

    return GRAPH_SUCCESS;
}
static graphStatus InferDataType4SplitV(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4SplitV start");
    auto input_x_dtype = context->GetInputDataType(SPLIT_V_IDX_IN_X);
    const auto output_num = context->GetComputeNodeOutputNum();
    for (size_t i = 0; i < output_num; i++) {
        context->SetOutputDataType(i, input_x_dtype);
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4SplitV end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SplitV)
    .InferShape(InferShape4SplitV)
    .InferDataType(InferDataType4SplitV)
    .InputsDataDependency({SPLIT_V_IDX_IN_SIZE_SPLITS, SPLIT_V_IDX_IN_SPLIT_DIM});

} // namespace ops
