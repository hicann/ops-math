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
#include "util/math_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
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

static graphStatus CheckSplitParams(const gert::InferShapeContext* context, const gert::Shape*& x_shape,
                                    int64_t split_dim, const int64_t num_split)
{
    OP_CHECK_IF(num_split <= 0,
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("num_split must be greater than 0, but it's ", num_split).c_str()),
                return GRAPH_FAILED);

    int64_t x_shape_dim = x_shape->GetDimNum();

    OP_CHECK_IF(!IsDimValid(x_shape_dim, split_dim),
                OP_LOGE(context->GetNodeName(), "%s", GenInvalidDimMsg("split_dim", x_shape_dim, split_dim).c_str()),
                return GRAPH_FAILED);

    if (split_dim < 0) {
        split_dim += x_shape_dim;
    }

    OP_CHECK_IF((x_shape->GetDim(split_dim) % num_split != 0) && (x_shape->GetDim(split_dim) != -1),
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("the split_dim dimension of x_shape must be divisible by num_split.",
                                     " x_shape is ", ToString(*x_shape), ", x_shape on split_dim is ",
                                     x_shape->GetDim(split_dim), ", num_split is ", num_split)
                            .c_str()),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

static void CalOutShape(const gert::Shape*& x_shape, gert::Shape*& out_shape, const int64_t num_split,
                        const int64_t split_dim)
{
    int64_t output_dim_size = x_shape->GetDim(split_dim) / num_split;
    *out_shape = *x_shape;
    out_shape->SetDim(split_dim, output_dim_size);
}

static graphStatus InferShape4Split(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numSplit = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplit);
    int64_t num_split = *numSplit;

    OP_CHECK_IF(IsUnknownRank(*x_shape),
                OP_LOGD(context->GetNodeName(), "input x is unknown rank, will set all output the same as input."),
                return UpdateDynamicShape(context, x_shape, num_split));

    int64_t split_dim = 0;
    if (!GetConstInt(context, 0, split_dim)) {
        OP_LOGD(context->GetNodeName(), "get split_dim unsuccessful, will set output to -1.");
        if (num_split > 1) {
            const int64_t input_rank = x_shape->GetDimNum();
            return UpdatetAllUnknownDim(context, num_split, input_rank);
        }
        // num_split is 1: output equals input on any axis, keep x shape
        return UpdateDynamicShape(context, x_shape, num_split);
    }

    OP_CHECK_IF(CheckSplitParams(context, x_shape, split_dim, num_split) == GRAPH_FAILED,
                OP_LOGE(context->GetNodeName(), "check split params failed."), return GRAPH_FAILED);

    split_dim = split_dim < 0 ? split_dim + x_shape->GetDimNum() : split_dim;
    if (x_shape->GetDim(split_dim) == -1) {
        OP_LOGD(context->GetNodeName(), "the split dim is -1 input x, will set all output the same as input.");
        return UpdateDynamicShape(context, x_shape, num_split);
    }

    gert::Shape* out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    CalOutShape(x_shape, out_shape, num_split, split_dim);

    // update dynamic output
    for (int64_t i = 1; i < num_split; i++) {
        gert::Shape* out_shape_dynamic = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_dynamic);
        *out_shape_dynamic = *out_shape;
    }

    return GRAPH_SUCCESS;
}

static bool GetSplitDimValue(gert::InferShapeRangeContext* context, int64_t& split_dim)
{
    const gert::TensorRange* split_dim_tensor_range = context->GetInputTensorRange(0);
    if (split_dim_tensor_range == nullptr) {
        return false;
    }
    const gert::Tensor* split_dim_tensor = split_dim_tensor_range->GetMax();
    if (split_dim_tensor == nullptr) {
        return false;
    }
    switch (split_dim_tensor->GetDataType()) {
        case DT_INT32: {
            const int32_t* split_dim_data = split_dim_tensor->GetData<int32_t>();
            if (split_dim_data == nullptr) {
                return false;
            }
            split_dim = static_cast<int64_t>(*split_dim_data);
            return true;
        }
        case DT_INT64: {
            const int64_t* split_dim_data = split_dim_tensor->GetData<int64_t>();
            if (split_dim_data == nullptr) {
                return false;
            }
            split_dim = *split_dim_data;
            return true;
        }
        default:
            return false;
    }
}

static graphStatus InferShapeRange4Split(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShapeRange4Split start");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numSplit = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplit);
    int64_t num_split = *numSplit;

    auto x_range = context->GetInputShapeRange(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_range);
    auto x_range_max = x_range->GetMax();
    OP_CHECK_NULL_WITH_CONTEXT(context, x_range_max);
    auto x_range_min = x_range->GetMin();
    OP_CHECK_NULL_WITH_CONTEXT(context, x_range_min);

    int64_t x_dim = static_cast<int64_t>(x_range_max->GetDimNum());

    int64_t split_dim = 0;
    bool split_dim_const = GetSplitDimValue(context, split_dim);
    if (split_dim_const && !IsDimValid(x_dim, split_dim)) {
        // invalid split_dim, fall back to the loose range
        split_dim_const = false;
    }
    if (split_dim_const && split_dim < 0) {
        split_dim += x_dim;
    }

    for (int64_t out_idx = 0; out_idx < num_split; ++out_idx) {
        auto y_range = context->GetOutputShapeRange(static_cast<size_t>(out_idx));
        OP_CHECK_NULL_WITH_CONTEXT(context, y_range);
        auto y_range_max = y_range->GetMax();
        OP_CHECK_NULL_WITH_CONTEXT(context, y_range_max);
        auto y_range_min = y_range->GetMin();
        OP_CHECK_NULL_WITH_CONTEXT(context, y_range_min);
        y_range_max->SetDimNum(static_cast<size_t>(x_dim));
        y_range_min->SetDimNum(static_cast<size_t>(x_dim));

        if (!split_dim_const) {
            // split_dim is not const: the output may be split on any axis, min is 0 and max keeps x range,
            // except that 1-D x shrinks its max to ceil(x_max / num_split)
            for (int64_t i = 0; i < x_dim; ++i) {
                y_range_min->SetDim(static_cast<size_t>(i), 0);
                if (x_dim == 1 && x_range_max->GetDim(0) != -1) {
                    y_range_max->SetDim(0, Ops::Base::CeilDiv(x_range_max->GetDim(0), num_split));
                } else {
                    y_range_max->SetDim(static_cast<size_t>(i), x_range_max->GetDim(static_cast<size_t>(i)));
                }
            }
            continue;
        }

        for (int64_t i = 0; i < x_dim; ++i) {
            if (split_dim == i) {
                if (x_range_min->GetDim(static_cast<size_t>(i)) == 1 ||
                    x_range_min->GetDim(static_cast<size_t>(i)) < 0) {
                    y_range_min->SetDim(static_cast<size_t>(i), x_range_min->GetDim(static_cast<size_t>(i)));
                } else {
                    y_range_min->SetDim(static_cast<size_t>(i),
                                        x_range_min->GetDim(static_cast<size_t>(i)) / num_split);
                }
                if (x_range_max->GetDim(static_cast<size_t>(i)) == -1) {
                    y_range_max->SetDim(static_cast<size_t>(i), -1);
                } else {
                    y_range_max->SetDim(static_cast<size_t>(i),
                                        Ops::Base::CeilDiv(x_range_max->GetDim(static_cast<size_t>(i)), num_split));
                }
            } else {
                y_range_max->SetDim(static_cast<size_t>(i), x_range_max->GetDim(static_cast<size_t>(i)));
                y_range_min->SetDim(static_cast<size_t>(i), x_range_min->GetDim(static_cast<size_t>(i)));
            }
        }
    }
    OP_LOGD(context->GetNodeName(), "InferShapeRange4Split end");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4Split(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4Split start");
    auto input_x_dtype = context->GetInputDataType(1);
    const auto output_num = context->GetComputeNodeOutputNum();
    for (size_t i = 0; i < output_num; i++) {
        context->SetOutputDataType(i, input_x_dtype);
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4Split end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Split)
    .InferShape(InferShape4Split)
    .InputsDataDependency({0})
    .InferShapeRange(InferShapeRange4Split)
    .InferDataType(InferDataType4Split);

} // namespace ops
