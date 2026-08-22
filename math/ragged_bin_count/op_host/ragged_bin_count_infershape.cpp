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
 * \file ragged_bin_count_infershape.cpp
 * \brief Shape inference for the RaggedBinCount operator.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/const_util.h"
#include "util/shape_util.h"

namespace ops {
namespace {
constexpr size_t INPUT_SPLITS = 0U;
constexpr size_t INPUT_VALUES = 1U;
constexpr size_t INPUT_SIZE = 2U;
constexpr size_t INPUT_WEIGHTS = 3U;
constexpr size_t OUTPUT_RESULT = 0U;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;
constexpr int64_t MIN_SPLITS_NUM = 2;
constexpr size_t MIN_VALUES_RANK = 1U;
constexpr size_t MAX_VALUES_RANK = 2U;

ge::graphStatus CheckRankInRange(const gert::InferShapeContext* context, const gert::Shape& shape, size_t minRank,
                                 size_t maxRank, const char* inputName)
{
    if (Ops::Base::IsUnknownRank(shape)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(shape.GetDimNum() < minRank || shape.GetDimNum() > maxRank,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), inputName, Ops::Base::ToString(shape),
                                                      "rank must be in [" + std::to_string(minRank) + ", " +
                                                          std::to_string(maxRank) + "], but got " +
                                                          std::to_string(shape.GetDimNum())),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckWeightsShape(const gert::InferShapeContext* context, const gert::Shape& valuesShape,
                                  const gert::Shape& weightsShape)
{
    if (Ops::Base::IsUnknownRank(weightsShape)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        CheckRankInRange(context, weightsShape, MIN_VALUES_RANK, MAX_VALUES_RANK, "weights") != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "weights", Ops::Base::ToString(weightsShape),
                                              "the rank of weights is invalid"),
        return ge::GRAPH_FAILED);

    const bool isEmpty = weightsShape.GetDimNum() == 1U && weightsShape.GetDim(0U) == 0;
    if (isEmpty || Ops::Base::IsUnknownRank(valuesShape)) {
        return ge::GRAPH_SUCCESS;
    }

    // A dynamic rank-1 weights shape may resolve to the only legal empty representation [0].
    const bool mayResolveToEmpty = weightsShape.GetDimNum() == 1U && weightsShape.GetDim(0U) == UNKNOWN_DIM_VALUE;
    OP_CHECK_IF(weightsShape.GetDimNum() != valuesShape.GetDimNum() && !mayResolveToEmpty,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "weights, values",
                    Ops::Base::ToString(weightsShape) + ", " + Ops::Base::ToString(valuesShape),
                    "non-empty weights must have the same rank as values"),
                return ge::GRAPH_FAILED);
    if (weightsShape.GetDimNum() != valuesShape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }

    for (size_t dim = 0U; dim < valuesShape.GetDimNum(); ++dim) {
        const int64_t valuesDim = valuesShape.GetDim(dim);
        const int64_t weightsDim = weightsShape.GetDim(dim);
        OP_CHECK_IF(
            valuesDim != UNKNOWN_DIM_VALUE && weightsDim != UNKNOWN_DIM_VALUE && valuesDim != weightsDim,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "weights, values",
                Ops::Base::ToString(weightsShape) + ", " + Ops::Base::ToString(valuesShape),
                "non-empty weights must have exactly the same shape as values, mismatch at dim " + std::to_string(dim)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// The four input shapes, so fetching and null-checking them is one call instead of eight lines.
struct InputShapes {
    const gert::Shape* splits = nullptr;
    const gert::Shape* values = nullptr;
    const gert::Shape* size = nullptr;
    const gert::Shape* weights = nullptr;
};

ge::graphStatus FetchInputShapes(gert::InferShapeContext* context, InputShapes& shapes)
{
    shapes.splits = context->GetInputShape(INPUT_SPLITS);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.splits);
    shapes.values = context->GetInputShape(INPUT_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.values);
    shapes.size = context->GetInputShape(INPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.size);
    shapes.weights = context->GetInputShape(INPUT_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.weights);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSizeShape(const gert::InferShapeContext* context, const gert::Shape& sizeShape)
{
    if (Ops::Base::IsUnknownRank(sizeShape)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(sizeShape.GetDimNum() != 1U,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "size", Ops::Base::ToString(sizeShape),
                    "size must be 1D with shape [1], but got rank " + std::to_string(sizeShape.GetDimNum())),
                return ge::GRAPH_FAILED);
    const int64_t sizeDim = sizeShape.GetDim(0U);
    OP_CHECK_IF(sizeDim != UNKNOWN_DIM_VALUE && sizeDim != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "size", Ops::Base::ToString(sizeShape),
                                                      "size must have shape [1]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputShapes(const gert::InferShapeContext* context, const InputShapes& shapes)
{
    if (!Ops::Base::IsUnknownRank(*shapes.splits)) {
        OP_CHECK_IF(shapes.splits->GetDimNum() != 1U,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "splits", Ops::Base::ToString(*shapes.splits),
                        "splits must be 1D, but got rank " + std::to_string(shapes.splits->GetDimNum())),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        CheckRankInRange(context, *shapes.values, MIN_VALUES_RANK, MAX_VALUES_RANK, "values") != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "values", Ops::Base::ToString(*shapes.values),
                                              "the rank of values is invalid"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSizeShape(context, *shapes.size) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "size", Ops::Base::ToString(*shapes.size),
                                                      "the shape of size is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckWeightsShape(context, *shapes.values, *shapes.weights) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "weights", Ops::Base::ToString(*shapes.weights),
                                              "the shape of weights is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Rows come from splits; -1 whenever splits is dynamic, because the row count is then not knowable yet.
ge::graphStatus InferRowNum(const gert::InferShapeContext* context, const gert::Shape& splitsShape, int64_t& rowNum)
{
    rowNum = UNKNOWN_DIM_VALUE;
    if (Ops::Base::IsUnknownRank(splitsShape)) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t splitsNum = splitsShape.GetDim(0U);
    if (splitsNum == UNKNOWN_DIM_VALUE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(splitsNum < MIN_SPLITS_NUM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "splits", Ops::Base::ToString(splitsShape),
                    "splits must contain at least two elements, but got " + std::to_string(splitsNum)),
                return ge::GRAPH_FAILED);
    rowNum = splitsNum - 1;
    return ge::GRAPH_SUCCESS;
}

// Bins come from the value-dependent `size` input; -1 whenever GE has not materialised it yet.
ge::graphStatus InferBinNum(gert::InferShapeContext* context, int64_t& binNum)
{
    binNum = UNKNOWN_DIM_VALUE;
    const gert::Tensor* sizeTensor = context->GetInputTensor(INPUT_SIZE);
    if (sizeTensor == nullptr || sizeTensor->GetAddr() == nullptr ||
        !Ops::Base::GetConstInt(context, static_cast<int64_t>(INPUT_SIZE), binNum)) {
        binNum = UNKNOWN_DIM_VALUE;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(binNum < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", std::to_string(binNum),
                                                      "size must be non-negative"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeForRaggedBinCount(gert::InferShapeContext* context)
{
    InputShapes shapes;
    OP_CHECK_IF(FetchInputShapes(context, shapes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to fetch the RaggedBinCount input shapes."),
                return ge::GRAPH_FAILED);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_RESULT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    OP_CHECK_IF(CheckInputShapes(context, shapes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RaggedBinCount input shape validation failed."),
                return ge::GRAPH_FAILED);

    int64_t rowNum = UNKNOWN_DIM_VALUE;
    OP_CHECK_IF(InferRowNum(context, *shapes.splits, rowNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to infer the RaggedBinCount row count."),
                return ge::GRAPH_FAILED);
    int64_t binNum = UNKNOWN_DIM_VALUE;
    OP_CHECK_IF(InferBinNum(context, binNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to infer the RaggedBinCount bin count."),
                return ge::GRAPH_FAILED);

    outputShape->SetDimNum(2U);
    outputShape->SetDim(0U, rowNum);
    outputShape->SetDim(1U, binNum);
    OP_LOGD(context->GetNodeName(), "RaggedBinCount output shape is [%ld, %ld].", rowNum, binNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RaggedBinCount).InferShape(InferShapeForRaggedBinCount).InputsDataDependency({INPUT_SIZE});
} // namespace ops
