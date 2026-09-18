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
 * \brief InferShape implementation for Reshape operator.
 */

#include <cstdint>
#include <limits>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputXIdx = 0U;
constexpr size_t kInputShapeIdx = 1U;
constexpr size_t kOutputYIdx = 0U;
constexpr int64_t kUnknownDim = -1LL;
constexpr int64_t kZeroDim = 0LL;
constexpr int64_t kOneDim = 1LL;

template <typename T>
ge::graphStatus GetOutputShapeForEmptyTensor(gert::InferShapeContext* context, const gert::Shape& xShape,
                                             const T* reshapeData, int64_t reshapeSize, gert::Shape& outputShape)
{
    OP_CHECK_IF(reshapeData == nullptr, OP_LOGE(context->GetNodeName(), "shape input data is null"),
                return ge::GRAPH_FAILED);

    int64_t negIndex = kUnknownDim;
    int64_t numOfNegOne = 0LL;
    int64_t product = kOneDim;
    int64_t shapeDesSize = kOneDim;
    outputShape.SetDimNum(static_cast<size_t>(reshapeSize));
    for (int64_t i = 0; i < reshapeSize; i++) {
        OP_LOGI(context->GetNodeName(), "reshape target shape[%ld] = %ld", i, static_cast<int64_t>(reshapeData[i]));
        outputShape.SetDim(static_cast<size_t>(i), static_cast<int64_t>(reshapeData[i]));
        if (reshapeData[i] == kUnknownDim) {
            negIndex = i;
            numOfNegOne++;
            continue;
        }
        product *= static_cast<int64_t>(reshapeData[i]);
        shapeDesSize = (reshapeData[i] == kZeroDim) ? shapeDesSize : (shapeDesSize * reshapeData[i]);
    }

    if (numOfNegOne == 0LL && product == kZeroDim) {
        return ge::GRAPH_SUCCESS;
    }
    if (numOfNegOne == kOneDim) {
        int64_t xShapeSize = kOneDim;
        for (size_t i = 0U; i < xShape.GetDimNum(); i++) {
            if (xShape[i] < kZeroDim) {
                OP_LOGE(context->GetNodeName(), "the shape of x %ld cannot be less than 0", xShape[i]);
                return ge::GRAPH_FAILED;
            }
            xShapeSize = (xShape[i] == kZeroDim) ? xShapeSize : (xShapeSize * xShape[i]);
        }
        const int64_t realDim = (xShapeSize == shapeDesSize) ? kZeroDim : (xShapeSize / shapeDesSize);
        outputShape.SetDim(static_cast<size_t>(negIndex), realDim);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context->GetNodeName(), "Empty Tensor InferShape failed");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus EmptyTensorProcess(gert::InferShapeContext* context, const gert::Shape& xShape,
                                          const gert::Tensor& shapeTensor, gert::Shape& outputShape,
                                          int64_t reshapeSize)
{
    if (shapeTensor.GetDataType() == ge::DT_INT32) {
        return GetOutputShapeForEmptyTensor<int32_t>(context, xShape, shapeTensor.GetData<int32_t>(), reshapeSize,
                                                     outputShape);
    }
    return GetOutputShapeForEmptyTensor<int64_t>(context, xShape, shapeTensor.GetData<int64_t>(), reshapeSize,
                                                 outputShape);
}

template <typename T>
ge::graphStatus ReshapeInferShapeImpl(gert::InferShapeContext* context, const T* reshapeDims, const gert::Shape& xShape,
                                      gert::Shape& outputShape, int64_t reshapeSize)
{
    OP_CHECK_IF(reshapeDims == nullptr, OP_LOGE(context->GetNodeName(), "shape input data is null"),
                return ge::GRAPH_FAILED);

    outputShape.SetDimNum(static_cast<size_t>(reshapeSize));
    const int64_t xShapeSize = xShape.GetShapeSize();
    int64_t outputShapeSize = kOneDim;
    size_t unknownDimIdx = std::numeric_limits<size_t>::max();
    for (int64_t i = 0; i < reshapeSize; i++) {
        OP_LOGI(context->GetNodeName(), "reshape target shape[%ld] = %ld", i, static_cast<int64_t>(reshapeDims[i]));
        if (reshapeDims[i] == kZeroDim) {
            if (i >= static_cast<int64_t>(xShape.GetDimNum())) {
                OP_LOGE(context->GetNodeName(), "shape input[%ld] is empty tensor, while x_shape dim num is only [%zu]",
                        i, xShape.GetDimNum());
                return ge::GRAPH_FAILED;
            }
            outputShape.SetDim(static_cast<size_t>(i), xShape[static_cast<size_t>(i)]);
            outputShapeSize *= xShape[static_cast<size_t>(i)];
        } else if (reshapeDims[i] == kUnknownDim) {
            if (unknownDimIdx != std::numeric_limits<size_t>::max()) {
                OP_LOGE(context->GetNodeName(),
                        "only one dim of shape-input can be -1, while both shape[%zu] and shape[%ld] are -1",
                        unknownDimIdx, i);
                return ge::GRAPH_FAILED;
            }
            outputShape.SetDim(static_cast<size_t>(i), kOneDim);
            unknownDimIdx = static_cast<size_t>(i);
        } else {
            outputShape.SetDim(static_cast<size_t>(i), static_cast<int64_t>(reshapeDims[i]));
            outputShapeSize *= static_cast<int64_t>(reshapeDims[i]);
        }
    }

    if (unknownDimIdx == std::numeric_limits<size_t>::max()) {
        if (outputShapeSize != xShapeSize) {
            OP_LOGE(context->GetNodeName(), "the output shapesize %ld is not equal to x %ld", outputShapeSize,
                    xShapeSize);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (xShapeSize % outputShapeSize == 0) {
        outputShape.SetDim(unknownDimIdx, xShapeSize / outputShapeSize);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context->GetNodeName(), "input shape size %ld cannot be divided from %ld", xShapeSize, outputShapeSize);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus InferShapeForReshape(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter Reshape InferShapeForReshape");
    const gert::Shape* xShape = context->GetInputShape(kInputXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Tensor* shapeTensor = context->GetInputTensor(kInputShapeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeTensor);
    gert::Shape* outputShape = context->GetOutputShape(kOutputYIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    OP_LOGI(context->GetNodeName(), "input x dim num: %zu, shape size: %ld", xShape->GetDimNum(),
            xShape->GetShapeSize());
    OP_LOGI(context->GetNodeName(), "shape tensor size: %ld", shapeTensor->GetShapeSize());

    const int64_t reshapeSize = static_cast<int64_t>(shapeTensor->GetShapeSize());
    if (reshapeSize < kZeroDim) {
        OP_LOGE(context->GetNodeName(), "reshape_size %ld cannot be less than 0", reshapeSize);
        return ge::GRAPH_FAILED;
    }

    if (reshapeSize == kZeroDim) {
        outputShape->SetDimNum(0U);
        OP_LOGI(context->GetNodeName(), "output y dim num: %zu, shape size: %ld", outputShape->GetDimNum(),
                outputShape->GetShapeSize());
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (xShape->GetShapeSize() == kZeroDim) {
        ret = EmptyTensorProcess(context, *xShape, *shapeTensor, *outputShape, reshapeSize);
    } else if (shapeTensor->GetDataType() == ge::DT_INT32) {
        ret = ReshapeInferShapeImpl<int32_t>(context, shapeTensor->GetData<int32_t>(), *xShape, *outputShape,
                                             reshapeSize);
    } else {
        ret = ReshapeInferShapeImpl<int64_t>(context, shapeTensor->GetData<int64_t>(), *xShape, *outputShape,
                                             reshapeSize);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        OP_LOGI(context->GetNodeName(), "output y dim num: %zu, shape size: %ld", outputShape->GetDimNum(),
                outputShape->GetShapeSize());
    }
    return ret;
}
} // namespace

IMPL_OP_INFERSHAPE(Reshape).InferShape(InferShapeForReshape).InputsDataDependency({kInputShapeIdx});
} // namespace ops
