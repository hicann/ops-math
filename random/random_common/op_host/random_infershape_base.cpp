/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file random_infershape_base.cpp
 * \brief
 */
#include "random_infershape_base.h"

namespace ops {
namespace randomCommon {
template <typename T>
ge::graphStatus HandleShapeTensor(gert::Shape& outShape, size_t xShapeSize, const T* xShapeData)
{
    outShape.SetDimNum(xShapeSize);
    for (size_t i = 0U; i < xShapeSize; i++) {
        outShape.SetDim(i, xShapeData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

bool InferShapeForUnknow(const gert::Shape& inShape, gert::Shape& outShape)
{
    if (Ops::Base::IsUnknownRank(inShape)) {
        Ops::Base::SetUnknownRank(outShape);
        return true;
    }
    if (Ops::Base::IsUnknownShape(inShape)) {
        Ops::Base::SetUnknownShape(inShape.GetDimNum(), outShape);
        return true;
    }
    return false;
}

bool DependencyMode(const gert::Tensor* inTensor, gert::Shape& outShape, size_t xShapeSize)
{
    ge::DataType shapeDtype = inTensor->GetDataType();
    if (shapeDtype == ge::DT_INT32) {
        auto xShapeData = inTensor->GetData<int32_t>();
        if (xShapeData == nullptr) {
            outShape.SetDimNum(0);
            return true;
        }
        if (HandleShapeTensor<int32_t>(outShape, xShapeSize, xShapeData) == ge::GRAPH_SUCCESS) {
            return true;
        }
    } else if (shapeDtype == ge::DT_INT64) {
        auto xShapeData = inTensor->GetData<int64_t>();
        if (xShapeData == nullptr) {
            outShape.SetDimNum(0);
            return true;
        }
        if (HandleShapeTensor<int64_t>(outShape, xShapeSize, xShapeData) == ge::GRAPH_SUCCESS) {
            return true;
        }
    }
    return false;
}

bool InputAndOutputCheck(
    gert::InferShapeContext* context, const std::unordered_map<std::string, size_t>& inputMap,
    const std::unordered_map<std::string, size_t>& outputMap, int64_t& maskIndex, int64_t& offsetIndex)
{
    for (const auto& item : inputMap) {
        size_t inputIndex = item.second;
        auto input = context->GetInputTensor(inputIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, input);
    }

    for (const auto& item : outputMap) {
        const std::string& outputName = item.first;
        size_t outputIndex = item.second;
        auto output = context->GetOutputShape(outputIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, output);
        if (outputName == "mask") {
            maskIndex = outputIndex;
        }
        if (outputName == "offset") {
            offsetIndex = outputIndex;
        }
    }
    return true;
}

ge::graphStatus CommonInferShape(
    gert::InferShapeContext* context, const std::unordered_map<std::string, size_t>& inputMap,
    const std::unordered_map<std::string, size_t>& outputMap, int32_t mode)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t maskIndex = -1;
    int64_t offsetIndex = -1;
    if (!InputAndOutputCheck(context, inputMap, outputMap, maskIndex, offsetIndex)) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    int64_t xShapeSize = inShape->GetShapeSize();
    if (xShapeSize < 0) {
        return ge::GRAPH_FAILED;
    }
    if (maskIndex >= 0) {
        static constexpr int64_t MASK_ALIGN_SIZE = 128;
        static constexpr int64_t MASK_BIT_TO_UINT8 = 8;
        gert::Shape* maskOutputShape = context->GetOutputShape(maskIndex);
        if (!InferShapeForUnknow(*inShape, *maskOutputShape)) {
            int64_t maskSize = (xShapeSize + MASK_ALIGN_SIZE - 1) / MASK_ALIGN_SIZE * MASK_ALIGN_SIZE / MASK_BIT_TO_UINT8;
            maskOutputShape->SetDimNum(1);
            maskOutputShape->SetDim(0, maskSize);
        }
    }
    if (offsetIndex >= 0) {
        gert::Shape* offsetOutputShape = context->GetOutputShape(offsetIndex);
        if (!InferShapeForUnknow(*inShape, *offsetOutputShape)) {
            offsetOutputShape->SetDimNum(1);
            offsetOutputShape->SetDim(0, 1);
        }
    }
    if (InferShapeForUnknow(*inShape, *outShape)) {
        return ge::GRAPH_SUCCESS;
    }
    if (mode == MODE_NO_DEPENDENCY) {
        *outShape = *inShape;
        return ge::GRAPH_SUCCESS;
    }
    if (mode == MODE_DEPENDENCY) {
        const gert::Tensor* inTensor = context->GetInputTensor(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, inTensor);
        if (DependencyMode(inTensor, *outShape, static_cast<size_t>(xShapeSize))) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}
} // namespace randomCommon
} // namespace ops
