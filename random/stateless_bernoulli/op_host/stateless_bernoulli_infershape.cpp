/**
 * Copyright (c) Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_bernoulli_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

using namespace ge;

namespace ops {
const int32_t INDEX_INPUT_SHAPE = 0;
const int32_t INDEX_INPUT_PROB = 1;
const int32_t INDEX_INPUT_SEED = 2;
const int32_t INDEX_INPUT_OFFSET = 3;
const int32_t INDEX_OUTPUT_Y = 0;
const int32_t INDEX_ATTR = 0;

template <typename T>
graphStatus InferShapeImpl(const T* shapeData, gert::Shape& outputShape, size_t shapeSize)
{
    outputShape.SetDimNum(shapeSize);
    for (size_t i = 0U; i < shapeSize; i++) {
        outputShape.SetDim(i, shapeData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeImplFromProb(const gert::Shape* probShape, gert::Shape& outputShape, size_t shapeSize)
{
    outputShape.SetDimNum(shapeSize);
    for (size_t i = 0U; i < shapeSize; i++) {
        outputShape.SetDim(i, probShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
graphStatus InferShapeCheckShapeAndProb(
    const T* shapeData, const gert::Shape* probShape, size_t shapeSize, uint64_t* checkFlag)
{
    *checkFlag = 0;
    bool allShapeMinusOne = true;
    for (size_t i = 0U; i < shapeSize; i++) {
        if (shapeData[i] != -1) {
            allShapeMinusOne = false;
            break;
        }
    }

    bool allProbNotMinusOne = true;
    for (size_t i = 0U; i < shapeSize; i++) {
        if (probShape->GetDim(i) == -1) {
            allProbNotMinusOne = false;
            break;
        }
    }
    if (allShapeMinusOne && allProbNotMinusOne) {
        *checkFlag = 1;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4StatelessBernoulli(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), " StatelessBernoulli runtime2.0 is begin.");
    const gert::Tensor* shapeTensor = context->GetInputTensor(INDEX_INPUT_SHAPE);
    auto probShape = context->GetInputShape(INDEX_INPUT_PROB);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeTensor);
    ge::DataType shapeDtype = shapeTensor->GetDataType();
    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    auto xShapeSize = shapeTensor->GetShapeSize();
    if (xShapeSize < 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t checkFlag = 0;
    switch (shapeDtype) {
        case ge::DT_INT32: {
            Ops::Base::GetValueToShape<int32_t>(shapeTensor, *yShape);
            auto shapeData = shapeTensor->GetData<int32_t>();
            OP_CHECK_IF(InferShapeCheckShapeAndProb<int32_t>(
                         shapeData, probShape, static_cast<size_t>(xShapeSize), &checkFlag) != ge::GRAPH_SUCCESS,
                     OP_LOGE(context->GetNodeName(), "Check shape(int32) and prob failed."), return ge::GRAPH_FAILED;);
            break;
        }
        case ge::DT_INT64: {
            Ops::Base::GetValueToShape<int64_t>(shapeTensor, *yShape);
            auto shapeData = shapeTensor->GetData<int64_t>();
            OP_CHECK_IF(InferShapeCheckShapeAndProb<int64_t>(
                         shapeData, probShape, static_cast<size_t>(xShapeSize), &checkFlag) != ge::GRAPH_SUCCESS,
                     OP_LOGE(context->GetNodeName(), "Check shape(int64) and prob failed."), return ge::GRAPH_FAILED;);
            break;
        }
        default:
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(
            "shape",context->GetNodeName(),Ops::Base::ToString(shapeDtype).c_str(),"[int32, int64]");
            return ge::GRAPH_FAILED;
    }

    InferShapeImplFromProb(probShape, *yShape, static_cast<size_t>(xShapeSize));
    
    if (shapeTensor->GetDataType() == ge::DT_INT32) {
        auto shapeData = shapeTensor->GetData<int32_t>();
        return InferShapeImpl<int32_t>(shapeData, *yShape, static_cast<size_t>(xShapeSize));
        } else {
            auto shapeData = shapeTensor->GetData<int64_t>();
        return InferShapeImpl<int64_t>(shapeData, *yShape, static_cast<size_t>(xShapeSize));
        }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StatelessBernoulli)
    .InferShape(InferShape4StatelessBernoulli)
    .InputsDataDependency({INDEX_INPUT_SHAPE});
} // namespace ops