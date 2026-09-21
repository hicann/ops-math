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
 * \file fill_v2_infershape.cpp
 * \brief FillV2 InferShape (gert新机制，处理dims为常量和非常量两种场景)
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/const_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_INDEX_DIMS = 0;

template <typename T>
static bool inline FillV2GetValueToShape(const gert::Tensor* constTensor, gert::Shape& shape)
{
    const T* constValue = constTensor->GetData<T>();
    if (constValue == nullptr) {
        return false;
    }
    const size_t constNum = static_cast<size_t>(constTensor->GetShapeSize());
    shape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        shape.AppendDim(static_cast<int64_t>(constValue[i]));
    }
    return true;
}

static ge::graphStatus InferShape4FillV2(gert::InferShapeContext* context)
{
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    const gert::Tensor* constTensor = context->GetInputTensor(INPUT_INDEX_DIMS);
    if (constTensor == nullptr || constTensor->GetData<int64_t>() == nullptr || constTensor->GetShapeSize() <= 0) {
        // dims不是常量，无法获取具体值
        auto dimsShape = context->GetInputShape(INPUT_INDEX_DIMS);
        if (dimsShape == nullptr) {
            out_shape->SetDimNum(0);
            out_shape->AppendDim(-2);
            OP_LOGW(context->GetNodeName(), "dims shape is nullptr, set output to UNKNOWN_RANK");
            return GRAPH_SUCCESS;
        }
        int64_t dimValue = dimsShape->GetDimNum() == 0 ? 0 : dimsShape->GetDim(0);
        out_shape->SetDimNum(0);
        for (int64_t m = 0; m < dimValue; ++m) {
            out_shape->AppendDim(-1);
        }
        if (dimValue <= 0) {
            out_shape->AppendDim(-2);
        }
        OP_LOGI(context->GetNodeName(), "dims is not const, dim value is %ld", dimValue);
        return GRAPH_SUCCESS;
    }

    // dims是常量，按dtype读取具体值
    ge::DataType constDtype = constTensor->GetDataType();
    bool ret = false;
    switch (constDtype) {
        case ge::DT_INT16: {
            ret = FillV2GetValueToShape<int16_t>(constTensor, *out_shape);
            break;
        }
        case ge::DT_INT32: {
            ret = FillV2GetValueToShape<int32_t>(constTensor, *out_shape);
            break;
        }
        case ge::DT_INT64: {
            ret = FillV2GetValueToShape<int64_t>(constTensor, *out_shape);
            break;
        }
        default: {
            OP_LOGE(context->GetNodeName(), "dims dtype only support [int16, int32, int64], but is %s",
                    Ops::Base::ToString(constDtype).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    OP_CHECK_IF(!ret, OP_LOGE(context->GetNodeName(), "FillV2GetValueToShape failed!"), return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "InferShape4FillV2: output shape is %s", Ops::Base::ToString(*out_shape).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FillV2).InferShape(InferShape4FillV2).InputsDataDependency({INPUT_INDEX_DIMS});
} // namespace ops
