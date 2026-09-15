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
 * \file stateless_drop_out_gen_mask_infershape.cpp
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kInputIndex0 = 0U;
constexpr int64_t kBitNumber = 128;
constexpr int64_t kBytesPer128Bit = 16;
constexpr int64_t kOutputRank = 1; // 输出恒为 1-D
} // namespace

using namespace ge;
namespace ops {
static int64_t InferShapeGetShapeSizeBytes(int64_t shape_size)
{
    if (shape_size < 0) {
        return ge::UNKNOWN_DIM;
    }

    int64_t n128s = shape_size / kBitNumber;
    if (shape_size % kBitNumber != 0) {
        n128s++;
    }

    return n128s * kBytesPer128Bit;
}

template <typename T>
static graphStatus InferShapeImpl(const T* shape_data, gert::Shape& output_shape, int64_t shape_size,
                                  const char* node_name)
{
    int64_t output_shapesize = 1;
    for (int64_t i = 0; i < shape_size; i++) {
        if (shape_data[i] < 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(node_name, "shape",
                                         std::to_string(static_cast<int64_t>(shape_data[i])).c_str(), "non-negative");
            return ge::GRAPH_FAILED;
        }
        output_shapesize *= shape_data[i];
    }
    int64_t output_shape_num = InferShapeGetShapeSizeBytes(output_shapesize);
    output_shape.SetDimNum(1);
    output_shape.SetDim(0, output_shape_num);
    return ge::GRAPH_SUCCESS;
}

static graphStatus DropOutGenMaskInferShapeFunc(gert::InferShapeContext* context)
{
    auto output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    auto shape_tensor = context->GetInputTensor(kInputIndex0);
    if (shape_tensor == nullptr) {
        Ops::Base::SetUnknownShape(kOutputRank, *output_shape);
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape& input_desc = shape_tensor->GetShape().GetStorageShape();
    if (Ops::Base::IsUnknownRank(input_desc) || Ops::Base::IsUnknownShape(input_desc)) {
        Ops::Base::SetUnknownShape(kOutputRank, *output_shape);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t shape_size = shape_tensor->GetShapeSize();
    if (shape_size < 0) {
        Ops::Base::SetUnknownShape(kOutputRank, *output_shape);
        return ge::GRAPH_SUCCESS;
    }

    const ge::DataType shape_dtype = shape_tensor->GetDataType();

    if (shape_dtype == ge::DT_INT32) {
        auto shape_data = shape_tensor->GetData<int32_t>();
        if (shape_data == nullptr) {
            Ops::Base::SetUnknownShape(kOutputRank, *output_shape);
            return ge::GRAPH_SUCCESS;
        }
        return InferShapeImpl<int32_t>(shape_data, *output_shape, shape_size, context->GetNodeName());
    }
    if (shape_dtype == ge::DT_INT64) {
        auto shape_data = shape_tensor->GetData<int64_t>();
        if (shape_data == nullptr) {
            Ops::Base::SetUnknownShape(kOutputRank, *output_shape);
            return ge::GRAPH_SUCCESS;
        }
        return InferShapeImpl<int64_t>(shape_data, *output_shape, shape_size, context->GetNodeName());
    }
    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "shape", Ops::Base::ToString(shape_dtype).c_str(),
                              "[int32, int64]");
    return ge::GRAPH_FAILED;
}

IMPL_OP_INFERSHAPE(StatelessDropOutGenMask)
    .InputsDataDependency({kInputIndex0})
    .InferShape(DropOutGenMaskInferShapeFunc);

} // namespace ops
