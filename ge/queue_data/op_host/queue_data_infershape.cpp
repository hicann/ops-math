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
 * \file queue_data_infershape.cpp
 * \brief
 */
#include <numeric>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static const std::map<ge::DataType, int64_t> kDataTypeSizeMap = {
    {ge::DT_FLOAT, sizeof(float)},     {ge::DT_FLOAT16, sizeof(float) / 2}, {ge::DT_INT8, sizeof(int8_t)},
    {ge::DT_INT16, sizeof(int16_t)},   {ge::DT_INT32, sizeof(int32_t)},     {ge::DT_INT64, sizeof(int64_t)},
    {ge::DT_UINT8, sizeof(uint8_t)},   {ge::DT_UINT16, sizeof(uint16_t)},   {ge::DT_UINT32, sizeof(uint32_t)},
    {ge::DT_UINT64, sizeof(uint64_t)}, {ge::DT_DOUBLE, sizeof(double)},     {ge::DT_BOOL, sizeof(bool)}};

static const int64_t kItemInfoSize = 64;

namespace {
constexpr size_t kAttrOutputTypesIndex = 2U;
constexpr size_t kAttrOutputShapesIndex = 3U;
} // namespace

static ge::graphStatus InferShape4QueueData(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* output_types_vec = attrs->GetAttrPointer<gert::ContinuousVector>(kAttrOutputTypesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_types_vec);
    auto types_num = output_types_vec->GetSize();

    const auto* output_shapes_vec = attrs->GetListListInt(kAttrOutputShapesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shapes_vec);
    auto shapes_num = output_shapes_vec->GetSize();

    if (types_num != shapes_num) {
        OP_LOGE(context->GetNodeName(), "attr[output_types] and attr[output_shapes] should be the same length");
        return ge::GRAPH_FAILED;
    }

    const auto* types_data = reinterpret_cast<const int64_t*>(output_types_vec->GetData());

    int64_t total_size = 0;
    for (size_t i = 0; i < types_num; ++i) {
        auto dtype = static_cast<ge::DataType>(types_data[i]);
        auto it = kDataTypeSizeMap.find(dtype);
        if (it == kDataTypeSizeMap.end()) {
            total_size = -1;
            break;
        }
        int64_t type_size = it->second;
        const auto* shape_cv = output_shapes_vec->Get(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape_cv);
        auto shape_dims = shape_cv->GetSize();
        const auto* shape_data = reinterpret_cast<const int64_t*>(shape_cv->GetData());
        int64_t data_len = std::accumulate(shape_data, shape_data + shape_dims, type_size, std::multiplies<int64_t>{});
        int64_t dims_size = sizeof(int64_t) * shape_dims;
        total_size += kItemInfoSize + dims_size + data_len;
    }

    auto yshape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yshape);
    yshape->SetDimNum(1);
    yshape->SetDim(0, total_size);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4QueueData(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, ge::DT_UINT8);
}

IMPL_OP_INFERSHAPE(QueueData).InferShape(InferShape4QueueData).InferDataType(InferDataType4QueueData);

} // namespace ops
