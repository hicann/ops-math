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
 * \file stateless_random_normal_v2_infershape.cpp
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
template <typename T>
static graphStatus InferShapeImpl(const T* shape_data, gert::Shape& output_shape, size_t shape_size)
{
    output_shape.SetDimNum(shape_size);
    for (size_t i = 0U; i < shape_size; i++) {
        output_shape.SetDim(i, shape_data[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static graphStatus StatelessRandomNormalV2InferShapeFunc(gert::InferShapeContext* context)
{
    auto shape_tensor = context->GetInputTensor(0);
    auto output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    auto x_shape_size = shape_tensor->GetShapeSize();
    if (x_shape_size < 0) {
        return ge::GRAPH_FAILED;
    }

    if (shape_tensor->GetDataType() == ge::DT_INT32) {
        auto shape_data = shape_tensor->GetData<int32_t>();
        return InferShapeImpl<int32_t>(shape_data, *output_shape, static_cast<size_t>(x_shape_size));
    } else {
        auto shape_data = shape_tensor->GetData<int64_t>();
        return InferShapeImpl<int64_t>(shape_data, *output_shape, static_cast<size_t>(x_shape_size));
    }
}

IMPL_OP_INFERSHAPE(StatelessRandomNormalV2).InputsDataDependency({0}).InferShape(StatelessRandomNormalV2InferShapeFunc);
} // namespace ops