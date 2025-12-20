/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file random_standard_normal_v2.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
template <typename T>
ge::graphStatus RandomStandardNormalV2InferShapeImpl(const T *shape_dims, gert::Shape &outputShape,
                                            size_t shape_size) {
    outputShape.SetDimNum(shape_size);
    for (size_t i = 0U; i < shape_size; i++) {
        outputShape.SetDim(i, shape_dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForRandomStandardNormalV2(gert::InferShapeContext *context) {
    auto x_shape_tensor = context->GetInputTensor(0);
    auto outputShape = context->GetOutputShape(0);
    auto const_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, const_shape);

    const_shape->SetDimNum(1);
    const_shape->SetDim(0, 1);

    auto x_shape_size = x_shape_tensor->GetShapeSize();
    if (x_shape_size < 0) {
        return ge::GRAPH_FAILED;
    }
    if (x_shape_tensor->GetDataType() == ge::DT_INT32) {
        auto xShapeData = x_shape_tensor->GetData<int32_t>();
        return RandomStandardNormalV2InferShapeImpl<int32_t>(xShapeData, *outputShape, static_cast<size_t>(x_shape_size));
    } else {
        auto xShapeData = x_shape_tensor->GetData<int64_t>();
        return RandomStandardNormalV2InferShapeImpl<int64_t>(xShapeData, *outputShape, static_cast<size_t>(x_shape_size));
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RandomStandardNormalV2).InputsDataDependency({0}).InferShape(InferShapeForRandomStandardNormalV2);
} // namespace ops