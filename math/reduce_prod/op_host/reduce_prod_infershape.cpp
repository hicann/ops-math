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
 * \file reduce_prod_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_api/op_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
template <typename T>
ge::graphStatus ReduceDimsWithKeepDims(const gert::Shape* x_shape, const T* axes_dims, int32_t axes_size,
                                       gert::Shape* output_shape) {
    T dim_num = x_shape->GetDimNum();
    const bool is_scalar = x_shape->GetDimNum() == 0;
    dim_num = is_scalar ? 1 : dim_num;
    *output_shape = *x_shape;
    for (int32_t i = 0; i < axes_size; i++) {
        OP_CHECK_IF(!IsDimValid(dim_num, axes_dims[i]), OP_LOGE("reduce", "axes_dims is invalid"), return ge::GRAPH_FAILED);
        if (is_scalar) {
            // no need to update output shape, when input is scalar
            continue;
        }
        T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
        output_shape->SetDim(dim, 1);
    }
    OP_LOGD("ReduceDimsWithKeepDims", "after reduce output shape is %s.", ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDimsWithoutKeepDims(const gert::Shape* x_shape, const T* axes_dims, int32_t axes_size,
                                          gert::Shape* output_shape) {
    T dim_num = x_shape->GetDimNum();
    output_shape->SetDimNum(0);
    for (T j = 0; j < dim_num; j++) {
        bool reduce_flag = false;
        for (int32_t i = 0; i < axes_size; i++) {
            OP_CHECK_IF(!IsDimValid(dim_num, axes_dims[i]), OP_LOGE("reduce", "axes_dims is invalid"), return ge::GRAPH_FAILED);
            T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
            if (dim == j) {
                reduce_flag = true;
                break;
            }
        }
        if (!reduce_flag) {
            output_shape->AppendDim(x_shape->GetDim(j));
        }
    }

    OP_LOGD("ReduceDimsWithoutKeepDims", "after reduce output shape is %s.", ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDims(const gert::Shape* x_shape, const gert::Tensor* axes_tensor, int32_t axes_size,
                           const bool keep_dims, gert::Shape* output_shape) {
    const T* axes_dims = axes_tensor->GetData<T>();
    if (keep_dims) {
        return ReduceDimsWithKeepDims<T>(x_shape, axes_dims, axes_size, output_shape);
    }
    return ReduceDimsWithoutKeepDims<T>(x_shape, axes_dims, axes_size, output_shape);
}

static ge::graphStatus InferShape4ReduceProd(gert::InferShapeContext *context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto axes_tensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const bool* keep_dims = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, keep_dims);

    auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());

    OP_CHECK_IF(axes_size < 0,
            OP_LOGE(context->GetNodeName(), "axes num cannot be less than 0!"),
            return ge::GRAPH_FAILED);

    if (axes_size == 0) {
        *out_shape = *in_shape;
        OP_LOGD(context->GetNodeName(), "axes is empty tensor, will ignore infer, set output shape = input shape");
        return ge::GRAPH_SUCCESS;
    }

    auto dtype = axes_tensor->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
            OP_LOGE(
                context->GetNodeName(), "axes datatype %s must in (int32, int64)", ToString(dtype).c_str()),
            return ge::GRAPH_FAILED);
    if (dtype == ge::DT_INT32) {
        return ReduceDims<int32_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
    }

    return ReduceDims<int64_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
}

IMPL_OP_INFERSHAPE(ReduceProd).InferShape(InferShape4ReduceProd).InputsDataDependency({1});
} // namespace ops