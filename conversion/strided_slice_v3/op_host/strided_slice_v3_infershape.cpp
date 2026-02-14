/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strided_slice_v3.cc
 * \brief
 */

#include <numeric>
#include <cmath>
#include "register/op_impl_registry.h"
#include "../../strided_slice/op_host/strided_slice_util.h"
#include "log/log.h"
#include "util/const_util.h"

using namespace std;
namespace {
const std::string OP_NAME = "StridedSliceV3";
const int INDEX_X = 0;
const int INDEX_BEGIN = 1;
const int INDEX_END = 2;
const int INDEX_AXES = 3;
const int INDEX_STRIDES = 4;
const int INDEX_Y = 0;
}  // namespace

namespace ops {
static int64_t GetConstIndexValue(const gert::Tensor* tensor, size_t idx) {
  // idx must be valid
  int64_t value = 0;
  if (tensor->GetDataType() == ge::DT_INT32) {
    const int32_t* data = tensor->GetData<int32_t>();
    value = static_cast<int64_t>(data[idx]);
  } else {
    const int64_t* data = tensor->GetData<int64_t>();
    value = data[idx];
  }
  OP_LOGD(OP_NAME.c_str(), "const tensor[%ld] is %ld.", idx, value);
  return value;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, size_t idx, int64_t input_size, int64_t clip_lower,
                                  int64_t clip_upper) {
  // idx must be valid
  int64_t value = 0;
  if (tensor->GetDataType() == ge::DT_INT32) {
    const int32_t* data = tensor->GetData<int32_t>();
    value = static_cast<int64_t>(data[idx]);
  } else {
    const int64_t* data = tensor->GetData<int64_t>();
    value = data[idx];
  }
  if (value < 0) {
    value += input_size;
  }

  // clamp value
  if (value < clip_lower) {
    value = clip_lower;
  } else if (value > clip_upper) {
    value = clip_upper;
  }
  OP_LOGD(OP_NAME.c_str(), "const tensor[%ld] is %ld.", idx, value);
  return value;
}

template <typename T>
static void PositiveAxisImpl(int32_t input_dims, const gert::Tensor* axis_tensor, vector<int32_t>& new_axis) {
  const int64_t axis_size = axis_tensor->GetShapeSize();
  const T* data = axis_tensor->GetData<T>();
  for (int i = 0; i < axis_size; i++) {
    int64_t value = static_cast<int64_t>(data[i]);
    if (value >= 0 && value < input_dims) {
      new_axis.push_back(value);
    } else if (value < 0 && value >= -input_dims) {
      new_axis.push_back(value + input_dims);
    }
  }
  return;
}

static std::vector<int32_t> ConstructValidAxis(int32_t input_dims, const gert::Tensor* axis_tensor) {
  std::vector<int32_t> new_axis;
  if (!axis_tensor || axis_tensor->GetShapeSize() == 0) {
    new_axis.resize(input_dims);
    std::iota(new_axis.begin(), new_axis.end(), 0);
    return new_axis;
  }
  if (axis_tensor->GetDataType() == ge::DT_INT32) {
    PositiveAxisImpl<int32_t>(input_dims, axis_tensor, new_axis);
  } else if (axis_tensor->GetDataType() == ge::DT_INT64) {
    PositiveAxisImpl<int64_t>(input_dims, axis_tensor, new_axis);
  }
  return new_axis;
}

static ge::graphStatus StridedSliceV3InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(INDEX_X);
  gert::Shape* out_shape = context->GetOutputShape(INDEX_Y);
  const gert::Tensor* begin_tensor = context->GetInputTensor(INDEX_BEGIN);
  const gert::Tensor* end_tensor = context->GetInputTensor(INDEX_END);
  if (x_shape == nullptr || out_shape == nullptr || begin_tensor == nullptr || end_tensor == nullptr) {
    OP_LOGE(OP_NAME.c_str(), "input tensor or output tensor is null. Please check.");
    return ge::GRAPH_FAILED;
  }
  *out_shape = *x_shape;  // init output_shape with input_shape

  int32_t input_dim_num = static_cast<int32_t>(x_shape->GetDimNum());
  std::vector<int32_t> new_axis = ConstructValidAxis(input_dim_num, context->GetOptionalInputTensor(INDEX_AXES));
  const gert::Tensor* strides_tensor = context->GetOptionalInputTensor(INDEX_STRIDES);
  const int32_t strides_size = (strides_tensor) ? static_cast<int32_t>(strides_tensor->GetShapeSize()) : 0;
  const int32_t begins_size = static_cast<int32_t>(begin_tensor->GetShapeSize());
  const int32_t ends_size = static_cast<int32_t>(end_tensor->GetShapeSize());

  const int32_t axis_size = static_cast<int32_t>(new_axis.size());
  if (axis_size == 0) {
    OP_LOGE(OP_NAME.c_str(), "axis_size is 0. Please check.");
    return ge::GRAPH_FAILED;
  }
  for (int32_t i = 0; i < axis_size; i++) {
    const int32_t axis_value = new_axis[i];
    int64_t step_value = 1;
    if (i < strides_size) {
      step_value = GetConstIndexValue(strides_tensor, i);
    }
    int64_t cur_axis_input_size = x_shape->GetDim(axis_value);
    int64_t begin_value = 0;
    if (i < begins_size) {
      int64_t clip_upper = cur_axis_input_size;
      if (step_value < 0) {
        clip_upper -= 1;  // if stpep <0, start from last valid_index
      }
      begin_value = GetConstIndexValue(begin_tensor, i, cur_axis_input_size, 0, clip_upper);
    }
    int64_t end_value = cur_axis_input_size;
    if (i < ends_size) {
      int64_t clip_lower = 0;
      if (step_value < 0) {
        clip_lower = -1;  // if stpep <0, end with first valid_index
      }
      end_value = GetConstIndexValue(end_tensor, i, cur_axis_input_size, clip_lower, cur_axis_input_size);
    }
    int64_t cur_out_size = std::ceil((end_value - begin_value) / static_cast<float>(step_value));
    if (cur_out_size < 0) {
      cur_out_size = 0;
    }
    out_shape->SetDim(axis_value, cur_out_size);
  }
  OP_LOGD(OP_NAME.c_str(), "out_shape: %s", Ops::Base::ToString(*out_shape).c_str());
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceV3).InferShape(StridedSliceV3InferShape).InputsDataDependency({1, 2, 3, 4});
}  // namespace ops