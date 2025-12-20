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
 * \file reduce_mean_with_count.cpp
 * \brief
 */

#include "reduce_mean_with_count.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceMeanWithCount);

const aclTensor* ReduceMeanWithCount(const aclTensor* input, const aclTensor* count, const aclTensor* countSum,
                                     const aclIntArray* axes, bool keepDims, aclOpExecutor* executor) {
  L0_DFX(ReduceMeanWithCount, input, count, countSum, axes, keepDims);
  auto out = executor->AllocTensor(input->GetDataType(), input->GetStorageFormat(), input->GetOriginalFormat());
  auto ret =
      INFER_SHAPE(ReduceMeanWithCount, OP_INPUT(input, count, countSum), OP_OUTPUT(out), OP_ATTR(axes, keepDims));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ReduceMeanWithCount infer shape failed.");
    return nullptr;
  }
  op::Shape shape = input->GetViewShape();
  auto size = axes->Size();
  size_t dimNum = shape.GetDimNum();
  if (keepDims) {
    for (uint64_t i = 0; i < size; i++) {
      int64_t dimIndex = static_cast<int64_t>((*axes)[i]);
      int64_t dimNew = dimIndex >= 0 ? dimIndex : dimIndex + dimNum;
      shape.SetDim(dimNew, 1);
    }
    out->SetViewShape(shape);
  }
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ReduceMeanWithCount, OP_INPUT(input, count, countSum), OP_OUTPUT(out),
                                         OP_ATTR(axes, keepDims));
  OP_CHECK(retAicore ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceMeanWithCountAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

}  // namespace l0op