/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bitwise_or.h"
#include "opdev/aicpu/aicpu_task.h"
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
OP_TYPE_REGISTER(BitwiseOr);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT16, DataType::DT_INT32, DataType::DT_UINT16};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT16, DataType::DT_UINT16, DataType::DT_INT32, DataType::DT_INT64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
  // 910B、910_93、910_95
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B ||
      socVersion == SocVersion::ASCEND910_93 ||
      socVersion == SocVersion::ASCEND910_95) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  // 910及其他
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor *BitwiseOrAiCore(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                               aclOpExecutor *executor) {
  L0_DFX(BitwiseOrAiCore, self, other, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore BitwiseOr算子加入任务队列
  // BitwiseOr是算子的OpType，self, other是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BitwiseOr, OP_INPUT(self, other), OP_OUTPUT(out));
  OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseOrAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel
static inline const aclTensor *BitwiseOrAiCpu(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                              aclOpExecutor *executor) {
  L0_DFX(BitwiseOrAiCpu, self, other, out);

  static internal::AicpuTaskSpace space("BitwiseOr", ge::DEPEND_IN_SHAPE, true);

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu BitwiseOr算子加入任务队列
  // BitwiseOr是算子的OpType，self, other是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(BitwiseOr, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseOrAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *BitwiseOr(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  // 根据输入shape推导输出shape
  Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", ToString(self->GetViewShape()).GetString(),
            ToString(other->GetViewShape()).GetString());
    return nullptr;
  }
  // 根据输出shape申请输出tensor
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }
  if (IsAiCoreSupport(self)) {
    return BitwiseOrAiCore(self, other, out, executor);
  } else {
    return BitwiseOrAiCpu(self, other, out, executor);
  }
}
}  // namespace l0op