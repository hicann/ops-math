/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tanh_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(TanhGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST  = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST  = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型和数据类型判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *gradOutput) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
      socVersion == SocVersion::ASCEND910_95) {
    return CheckType(gradOutput->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(gradOutput->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor *TanhGradAICpu(const aclTensor *gradOutput, const aclTensor *output,
                                      aclTensor *gradInput, aclOpExecutor *executor) {
  L0_DFX(TanhGradAICpu, output, gradOutput, gradInput);
  static internal::AicpuTaskSpace space("TanhGrad", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(TanhGrad, OP_ATTR_NAMES(), OP_INPUT(output, gradOutput), OP_OUTPUT(gradInput));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return gradInput;
}

// AICORE算子kernel
static const aclTensor *TanhGradAICore(const aclTensor *gradOutput, const aclTensor *output,
                                       aclTensor *gradInput, aclOpExecutor *executor) {
  L0_DFX(TanhGradAICore, output, gradOutput, gradInput);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(TanhGrad, OP_INPUT(output, gradOutput), OP_OUTPUT(gradInput));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "TanhGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return gradInput;
}

const aclTensor *TanhGrad(const aclTensor *gradOutput, const aclTensor *output, aclOpExecutor *executor) {
  op::Shape broadcastShape;
  op::DataType gradInputDtype;
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 && IsAiCoreSupport(gradOutput) &&
      gradOutput->GetDataType() != output->GetDataType()) {
    gradInputDtype = op::DataType::DT_FLOAT;
  } else {
    gradInputDtype = gradOutput->GetDataType();
  }
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, output, broadcastShape, return nullptr);
  auto gradInput = executor->AllocTensor(broadcastShape, gradInputDtype,
                                         gradOutput->GetViewFormat());
  CHECK_RET(gradInput != nullptr, nullptr);
  if (IsAiCoreSupport(gradOutput)) {
    return TanhGradAICore(gradOutput, output, gradInput, executor);
  } else {
    return TanhGradAICpu(gradOutput, output, gradInput, executor);
  }
}
}  // namespace l0op