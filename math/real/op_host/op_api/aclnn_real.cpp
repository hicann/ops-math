/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_real.h"
#include "real.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_COMPLEX32, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128
};

static const std::initializer_list<DataType> OTHER_DTYPE_SUPPORT_LIST = {
  DataType::DT_COMPLEX64, DataType::DT_COMPLEX128
};

static const std::initializer_list<std::pair<DataType, DataType>> DTYPE_SUPPORT_LIST = {
    {DataType::DT_COMPLEX64, DataType::DT_FLOAT}, {DataType::DT_COMPLEX32, DataType::DT_FLOAT16},
    {DataType::DT_COMPLEX128, DataType::DT_DOUBLE}, {DataType::DT_FLOAT16, DataType::DT_FLOAT16},
    {DataType::DT_FLOAT, DataType::DT_FLOAT}};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return OTHER_DTYPE_SUPPORT_LIST;
  }
}

static bool SpecialDtypeTensorNeedPassthrough(const aclTensor *self) {
  if (self->GetDataType() == DataType::DT_FLOAT ||
      self->GetDataType() == DataType::DT_FLOAT16) {
    return true;
  }

  return false;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor *self) {
  // 检查self的数据类型是否在支持列表内
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  // 所有算子的维度都不能超过8
  OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);

  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  return true;
}

static bool CheckInOutDtypeMatch(const aclTensor *self, const aclTensor *out) {
  auto found = std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(),
                std::pair<DataType, DataType>(self->GetDataType(), out->GetDataType()));
  if (found != DTYPE_SUPPORT_LIST.end()) {
      return true;
  }
  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when dtype of input is [%s] output is [%s],the param is invalid.",
          ToString(self->GetDataType()).GetString(), ToString(out->GetDataType()).GetString());
  return false;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入输出数据类型是否匹配
  CHECK_RET(CheckInOutDtypeMatch(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnRealGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnReal, DFX_IN(self), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转换
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // FLOAT和FLOAT16类型，直接拷贝透传
  if (SpecialDtypeTensorNeedPassthrough(selfContiguous)) {
    auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return (viewCopyResult != nullptr) ? ACLNN_SUCCESS : ACLNN_ERR_INNER_NULLPTR;
  }

  // 调用l0算子Real进行计算
  auto realResult = l0op::Real(selfContiguous, uniqueExecutor.get());
  CHECK_RET(realResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(realResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnReal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnReal);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
