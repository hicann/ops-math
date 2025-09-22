/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kl_div.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "../../../../conversion/fill/op_host/op_api/fill.h"
#include "../../../../conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "kl_div.h"
#include "../../../real_div/op_host/op_api/realdiv.h"
#include "../../../../conversion/broadcast_to/op_host/op_api/broadcast_to.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

enum Reduction
{
    None = 0,
    Mean = 1,
    Sum = 2,
    End
};

static std::string REDUCTION_NONE = "none";
static std::string REDUCTION_MEAN = "batchmean";
static std::string REDUCTION_SUM = "sum";

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                        op::DataType::DT_FLOAT16,
                                                                                        op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* target, const aclTensor* out) {
  // 检查输入输出是否是空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* target, const aclTensor* out) {
  const auto& supportList = GetDtypeSupportList();
  // 检查输入输出数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* target, const aclTensor* out) {
  // 检查self和target能否做数据类型推导
  op::DataType promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
  if (promoteType == op::DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and target dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(target->GetDataType()).GetString());
    return false;
  }
  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* target) {
  // 输入的维度必须小于等于8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target, aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, target, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和target能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, target, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输出输出shape, 双输入shape能否做broadcast
  CHECK_RET(CheckShape(self, target), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static std::string &GetReductionStr(int64_t reduction) {
  if (reduction == Mean) {
    return REDUCTION_MEAN;
  } else if (reduction == Sum) {
    return REDUCTION_SUM;
  }
  return REDUCTION_NONE;
}

aclnnStatus FillScalar(aclTensor *out, float val, aclOpExecutor *executor) {
  // 输入为空，输出为0维tensor，使用fill实现
  FVector<int64_t> shape;
  shape.push_back(1);
  auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
  CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
  CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  FVector<float> valVector = {val};
  auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
  CHECK_RET(valTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
  CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnKlDivGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction,
                                       bool logTarget, aclTensor* out, uint64_t* workspaceSize,
                                       aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnKlDiv, DFX_IN(self, target, reduction, logTarget), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, target, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || target->IsEmpty()) {
    // 根据实际支持情况补充
    if (reduction == Mean) {
      ret = FillScalar(out, NAN, uniqueExecutor.get());
      CHECK_RET(ret == ACLNN_SUCCESS, ret);
      // 固定写法，获取计算过程中需要使用的workspace大小
      *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    } else if (reduction == Sum) {
      ret = FillScalar(out, 0, uniqueExecutor.get());
      CHECK_RET(ret == ACLNN_SUCCESS, ret);
      // 固定写法，获取计算过程中需要使用的workspace大小
      *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    } else {
      *workspaceSize = 0;
    }
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入target转换成连续的tensor
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入target的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将0D tensor转化为1D
  auto selfDimNum = self->GetViewShape().GetDimNum();
  auto targetDimNum = target->GetViewShape().GetDimNum();
  auto selfUnsqueeze = selfCasted;
  auto targetUnsqueeze = targetCasted;
  const int64_t dim = 0;
  if (selfDimNum == 0) {
    selfUnsqueeze = l0op::UnsqueezeNd(selfCasted, dim, uniqueExecutor.get());
    CHECK_RET(selfUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (targetDimNum == 0) {
    targetUnsqueeze = l0op::UnsqueezeNd(targetCasted, dim, uniqueExecutor.get());
    CHECK_RET(targetUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 检查self和target能否做broadcast
  op::Shape broadcastShape;
  auto selfBroadcast = selfUnsqueeze;
  auto targetBroadcast = targetUnsqueeze;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfUnsqueeze, targetUnsqueeze, broadcastShape,
                                     return ACLNN_ERR_PARAM_INVALID);
  if (reduction != Mean && reduction != Sum) {
    // reduction为none的场景，检查out和broadcast的shape是否一致
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape,
                                                return ACLNN_ERR_PARAM_INVALID);
  }
  op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
  auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
  selfBroadcast = l0op::BroadcastTo(selfUnsqueeze, broadcastShapeArray, uniqueExecutor.get());
  CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  targetBroadcast = l0op::BroadcastTo(targetUnsqueeze, broadcastShapeArray, uniqueExecutor.get());
  CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行计算
  auto klRes = l0op::KlDiv(selfBroadcast, targetBroadcast, GetReductionStr(reduction), logTarget, uniqueExecutor.get());
  CHECK_RET(klRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(CheckReduceOutShape(klRes, out), ACLNN_ERR_PARAM_INVALID);

  auto klResCasted = l0op::Cast(klRes, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(klResCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* res = nullptr;
  if (reduction == Mean && selfDimNum > 1) {
    const op::Shape selfShape = selfBroadcast->GetViewShape();
    int64_t cnt = 1;
    for (size_t i = 1; i < selfDimNum; i++) {  // 算子仅作batchmean运算，l2层除以其他维度，以还原mean语义
      cnt *= selfShape.GetDim(i);
    }

    auto cntTensor = uniqueExecutor->ConvertToTensor(uniqueExecutor->AllocScalar(cnt), klResCasted->GetDataType());
    CHECK_RET(cntTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    res = l0op::RealDiv(klResCasted, cntTensor, uniqueExecutor.get());
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    res = klResCasted;
  }

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(res, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnKlDiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnKlDiv);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
