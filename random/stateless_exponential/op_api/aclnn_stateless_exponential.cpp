/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_stateless_exponential.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "stateless_exponential.h"
#include "conversion/fill/op_api/fill.h"
#include "math/zero_op/op_api/zero_op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "math/add/op_api/add.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t ONE = 1;
static constexpr size_t MAX_DIM_LEN = 8;
static const float FLT_MIN = 1.1754943508222875e-38f;

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SEED_AND_OFFSET_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64};

static inline bool CheckNotNullWithTensor(const aclTensor* self, const aclTensor* seedTensor,
                                          const aclTensor* offsetTensor)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(seedTensor, return false);
    OP_CHECK_NULL(offsetTensor, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* seedTensor, const aclTensor* offsetTensor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(seedTensor, SEED_AND_OFFSET_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(offsetTensor, SEED_AND_OFFSET_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckLambdability(double lambd)
{
    if (lambd <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of lambd has to be greater than 0, but current is %f.", lambd);
        return false;
    }
    return true;
}

static inline bool CheckDim(const aclTensor* self, const aclTensor* seedTensor, const aclTensor* offsetTensor)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    auto seedShape = seedTensor->GetViewShape();
    auto offsetShape = offsetTensor->GetViewShape();

    if (seedShape.GetDimNum() != ONE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of seedTensor must be 1.");
        return false;
    }
    if (offsetShape.GetDimNum() != ONE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of offsetTensor must be 1.");
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParamsWithTensor(const aclTensor* self, const aclTensor* seedTensor,
                                                const aclTensor* offsetTensor, double lambd)
{
    CHECK_RET(CheckNotNullWithTensor(self, seedTensor, offsetTensor), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, seedTensor, offsetTensor), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckLambdability(lambd), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDim(self, seedTensor, offsetTensor), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnStatelessExponentialTensorGetWorkspaceSize(aclTensor* self, const aclTensor* seedTensor,
                                                                      const aclTensor* offsetTensor, int64_t offset,
                                                                      double lambd, uint64_t* workspaceSize,
                                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnStatelessExponentialTensor, DFX_IN(self, seedTensor, offsetTensor, offset, lambd),
                   DFX_OUT(self));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsWithTensor(self, seedTensor, offsetTensor, lambd);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    FVector<int64_t> offsetVector{static_cast<int64_t>(offset)};
    aclIntArray* offsetList = uniqueExecutor.get()->AllocIntArray(offsetVector.data(), offsetVector.size());
    auto tmpTensor = uniqueExecutor.get()->ConvertToTensor(offsetList, op::DataType::DT_INT64);
    auto offset1Tensor = l0op::Add(offsetTensor, tmpTensor, uniqueExecutor.get());
    CHECK_RET(offset1Tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::StatelessExponential(selfContiguous, seedTensor, offset1Tensor, lambd, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, self, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnStatelessExponentialTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStatelessExponentialTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
