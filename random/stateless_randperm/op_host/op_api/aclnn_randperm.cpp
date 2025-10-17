/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_randperm.h"
#include "stateless_randperm.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT64, op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_INT16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_DOUBLE, op::DataType::DT_INT64,   op::DataType::DT_INT8,
    op::DataType::DT_UINT8,  op::DataType::DT_INT16,   op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* out, int64_t n)
{
    op::Shape outShape;
    outShape.SetDimNum(1);
    outShape.SetDim(0, n);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* out)
{
    bool isBf16Support =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST =
        isBf16Support ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(int64_t n, aclTensor* out)
{
    CHECK_RET(CheckNotNull(out), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CheckDtypeValid(out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeValid(out, n), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(n >= 0, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRandpermGetWorkspaceSize(
    int64_t n, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRandperm, DFX_IN(n, seed, offset), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(n, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调用算子kernel
    auto executorP = uniqueExecutor.get();
    FVector<int64_t> nVector;
    nVector.push_back(n);
    auto nTensor = executorP->ConvertToTensor(nVector.data(), nVector.size(), op::DataType::DT_INT64);
    FVector<int64_t> seedVector;
    seedVector.push_back(seed);
    auto seedTensor = executorP->ConvertToTensor(seedVector.data(), seedVector.size(), op::DataType::DT_INT64);
    FVector<int64_t> offsetVector;
    offsetVector.push_back(offset);
    auto offsetTensor = executorP->ConvertToTensor(offsetVector.data(), offsetVector.size(), op::DataType::DT_INT64);
    const int64_t layout = 1;
    op::Shape shape({n});
    auto randpermOut = l0op::StatelessRandperm(
        shape, nTensor, seedTensor, offsetTensor, layout, op::DataType::DT_INT64, uniqueExecutor.get());
    CHECK_RET(randpermOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(randpermOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRandperm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRandperm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
