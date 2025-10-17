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

#include "aclnn_exp.h"
#include "exp.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "common/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_FLOAT,      DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_COMPLEX64,
    DataType::DT_COMPLEX128, DataType::DT_BOOL,    DataType::DT_INT64};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_BF16, DataType::DT_DOUBLE,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BOOL, DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND910_DTYPE_SELFREF_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SELFREF_LIST = {
    DataType::DT_FLOAT,  DataType::DT_FLOAT16,   DataType::DT_BF16,
    DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 获取芯片类型,判断是1971还是1980
    bool is910BSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95);
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_CURRENT =
        is910BSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;

    // 检查数据类型是否在Exp算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_CURRENT, return false);

    DataType resultDtype = self->GetDataType();
    if (self->GetDataType() == DataType::DT_BOOL || self->GetDataType() == DataType::DT_INT64) {
        resultDtype = DataType::DT_FLOAT;
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(resultDtype, out->GetDataType(), return false);

    return true;
}

static bool CheckInplaceDtypeValid(aclTensor* selfRef)
{
    auto inplaceSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SELFREF_LIST, ASCEND910_DTYPE_SELFREF_LIST);
    // 检查selfRef的数据类型是否在inplace exp算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckParamsExp(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsExp(aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace exp算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsExp(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果self是空tensor，则out也是空tensor，直接返回
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的Tensor
    auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castSelf = contiguousSelf;
    if (self->GetDataType() == DataType::DT_BOOL || self->GetDataType() == DataType::DT_INT64) {
        castSelf = l0op::Cast(contiguousSelf, DataType::DT_FLOAT, uniqueExecutor.get());
    }

    // 调用Exp算子kernel
    auto expOut = l0op::Exp(castSelf, uniqueExecutor.get());
    CHECK_RET(expOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(expOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnExp, DFX_IN(self), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, out, workspaceSize, executor);
}

aclnnStatus aclnnExp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnExp);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceExpGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceExp, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto ret = CheckInplaceParamsExp(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return GetWorkspaceSizeCommon(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceExp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceExp);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
