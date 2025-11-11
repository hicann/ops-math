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

#include "aclnn_asin.h"
#include "asin.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "common/level2_base.h"
#include "common/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8,  DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_BOOL,   DataType::DT_UINT8, DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_OUT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_V100 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_BOOL,   DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_OUT_LIST_V100 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE};

static bool isSupportBf16Version()
{
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
            return true;
        default: {
            OP_LOGD("The soc version [%s] does not support BFloat16 with input", op::ToString(socVersion).GetString());
            return false;
        }
    }
}

static const std::initializer_list<DataType>& GetSelfRefDtypeList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return DTYPE_OUT_LIST;
    } else {
        return DTYPE_OUT_LIST_V100;
    }
}

// 1个输入1个输出的参数校验
static aclnnStatus CheckParams1In1Out(
    const aclTensor* self, const aclTensor* out, const std::initializer_list<op::DataType>& dtypeSupportList,
    const std::initializer_list<op::DataType>& dtypeOutList)
{
    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid1In1Out(self, out, dtypeSupportList, dtypeOutList), ACLNN_ERR_PARAM_INVALID);
    // 检查输入的数据的值是否合理
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckInplaceDtypeValid(aclTensor* selfRef)
{
    auto inplaceSupportList = GetSelfRefDtypeList();
    // 检查selfRef的数据类型是否在inplace asin算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckInplaceParams(aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace asin算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecAsinGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_NOT_NULL(self, out);
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查输入的数据类型是否在算子的支持列表内
    std::initializer_list<op::DataType> dtypeSupportList =
        (isSupportBf16Version()) ? DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_V100;
    std::initializer_list<op::DataType> dtypeOutList = (isSupportBf16Version()) ? DTYPE_OUT_LIST : DTYPE_OUT_LIST_V100;
    // 参数检查
    auto ret = CheckParams1In1Out(self, out, dtypeSupportList, dtypeOutList);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用cast算子将不支持的类型转化为float / 将bfloat16类型转化为float
    auto castDtype = selfContiguous->GetDataType();
    if (!CheckType(castDtype, DTYPE_OUT_LIST) || castDtype == DataType::DT_BF16) {
        castDtype = DataType::DT_FLOAT;
    }

    auto selfCast = l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行L0算子
    auto acosOutRet = l0op::Asin(selfCast, uniqueExecutor.get());
    CHECK_RET(acosOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(acosOutRet, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAsinGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAsin, DFX_IN(self), DFX_OUT(out));
    return ExecAsinGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAsinGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceAsin, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckInplaceParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecAsinGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnAsin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnAsin);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAsin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceAsin);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
