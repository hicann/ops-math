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

#include "aclnn_fmod_tensor.h"
#include "mod.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT8, op::DataType::DT_FLOAT16,
    op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT8,  op::DataType::DT_FLOAT16,
    op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_INT64, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        socVersion == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 检查self的数据类型是否在Mod算子的支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查other的数据类型是否在Mod算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

    // 检查out的数据类型是否在Mod算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* other)
{
    // 检查self和other能否做数据类型推导
    op::DataType promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }
    OP_LOGI(
        "Self dtype %s and other dtype %s need promote dtype to %s", op::ToString(self->GetDataType()).GetString(),
        op::ToString(other->GetDataType()).GetString(), op::ToString(promoteType).GetString());
    auto supportList = GetDtypeSupportList();
    if (!CheckType(promoteType, supportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Self dtype %s and other dtype %s need promote dtype to %s, but now not support promote dtype %s.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString(),
            op::ToString(promoteType).GetString(), op::ToString(promoteType).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和other能否做数据类型推导
    CHECK_RET(CheckPromoteType(self, other), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查双输入是否能broadcast
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Mod算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Mod算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 由于经过cast转换，需要将mod的输入再次校验
    CHECK_RET(CheckDtypeValid(selfCasted, otherCasted, out), ACLNN_ERR_PARAM_INVALID);

    // 调用Mod算子kernel
    auto fmodOpOut = l0op::Mod(selfCasted, otherCasted, uniqueExecutor.get());
    CHECK_RET(fmodOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(fmodOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto view_copy_result = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFmodTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFmodTensor, DFX_IN(self, other), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnFmodTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFmodTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceFmodTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFmodTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    return GetWorkspaceSizeCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceFmodTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFmodTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
