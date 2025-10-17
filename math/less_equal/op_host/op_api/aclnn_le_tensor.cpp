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

#include "aclnn_le_tensor.h"
#include "less_equal.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_UINT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_UINT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_UINT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_UINT64};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,     op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,      op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,    op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_95_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,     op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910_95) {
        return ASCEND910_95_DTYPE_SUPPORT_LIST;
    }
    if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static inline const std::initializer_list<op::DataType>& GetOutDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
        return OUT_DTYPE_SUPPORT_910_95_LIST;
    }

    return OUT_DTYPE_SUPPORT_910_LIST;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在LessEqual算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查other的数据类型是否在LessEqual算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

    // 检查out的数据类型是否在LessEqual算子的支持列表内
    op::DataType outType = out->GetDataType();
    auto outSuportList =
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 ? GetOutDtypeSupportList() : supportList;
    if ((!CheckType(outType, outSuportList)) && (outType != DataType::DT_BOOL)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Out dtype %s should be in dtype support list [%s].",
            op::ToString(out->GetDataType()).GetString(), op::ToString(outSuportList).GetString());
        return false;
    }
    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 检查self和other能否做数据类型推导
    op::DataType promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        // 查看promoteType是否在inputList
        auto inputSupportList = GetDtypeSupportList();
        // 检查self的数据类型是否在LessEqual算子的支持列表内
        if (!CheckType(promoteType, inputSupportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "promote dtype %s should be in dtype support list [%s].",
                op::ToString(promoteType).GetString(), op::ToString(inputSupportList).GetString());
            return false;
        }
        // check self和other能否转换为promoteType
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);
    }
    // 检查BOOL类型能否转换为输出的数据类型(算子返回的都是BOOL类型)
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);
    return true;
}

static bool CheckOutShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    const size_t MAX_DIM = 8;
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(other, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM, return false);

    op::Shape outShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, outShape, return false);

    if (outShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "BroadcastShape %s is not equal out's shape %s.",
            op::ToString(outShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查双输入是否能broadcast,检查boradcast后的输出与out是否一致
    CHECK_RET(CheckOutShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnLeTensorCommon(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查三个入参参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 空tensor处理
    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    if (promoteType == DataType::DT_BOOL) {
        promoteType = DataType::DT_FLOAT;
    }
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

    // 调用LessEqual算子kernel
    auto lessEqualOpOut = l0op::LessEqual(selfCasted, otherCasted, uniqueExecutor.get());
    CHECK_RET(lessEqualOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果(BOOL)转换成输出out的数据类型
    auto castOut = l0op::Cast(lessEqualOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLeTensor, DFX_IN(self, other), DFX_OUT(out));
    return aclnnLeTensorCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnLeTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLeTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLeTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceLeTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    return aclnnLeTensorCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceLeTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceLeTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
