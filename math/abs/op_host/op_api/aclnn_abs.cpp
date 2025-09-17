/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_abs.h"
#include "abs.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT64, DataType::DT_INT32,
    DataType::DT_INT16,  DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT64, DataType::DT_INT32,
    DataType::DT_INT16,  DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_BOOL,  DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool HasEmptyTensor(const aclTensor* self)
{
    // 检查张量是否存在空维
    if (self->IsEmpty()) {
        return true;
    }

    return false;
}

static bool SpecialDtypeTensorNeedPassthrough(const aclTensor* self)
{
    if (self->GetDataType() == DataType::DT_UINT8 || self->GetDataType() == DataType::DT_BOOL) {
        return true;
    }

    return false;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    // 检查self的数据类型是否在支持列表内，out类型随动检查
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    // 输入输出的格式需要一致
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal. self [%s], out [%s].",
            ToString(self->GetStorageShape()).GetString(), ToString(out->GetStorageShape()).GetString());
        return false;
    }

    // self格式不能是私有格式
    if (IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // 所有算子的维度都不能超过8
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);

    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查诗句格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAbsGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAbs, DFX_IN(self), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (HasEmptyTensor(self)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转换
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // UINT8和BOOL类型，直接拷贝透传
    if (SpecialDtypeTensorNeedPassthrough(selfContiguous)) {
        auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return (viewCopyResult != nullptr) ? ACLNN_SUCCESS : ACLNN_ERR_INNER_NULLPTR;
    }

    // 调用l0算子Abs进行计算
    auto absResult = l0op::Abs(selfContiguous, uniqueExecutor.get());
    CHECK_RET(absResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(absResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAbs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAbs);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif