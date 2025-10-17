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

#include "aclnn_square.h"
#include "square.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/op_api_def.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910D_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT64, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ASCEND910D_DTYPE_DTYPE_SUPPORT_LIST;
    }
    return ASCEND910D_DTYPE_DTYPE_SUPPORT_LIST;
}

static bool HasEmptyTensor(const aclTensor* self)
{
    // 检查张量是否存在空维
    if (self->IsEmpty()) {
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
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    // 当前level2接口仅支持910D
    OP_CHECK(
        socVersion == SocVersion::ASCEND910_95,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not implemented for %s", op::ToString(socVersion).GetString()), return false);

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

    // self格式只支持ND
    if (IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND.");
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    // shape范围为1~8维
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSquareGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSquare, DFX_IN(self), DFX_OUT(out));

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

    // 调用l0算子Square进行计算
    auto squareResult = l0op::Square(selfContiguous, uniqueExecutor.get());
    CHECK_RET(squareResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(squareResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSquare(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSquare);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif