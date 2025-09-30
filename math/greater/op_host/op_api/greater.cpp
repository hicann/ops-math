/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "greater.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Greater);

// 1980
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8};

// 1971
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BF16};

// 610lite支持类型
static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, op::DataType::DT_INT8, op::DataType::DT_UINT8};

// 910_95支持类型
static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_BF16,
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

// 根据dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), ASCEND910_95_DTYPE_SUPPORT_LIST);
    }
    // 获取芯片类型,判断是1971还是1980
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_910B_LIST);
    }
    if (socVersion == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_910_LIST);
}

// AICORE算子kernel
static const aclTensor* GreaterAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* gtOut, aclOpExecutor* executor)
{
    L0_DFX(GreaterAiCore, self, other, gtOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将Greater算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Greater, OP_INPUT(self, other), OP_OUTPUT(gtOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return gtOut;
}

// AICPU算子kernel
static const aclTensor* GreaterAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* gtOut, aclOpExecutor* executor)
{
    L0_DFX(GreaterAiCpu, self, other);

    static internal::AicpuTaskSpace space("Greater");
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICPU，将Greater算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Greater, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(gtOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return gtOut;
}

const aclTensor* Greater(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape outShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto gtOut = executor->AllocTensor(outShape, op::DataType::DT_BOOL, op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
        return GreaterAiCore(self, other, gtOut, executor);
    } else {
        return GreaterAiCpu(self, other, gtOut, executor);
    }
}
} // namespace l0op
