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

#include "ceil.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(Ceil);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return CheckType(self->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* CeilAiCore(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(CeilAiCore, self, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Ceil, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Ceil ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

static const aclTensor* CeilAiCpu(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(CeilAiCpu, self, out);

    static internal::AicpuTaskSpace space("Ceil");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Ceil, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CeilAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* Ceil(const aclTensor* self, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

    if (IsAiCoreSupport(self)) {
        return CeilAiCore(self, out, executor);
    } else {
        return CeilAiCpu(self, out, executor);
    }
}
} // namespace l0op
