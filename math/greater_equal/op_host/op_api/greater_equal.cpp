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
#include "greater_equal.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GreaterEqual);

// 仅1971支持DT_BF16
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16,  op::DataType::DT_INT64};

// 610lite支持类型
static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, op::DataType::DT_INT8, op::DataType::DT_UINT8};

// 910_95支持类型
static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_BF16,
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), ASCEND910_95_DTYPE_SUPPORT_LIST);
    }
    if (socVersion == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    // 只需要判断dtype
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* GreaterEqual(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(GreaterEqual, self, other);

    op::Shape outShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "InferShape %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto out = executor->AllocTensor(outShape, op::DataType::DT_BOOL);
    auto ret = ACL_SUCCESS;

    if (IsAiCoreSupport(self)) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(GreaterEqual, OP_INPUT(self, other), OP_OUTPUT(out));
        OP_CHECK(
            ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterEqual AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
    } else {
        static internal::AicpuTaskSpace space("GreaterEqual");
        ret = ADD_TO_LAUNCHER_LIST_AICPU(GreaterEqual, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
        OP_CHECK(
            ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterEqual AiCPU ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    }
    return out;
}
} // namespace l0op
