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

#include "right_shift.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RightShift);

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,  op::DataType::DT_INT16,  op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64};

bool IsSupported(const aclTensor* x, const aclTensor* y)
{
    if (!op::CheckType(x->GetDataType(), DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (!op::CheckType(y->GetDataType(), DTYPE_SUPPORT_LIST)) {
        return false;
    }
    return true;
}

const aclTensor* RightShift(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    auto z = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    CHECK_RET(z != nullptr, nullptr);
    if (!IsSupported(x, y)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input dtype for rightshift is not supported。");
        return nullptr;
    }

    L0_DFX(RightShift, x, y, z);

    static internal::AicpuTaskSpace space("RightShift");
    ADD_TO_LAUNCHER_LIST_AICPU(RightShift, OP_ATTR_NAMES(), OP_INPUT(x, y), OP_OUTPUT(z));
    return z;
}

} // namespace l0op