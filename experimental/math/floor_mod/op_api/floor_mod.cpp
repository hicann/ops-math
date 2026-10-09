/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "floor_mod.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FloorMod);

// Supports BF16, FP16, FP32, INT32, INT64 and DOUBLE.
const aclTensor* FloorModAiCore(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(FloorModAiCore, input, other, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FloorMod, OP_INPUT(input, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FloorModAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

// Kept for compatibility with callers that explicitly select the legacy path.
const aclTensor* FloorModAiCpu(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(FloorModAiCpu, input, other, out);

    static internal::AicpuTaskSpace space("FloorMod", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(FloorMod, OP_ATTR_NAMES(), OP_INPUT(input, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FloorModAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return out;
}

// Use the custom AI Core implementation for every dtype, including FP64.
// The double kernel keeps uint64 storage and uses a bounded vector fast path.
// Precision-sensitive values are recomputed from their original IEEE-754
// significands inside the same device task, without framework CastAiCpu tasks.
const aclTensor* FloorMod(const aclTensor* input, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(FloorMod, input, other);
    op::Shape broadcastShape;
    if (!BroadcastInferShape(input->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(input->GetViewShape()).GetString(),
                op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, input->GetDataType());

    return FloorModAiCore(input, other, out, executor);
}

const aclTensor* FloorMod(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(FloorMod, input, other, out);
    return FloorModAiCore(input, other, out, executor);
}
} // namespace l0op
