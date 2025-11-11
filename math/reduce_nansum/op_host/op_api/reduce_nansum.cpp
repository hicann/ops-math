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
#include "reduce_nansum.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceNansum);

static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // 获取芯片类型，判断是910还是910B
    return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ReduceNansumOpAiCore(
    const aclTensor* self, const aclTensor* dim, bool keepDim, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ReduceNansumOpAiCore, self, dim, keepDim, out);
    ADD_TO_LAUNCHER_LIST_AICORE(ReduceNansum, OP_INPUT(self, dim), OP_OUTPUT(out), OP_ATTR(keepDim));
    return out;
}

static const aclTensor* GenerateDimTensor(const aclTensor* self, const aclIntArray* dim, aclOpExecutor* executor)
{
    if (dim->Size() == 0) {
        FVector<int64_t> dimVector;
        for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
            dimVector.emplace_back(i);
        }
        return executor->ConvertToTensor(dimVector.data(), dimVector.size(), DataType::DT_INT64);
    }
    return executor->ConvertToTensor(dim, DataType::DT_INT64);
}

const aclTensor* ReduceNansum(const aclTensor* self, const aclIntArray* dim, bool keepDim, aclOpExecutor* executor)
{
    // self如果为标量，不调用算子，直接返回
    if (self->GetViewShape().GetDimNum() == 0) {
        return self;
    }
    auto out = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dimTensor = GenerateDimTensor(self, dim, executor);
    INFER_SHAPE(ReduceNansum, OP_INPUT(self, dimTensor), OP_OUTPUT(out), OP_ATTR(keepDim));

    if (IsAiCoreSupport(self)) {
        return ReduceNansumOpAiCore(self, dimTensor, keepDim, out, executor);
    }
    return out;
}
} // namespace l0op
