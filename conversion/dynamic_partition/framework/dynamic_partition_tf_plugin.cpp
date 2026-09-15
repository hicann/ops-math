/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"

namespace domi {
namespace {
constexpr int64_t Y_OUTPUT_PORT_NAME_LEN = 1;
constexpr int64_t NUM_PARTITIONS_OUTPUT_ATTR_NAME_LEN = 14;
} // namespace

static Status DynamicPartitionMapping(const ge::Operator& op_src, ge::Operator& op)
{
    std::vector<DynamicInputOutputInfo> value;
    DynamicInputOutputInfo output(kOutput, "y", Y_OUTPUT_PORT_NAME_LEN, "num_partitions",
                                  NUM_PARTITIONS_OUTPUT_ATTR_NAME_LEN);
    value.push_back(output);
    (void)AutoMappingByOpFnDynamic(op_src, op, value);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicPartition")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicPartition")
    .ParseParamsByOperatorFn(DynamicPartitionMapping)
    .ImplyType(ImplyType::TVM);
} // namespace domi
