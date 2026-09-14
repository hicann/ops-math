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
static Status AutoMappingFnHistogramFixedWidth(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        return FAILED;
    }
    ge::DataType dataType;
    if (op.GetAttr("dtype", dataType) != ge::GRAPH_SUCCESS) {
        return FAILED;
    }
    (void)op.SetAttr("dtype", static_cast<int>(dataType));
    return SUCCESS;
}

REGISTER_CUSTOM_OP("HistogramFixedWidth")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HistogramFixedWidth")
    .ParseParamsByOperatorFn(AutoMappingFnHistogramFixedWidth)
    .ImplyType(ImplyType::TVM);
} // namespace domi
