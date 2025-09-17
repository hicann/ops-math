/**
 * Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
 * This File is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
 * Please refer to the Licence for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file non_finite_check_graph_infer.cpp
 * \brief non_finite_check operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
const int64_t OUTPUT_IDX = 0;

static graphStatus InferDataTypeForNonFiniteCheck(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForNonFiniteCheck");
    context->SetOutputDataType(OUTPUT_IDX, ge::DT_FLOAT);
    OP_LOGD(context, "End to do InferDataTypeForNonFiniteCheck");
    return GRAPH_SUCCESS;
}

IMPL_OP(NonFiniteCheck).InferDataType(InferDataTypeForNonFiniteCheck);

}; // namespace ops