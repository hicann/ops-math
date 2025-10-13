/**
 * Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License");
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_d_graph_infer.cpp
 * \brief lin_space_d operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t IDX_OUT_OUTPUT = 0;

static graphStatus InferDataTypeLinSpaceD(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataTypeLinSpaceD enter");
    context->SetOutputDataType(IDX_OUT_OUTPUT, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "InferDataTypeLinSpaceD end");
    return GRAPH_SUCCESS;
}
IMPL_OP(LinSpaceD).InferDataType(InferDataTypeLinSpaceD);

}; // namespace ops