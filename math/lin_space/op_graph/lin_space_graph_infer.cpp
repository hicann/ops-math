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
 * \file lin_space_graph_infer.cpp
 * \brief lin_space operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t LINSPACE_IDX_IN_START = 0;
static constexpr size_t LINSPACE_IDX_OUT_OUTPUT = 0;

static graphStatus InferDataType4LinSpace(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4LinSpace enter");

    auto startDtype = context->GetInputDataType(LINSPACE_IDX_IN_START);
    bool validStartDtype = ((startDtype != ge::DT_UNDEFINED) && (startDtype != ge::DT_MAX));
    if (validStartDtype) {
        context->SetOutputDataType(LINSPACE_IDX_OUT_OUTPUT, startDtype);
    } else {
        context->SetOutputDataType(LINSPACE_IDX_OUT_OUTPUT, ge::DT_FLOAT);
    }

    OP_LOGD(context->GetNodeName(), "InferDataType4LinSpace end");

    return GRAPH_SUCCESS;
}
IMPL_OP(LinSpace).InferDataType(InferDataType4LinSpace);

}; // namespace ops