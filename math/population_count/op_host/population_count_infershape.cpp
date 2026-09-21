/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file population_count_infershape.cpp
 * \brief PopulationCount InferShape
 *
 * Semantics:
 *   - y.shape = x.shape (element-wise, no broadcast)
 */

#include "util/shape_util.h"
#include <set>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4PopulationCount(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    // 输入 dtype 白名单校验：以 CANN 预定义错误码 EZ0020（参数 dtype 错误）上报并拦截
    const gert::Tensor* input_tensor = context->GetInputTensor(0);
    if (input_tensor != nullptr) {
        const ge::DataType input_dtype = input_tensor->GetDataType();
        const std::string inputDtypeStr = ge::TypeUtils::DataTypeToSerialString(input_dtype);
        OP_LOGI(context->GetNodeName(), "[InferShape] current input dtype of x = %s", inputDtypeStr.c_str());
        static const std::set<ge::DataType> supportedDtypes = {ge::DT_INT8,   ge::DT_INT16, ge::DT_INT32,
                                                               ge::DT_INT64,  ge::DT_UINT8, ge::DT_UINT16,
                                                               ge::DT_UINT32, ge::DT_UINT64};
        if (supportedDtypes.count(input_dtype) == 0) {
            OP_LOGE(context->GetNodeName(), "PopulationCount: current input dtype of x = %s, which is not supported",
                    inputDtypeStr.c_str());
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "x", inputDtypeStr.c_str(),
                "only integer dtypes are supported; the Ascend950 AICore implementation supports "
                "DT_INT16/DT_UINT16");
            return ge::GRAPH_FAILED;
        }
    }

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // Shape passthrough: y.shape = x.shape
    *output_shape = *input_shape;

    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PopulationCount).InferShape(InferShape4PopulationCount);

} // namespace ops
