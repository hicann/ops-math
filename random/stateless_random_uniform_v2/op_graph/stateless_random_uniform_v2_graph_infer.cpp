/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file stateless_random_uniform_v2_graph_infer.cpp
 * \brief
 */

#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"

namespace {
constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kFirstAttrIdx = 0U;
constexpr size_t kThirdAttrIdx = 2U;
constexpr int64_t kDimValue1 = 1LL;
constexpr int64_t kDimValue2 = 2LL;
constexpr int64_t kDimValue3 = 3LL;
constexpr int64_t kDimValue4 = 4LL;
constexpr int64_t kSeedValue = 2LL;
constexpr int64_t kCounterValue = 2LL;
} // namespace

using namespace ge;

namespace ops {

static graphStatus RandomOpCommonInferDataTypeByAttr(
    gert::InferDataTypeContext* context, const DataType defalut_dtype, const size_t dtype_index,
    const std::set<DataType>& support_dtype, const bool is_required = false)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    OP_LOGI(
        context->GetNodeName(), "RandomOpCommonInferDataTypeByAttr begin. Data type is %s, is_required is %d.",
        Ops::Base::ToString(defalut_dtype).c_str(), is_required);
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    DataType y_dtype = defalut_dtype;
    const int32_t* dst_type_pointer = attrs->GetAttrPointer<int32_t>(dtype_index);
    if (dst_type_pointer != nullptr) {
        y_dtype = static_cast<DataType>(*dst_type_pointer);
        std::string error_msg = ConcatString("Attr dtype not support ", Ops::Base::ToString(y_dtype).c_str());
        OP_CHECK_IF(
            support_dtype.count(y_dtype) == 0, OP_LOGI(context->GetNodeName(), "%s", error_msg.c_str()),
            return GRAPH_FAILED);
    } else {
        if (is_required) {
            std::string error_msg = std::string("Get attr dtype failed, but the attr is required.");
            OP_LOGI(context->GetNodeName(), "%s", error_msg.c_str());
            return GRAPH_FAILED;
        }
    }

    context->SetOutputDataType(kOutputIndex0, y_dtype);
    OP_LOGI(
        context->GetNodeName(), "RandomOpCommonInferDataTypeByAttr end. Data type is %s.",
        Ops::Base::ToString(y_dtype).c_str());
    return GRAPH_SUCCESS;
}

static graphStatus StatelessTruncatedNormalV2InferDataType(gert::InferDataTypeContext* context)
{
    std::set<DataType> support_dtype = {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16};
    return RandomOpCommonInferDataTypeByAttr(context, DT_FLOAT, kFirstAttrIdx, support_dtype, false);
}

IMPL_OP(StatelessRandomUniformV2).InferDataType(StatelessTruncatedNormalV2InferDataType);

} // namespace ops