/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include "exe_graph/runtime/compute_node_info.h" // AnchorInstanceInfo
#include "graph/types.h"                         // ge::DataType
#include "op_common/log/log.h"                   // OP_LOGE / OP_LOGI / OP_CHECK_NULL_WITH_CONTEXT

// Use GE namespace for graphStatus and GRAPH_SUCCESS.
using namespace ge;

namespace ops {
namespace {

constexpr size_t IR_INDEX_VALUES = 0U;

// 12-dtype whitelist membership — set-style containment over a constexpr
// member list (the ge::DataType values are sparse, so no interval check is
// possible; membership cannot be derived from GetSizeByDataType > 0).
inline bool IsSupportedDtype(const ge::DataType dt)
{
    constexpr ge::DataType kWhitelist[] = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_INT8,
                                           ge::DT_INT16,  ge::DT_INT32,   ge::DT_INT64,  ge::DT_UINT8,
                                           ge::DT_UINT16, ge::DT_UINT32,  ge::DT_UINT64, ge::DT_BOOL};
    return std::find(std::begin(kWhitelist), std::end(kWhitelist), dt) != std::end(kWhitelist);
}

const char* const
    DTYPE_WHITELIST_TEXT = "float32/float16/bfloat16/int8/int16/int32/int64/uint8/uint16/uint32/uint64/bool";

} // namespace

static ge::graphStatus InferDataTypeForParallelConcat(gert::InferDataTypeContext* context)
{
    // Null defense (null_input) — the macro covers the context itself.
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "InferDataType: enter");
    const gert::AnchorInstanceInfo* instInfo = context->GetIrInputInstanceInfo(IR_INDEX_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context, instInfo);
    const size_t inputCount = instInfo->GetInstanceNum(); // len(values)
    if (inputCount == 0UL) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "values (dynamic input instances)");
        return ge::GRAPH_FAILED; // null_input
    }

    // Instance 0 dtype: whitelist gate + propagation source.
    const ge::DataType dtype0 = context->GetDynamicInputDataType(IR_INDEX_VALUES, 0);
    if (!IsSupportedDtype(dtype0)) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "values[0]", std::to_string(static_cast<int>(dtype0)),
                                         DTYPE_WHITELIST_TEXT);
        return ge::GRAPH_FAILED; // dtype_not_supported
    }
    // Cross-instance consistency: all N instances must share one dtype (no
    // promotion); any mismatch is rejected before the output is written.
    for (size_t i = 1; i < inputCount; ++i) {
        const ge::DataType dt = context->GetDynamicInputDataType(IR_INDEX_VALUES, i);
        if (!IsSupportedDtype(dt)) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "values[" + std::to_string(i) + "]",
                                             std::to_string(static_cast<int>(dt)), DTYPE_WHITELIST_TEXT);
            return ge::GRAPH_FAILED; // dtype_not_supported
        }
        if (dt != dtype0) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "values", std::to_string(static_cast<int>(dt)),
                                             "identical to values[0] (no promotion)");
            return ge::GRAPH_FAILED; // dtype_not_supported
        }
    }

    // Set output 0 (output_data) only after all validation has succeeded;
    // SetOutputDataType's status is checked (a failed write must not report
    // success).
    if (context->SetOutputDataType(0, dtype0) != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "InferDataType: SetOutputDataType(0, %d) failed!", static_cast<int>(dtype0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ParallelConcat).InferDataType(InferDataTypeForParallelConcat);

} // namespace ops
