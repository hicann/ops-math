/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compare_and_bit_pack_graph_infer_internal.h"

namespace ops {
namespace {
constexpr size_t kOutputYIdx = 0;
} // namespace

namespace compare_and_bit_pack_graph_infer_internal {
ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutputYIdx, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}
} // namespace compare_and_bit_pack_graph_infer_internal

static ge::graphStatus InferDataType4CompareAndBitpack(gert::InferDataTypeContext* context)
{
    return compare_and_bit_pack_graph_infer_internal::InferDataType(context);
}

IMPL_OP(CompareAndBitpack).InferDataType(InferDataType4CompareAndBitpack);

} // namespace ops
