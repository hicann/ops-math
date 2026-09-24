/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "../../../op_graph/compare_and_bit_pack_graph_infer_internal.h"

namespace {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
constexpr size_t kInputXIdx = 0;
constexpr size_t kInputThresholdIdx = 1;
constexpr size_t kOutputYIdx = 0;
constexpr char kOpType[] = "CompareAndBitpack";
} // namespace

TEST(CompareAndBitpackGraphInfer, SetsOutputDataTypeToUint8)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType(kOpType).OpName(kOpType).IONum(kInputNum, kOutputNum);
    builder.InputTensorDesc(kInputXIdx, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(kInputThresholdIdx, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(kOutputYIdx, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();

    ASSERT_EQ(ops::compare_and_bit_pack_graph_infer_internal::InferDataType(contextHolder.GetContext()),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext()->GetOutputDataType(kOutputYIdx), ge::DT_UINT8);
}
