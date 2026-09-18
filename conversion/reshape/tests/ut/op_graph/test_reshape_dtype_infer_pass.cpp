/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "gtest/gtest.h"

#include "op_graph/graph_infer_comm_funcs.h"

namespace {
constexpr size_t kInputNum = 2U;
constexpr size_t kOutputNum = 1U;
constexpr int32_t kInputXIdx = 0;
constexpr int32_t kInputShapeIdx = 1;
constexpr int32_t kOutputYIdx = 0;

void CheckInferDataType(ge::DataType xDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Reshape").OpName("Reshape");
    builder.IONum(kInputNum, kOutputNum);
    builder.InputTensorDesc(kInputXIdx, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(kInputShapeIdx, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(kOutputYIdx, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(ops::InferDataTypeOutputSameAsInput(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(kOutputYIdx), xDtype);
}
} // namespace

TEST(ReshapeGraphInferTest, infer_datatype_follows_x)
{
    const std::vector<ge::DataType> xDtypes = {
        ge::DT_FLOAT16,
        ge::DT_FLOAT,
        ge::DT_INT32,
        ge::DT_BOOL,
    };
    for (const auto xDtype : xDtypes) {
        CheckInferDataType(xDtype);
    }
}
