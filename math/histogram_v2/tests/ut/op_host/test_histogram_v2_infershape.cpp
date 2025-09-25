/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class HistogramV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HistogramV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HistogramV2Test TearDown" << std::endl;
  }
};

TEST_F(HistogramV2Test, histogram_v2_infer_shape) {
  ge::op::HistogramV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_x = create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto tensor_min = create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto tensor_max = create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_x);
  op.UpdateInputDesc("min", tensor_min);
  op.UpdateInputDesc("max", tensor_max);

  int64_t bins = 100;
  op.SetAttr("bins", bins);

  Runtime2TestParam param{{"bins"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {100};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(HistogramV2Test, histogram_v2_infer_dtype) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("HistogramV2"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("HistogramV2")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType input_min = ge::DT_FLOAT;
    ge::DataType input_max = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_INT32;
    auto context_holder = gert::InferDataTypeContextFaker()
        .IrInputNum(3)
        .NodeIoNum(3,1)
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"bins", ge::AnyValue::CreateFrom<int64_t>(100)}
        })
        .InputDataTypes({&input_ref, &input_min, &input_max})
        .OutputDataTypes({&output_ref})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}