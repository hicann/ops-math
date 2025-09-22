/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class AngleV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AngleV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AngleV2Test TearDown" << std::endl;
  }
};

TEST_F(AngleV2Test, angle_v2_infer_shape_neg2) {
  ge::op::AngleV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  Runtime2TestParam param;
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AngleV2Test, angle_v2_infer_shape_fp16_normal) {
  ge::op::AngleV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({32, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  Runtime2TestParam param;
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape = {32, 32, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AngleV2Test, angle_v2_infer_shape_fp16_neg1) {
  ge::op::AngleV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({-1, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  Runtime2TestParam param;
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape = {-1, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AngleV2Test, angle_v2_infer_dtype_test) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AngleV2"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AngleV2")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
        .IrInputNum(1)
        .NodeIoNum(1,1)
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&input_ref})
        .OutputDataTypes({&output_ref})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}
