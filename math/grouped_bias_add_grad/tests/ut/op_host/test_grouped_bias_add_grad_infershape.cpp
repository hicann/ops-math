/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "reduce_ops.h"         // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class GroupedBiasAddGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GroupedBiasAddGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GroupedBiasAddGrad TearDown" << std::endl;
  }
};

TEST_F(GroupedBiasAddGrad, GroupedBiasAddGrad_infershape_case_0) {
  ge::op::GroupedBiasAddGrad op;
  op.UpdateInputDesc("grad_y", create_desc({20, 128, 6912}, ge::DT_FLOAT16));
  op.UpdateInputDesc("group_idx", create_desc({6912}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

TEST_F(GroupedBiasAddGrad, GroupedBiasAddGrad_infershape_case_1) {
  ge::op::GroupedBiasAddGrad op;
  op.UpdateInputDesc("grad_y", create_desc({20, 6912}, ge::DT_FLOAT16));
  op.UpdateInputDesc("group_idx", create_desc({6912, 40}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

TEST_F(GroupedBiasAddGrad, GroupedBiasAddGrad_infershape_case_2) {
  ge::op::GroupedBiasAddGrad op;
  op.UpdateInputDesc("grad_y", create_desc({40, 6912}, ge::DT_FLOAT16));
  op.UpdateInputDesc("group_idx", create_desc({10}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
  auto grad_bias = op.GetOutputDesc(0);
  std::vector<int64_t> expected_shape = {10, 6912};
  EXPECT_EQ(grad_bias.GetShape().GetDims(), expected_shape);
}

TEST_F(GroupedBiasAddGrad, GroupedBiasAddGrad_infershape_case_3) {
  ge::op::GroupedBiasAddGrad op;
  op.UpdateInputDesc("grad_y", create_desc({40, 6912}, ge::DT_FLOAT16));
  op.UpdateInputDesc("group_idx", create_desc({1024}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
  auto grad_bias = op.GetOutputDesc(0);
  std::vector<int64_t> expected_shape = {1024, 6912};
  EXPECT_EQ(grad_bias.GetShape().GetDims(), expected_shape);
}

TEST_F(GroupedBiasAddGrad, GroupedBiasAddGrad_infershape_case_4) {
  ge::op::GroupedBiasAddGrad op;
  op.UpdateInputDesc("grad_y", create_desc({40, 6912}, ge::DT_FLOAT16));
  op.UpdateInputDesc("group_idx", create_desc({2049}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

