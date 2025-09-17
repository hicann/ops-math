/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "pad_ops.h"             // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class DiagV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DiagV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DiagV2 TearDown" << std::endl;
  }
};

TEST_F(DiagV2, diag_infer_shape_fp16_dim0) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 0);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DiagV2, diag_infer_shape_fp16_dim3) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({32, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 1);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DiagV2, diag_infer_shape_fp16_dim2_1) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({-1, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 0);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DiagV2, diag_infer_shape_fp16_dim2_2) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({32, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 1);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {31};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DiagV2, diag_infer_shape_fp16_dim2_3) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({32, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", -1);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {31};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DiagV2, diag_infer_shape_fp16_dim_4) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 0);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DiagV2, diag_infer_shape_fp16_dim1) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 1);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {33, 33};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DiagV2, diag_infer_datatype_fp16_dim1) {
  ge::op::DiagV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto tensor_desc = create_desc_shape_range({32}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("diagonal", 1);
  Runtime2TestParam param{{"diagonal"}};
  auto ret = InferDataTypeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}