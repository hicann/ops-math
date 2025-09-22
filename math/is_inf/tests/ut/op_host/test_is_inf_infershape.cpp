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
#include "reduce_ops.h"
#include "math_ops.h"
#include "common/utils/ut_op_common.h"
#include "util/util.h"
#include "array_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

// ----------------IsInf--------------
class is_inf : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "is_inf SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "is_inf TearDown" << std::endl;
    }
};

TEST_F(is_inf, is_inf_infershape_test1) {
ge::op::IsInf op;
op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDescByName("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {3, 4};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(is_inf, is_inf_infershape_test2) {
ge::op::IsInf op;
op.UpdateInputDesc("x", create_desc_with_ori({5, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {5, -1}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDescByName("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {5, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(is_inf, is_inf_infershape_test3) {
ge::op::IsInf op;
op.UpdateInputDesc("x", create_desc_with_ori({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {-2}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDescByName("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {-2};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(is_inf, is_inf_infershape_test4) {
ge::op::IsInf op;
op.UpdateInputDesc("x", create_desc_with_ori({
4, 5, 6, 7, 8, 1, 5, 9}, ge::DT_FLOAT16, ge::FORMAT_ND, {
4, 5, 6, 7, 8, 1, 5, 9}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDescByName("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {4, 5, 6, 7, 8, 1, 5, 9};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

static ge::Operator BuildGraph4InferAxisType(const std::initializer_list<int64_t>& dims1,
                                             const ge::Format& format,
                                             const ge::DataType& dataType) {
  auto apply_op = op::IsInf("IsInf");
  TENSOR_INPUT_WITH_SHAPE(apply_op, x, dims1, dataType, format, {});
  apply_op.InferShapeAndType();
  return apply_op;
}

TEST_F(is_inf, infer_axis_type_case1) {
  auto op1 = BuildGraph4InferAxisType({-1, -1},
                                      ge::FORMAT_ND, ge::DT_FLOAT16);

  ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("IsInf");
  EXPECT_NE(infer_func, nullptr);

  std::vector<ge::AxisTypeInfo> infos;
  const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
  EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);

  std::vector<ge::AxisTypeInfo> expect_infos = {
    ge::AxisTypeInfoBuilder()
      .AxisType(ge::AxisType::ELEMENTWISE)
      .AddInputCutInfo({0, {0}})
      .AddOutputCutInfo({0, {0}})
      .Build(),
    ge::AxisTypeInfoBuilder()
      .AxisType(ge::AxisType::ELEMENTWISE)
      .AddInputCutInfo({0, {1}})
      .AddOutputCutInfo({0, {1}})
      .Build(),
  };

  EXPECT_STREQ(AxisTypeInfoToString(infos).c_str(),
               AxisTypeInfoToString(expect_infos).c_str());
}

TEST_F(is_inf, infer_axis_type_case2) {
  auto op1 = BuildGraph4InferAxisType({-2, },
                                      ge::FORMAT_ND, ge::DT_FLOAT16);

  ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("IsInf");
  EXPECT_NE(infer_func, nullptr);

  std::vector<ge::AxisTypeInfo> infos;
  const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
  EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);

  EXPECT_EQ(infos.size(), 0);
}