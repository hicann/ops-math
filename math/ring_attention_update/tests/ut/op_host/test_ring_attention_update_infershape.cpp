/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
 /*!
 * \file test_ring_attention_update_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "all_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class RingAttentionUpdate : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RingAttentionUpdate SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RingAttentionUpdate TearDown" << std::endl;
  }
};

TEST_F(RingAttentionUpdate, RingAttentionUpdate_infershape_test_0) {
  ge::op::RingAttentionUpdate op;
  auto tensor_desc_prev_attn_out = create_desc_shape_range(
    {-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {1024, 2, 384}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_prev_softmax_max = create_desc_shape_range(
    {-1, -1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_prev_softmax_sum = create_desc_shape_range(
    {-1, -1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_cur_attn_out = create_desc_shape_range(
    {-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {1024, 2, 384}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_cur_softmax_max = create_desc_shape_range(
    {-1, -1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_cur_softmax_sum = create_desc_shape_range(
    {-1, -1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );

  op.UpdateInputDesc("prev_attn_out", tensor_desc_prev_attn_out);
  op.UpdateInputDesc("prev_softmax_max", tensor_desc_prev_softmax_max);
  op.UpdateInputDesc("prev_softmax_sum", tensor_desc_prev_softmax_sum);
  op.UpdateInputDesc("cur_attn_out", tensor_desc_cur_attn_out);
  op.UpdateInputDesc("cur_softmax_max", tensor_desc_cur_softmax_max);
  op.UpdateInputDesc("cur_softmax_sum", tensor_desc_cur_softmax_sum);
  op.SetAttr("input_layout", "SBH");
  std::vector<int64_t> expected_attn_out = {-1, -1, -1};
  std::vector<int64_t> expected_softmax_max = {-1, -1, -1, -1};
  std::vector<int64_t> expected_softmax_sum = {-1, -1, -1, -1};
  // runtime 2.0
  Runtime2TestParam rt_param;
  EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
  auto attn_out_desc = op.GetOutputDesc(0);
  auto softmax_max_desc = op.GetOutputDesc(1);
  auto softmax_sum_desc = op.GetOutputDesc(2);
  EXPECT_EQ(attn_out_desc.GetShape().GetDims(), expected_attn_out);
  EXPECT_EQ(softmax_max_desc.GetShape().GetDims(), expected_softmax_max);
  EXPECT_EQ(softmax_sum_desc.GetShape().GetDims(), expected_softmax_sum);
}

TEST_F(RingAttentionUpdate, RingAttentionUpdate_infershape_test_1) {
  ge::op::RingAttentionUpdate op;
  auto tensor_desc_prev_attn_out = create_desc_shape_range(
    {1024, 2, 384}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {1024, 2, 384}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_prev_softmax_max = create_desc_shape_range(
    {2, 3, 1024, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_prev_softmax_sum = create_desc_shape_range(
    {2, 3, 1024, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_cur_attn_out = create_desc_shape_range(
    {1024, 2, 384}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {1024, 2, 384}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_cur_softmax_max = create_desc_shape_range(
    {2, 3, 1024, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );
  auto tensor_desc_cur_softmax_sum = create_desc_shape_range(
    {2, 3, 1024, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
    {2, 3, 1024, 8}, ge::FORMAT_ND, {{1, -1}, {1, -1}, {1, -1}, {1, -1}}
  );

  op.UpdateInputDesc("prev_attn_out", tensor_desc_prev_attn_out);
  op.UpdateInputDesc("prev_softmax_max", tensor_desc_prev_softmax_max);
  op.UpdateInputDesc("prev_softmax_sum", tensor_desc_prev_softmax_sum);
  op.UpdateInputDesc("cur_attn_out", tensor_desc_cur_attn_out);
  op.UpdateInputDesc("cur_softmax_max", tensor_desc_cur_softmax_max);
  op.UpdateInputDesc("cur_softmax_sum", tensor_desc_cur_softmax_sum);
  op.SetAttr("input_layout", "SBH");
  std::vector<int64_t> expected_attn_out = {1024, 2, 384};
  std::vector<int64_t> expected_softmax_max = {2, 3, 1024, 8};
  std::vector<int64_t> expected_softmax_sum = {2, 3, 1024, 8};
  // runtime 2.0
  Runtime2TestParam rt_param;
  EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);
  auto attn_out_desc = op.GetOutputDesc(0);
  auto softmax_max_desc = op.GetOutputDesc(1);
  auto softmax_sum_desc = op.GetOutputDesc(2);
  EXPECT_EQ(attn_out_desc.GetShape().GetDims(), expected_attn_out);
  EXPECT_EQ(softmax_max_desc.GetShape().GetDims(), expected_softmax_max);
  EXPECT_EQ(softmax_sum_desc.GetShape().GetDims(), expected_softmax_sum);
}

TEST_F(RingAttentionUpdate, RingAttentionUpdate_inferdtype_test_0) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RingAttentionUpdate"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RingAttentionUpdate")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType prev_attn_out_ref = ge::DT_FLOAT16;
    ge::DataType prev_softmax_max_ref = ge::DT_FLOAT;
    ge::DataType prev_softmax_sum_ref = ge::DT_FLOAT;
    ge::DataType cur_attn_out_ref = ge::DT_FLOAT16;
    ge::DataType cur_softmax_max_ref = ge::DT_FLOAT;
    ge::DataType cur_softmax_sum_ref = ge::DT_FLOAT;

    ge::DataType attn_out_ref = ge::DT_FLOAT16;
    ge::DataType softmax_max_ref = ge::DT_FLOAT;
    ge::DataType softmax_sum_ref = ge::DT_FLOAT;

    auto context_holder = gert::InferDataTypeContextFaker()
        .NodeIoNum(6, 3)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&prev_attn_out_ref, &prev_softmax_max_ref, &prev_softmax_sum_ref, &cur_attn_out_ref, &cur_softmax_max_ref, &cur_softmax_sum_ref})
        .OutputDataTypes({&attn_out_ref, &softmax_max_ref, &softmax_sum_ref})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetInputDataType(0), prev_attn_out_ref);
    EXPECT_EQ(context->GetInputDataType(1), prev_softmax_max_ref);
    EXPECT_EQ(context->GetInputDataType(2), prev_softmax_sum_ref);
    EXPECT_EQ(context->GetInputDataType(3), cur_attn_out_ref);
    EXPECT_EQ(context->GetInputDataType(4), cur_softmax_max_ref);
    EXPECT_EQ(context->GetInputDataType(5), cur_softmax_sum_ref);

    EXPECT_EQ(context->GetOutputDataType(0), attn_out_ref);
    EXPECT_EQ(context->GetOutputDataType(1), softmax_max_ref);
    EXPECT_EQ(context->GetOutputDataType(2), softmax_sum_ref);
  }
}