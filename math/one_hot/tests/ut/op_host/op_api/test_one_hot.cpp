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
 * \file test_one_hot.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "level2/aclnn_one_hot.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_one_hot_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "l2_one_hot_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "l2_one_hot_test TearDown" << std::endl; }
};

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_1) {
  auto selfDesc = nullptr;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  int64_t numClasses = 11;
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_2) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -2;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, axis), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_3) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = 3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, axis), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_4) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, nullptr, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_5) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, nullptr, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_6) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_7) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_8) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_9) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_10) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_FLOAT, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_11) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);;
  auto outDesc = TensorDesc({2, 2, 11}, ACL_FLOAT, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  int64_t axis = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_output_invalid) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  auto outDesc = nullptr;
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_self_dtype_invalid_1) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_self_dtype_invalid_2) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_self_dtype_not_match) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_num_classes_value_invalid_1) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = -3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_num_classes_value_invalid_2) {
  auto selfDesc = TensorDesc({2, 2, 0}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 0, 0}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 0;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_num_classes_value_invalid_3) {
  auto selfDesc = TensorDesc({2, 2, 0}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 0, 0}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = -1;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_dtype_int32) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  selfDesc.ValueRange(-100, 100);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT32, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_dtype_int64) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(-100, 100);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_dtype_float_1) {
  auto selfDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  selfDesc.ValueRange(-2, 3);
  auto outDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_dtype_float_2) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 20);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_FLOAT, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_dtype_float16_1) {
  auto selfDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  selfDesc.ValueRange(-2, 3);
  auto outDesc = TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_dtype_float16_2) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 20);
  auto outDesc = TensorDesc({2, 2, 11}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_empty) {
  auto selfDesc = TensorDesc({2, 2, 0}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(-100, 100);
  auto outDesc = TensorDesc({2, 2, 0, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_num_classes_zero) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(-100, 100);
  auto outDesc = TensorDesc({2, 2, 0}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 0;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_num_classes_greater_than_self) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(-100, 100);
  auto outDesc = TensorDesc({2, 2, 14}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 14;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_num_classes_less_than_self) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(3, 10);
  auto outDesc = TensorDesc({2, 2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_dim_greater_than_eight) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  selfDesc.ValueRange(0, 10);
  auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 11;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_one_hot_test, ascend910_9589_l2_one_hot_test_uint8_and_int8_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
  selfDesc.ValueRange(3, 10);
  auto outDesc = TensorDesc({2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_one_hot_test, ascend910_9589_l2_one_hot_test_uint8_and_uint8_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
  selfDesc.ValueRange(3, 10);
  auto outDesc = TensorDesc({2, 2, 3}, ACL_UINT8, ACL_FORMAT_ND);
  auto onValue = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND);
  onValue.ValueRange(1,2);
  auto offValue = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND);
  offValue.ValueRange(0,1);
  int64_t numClasses = 3;
  auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}