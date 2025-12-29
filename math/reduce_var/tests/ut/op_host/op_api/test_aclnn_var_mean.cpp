/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_var_mean.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_var_mean_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_var_mean_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_var_mean_test TearDown" << std::endl;
  }
};

// 正常场景_float32_dim为0_keepdim为true
TEST_F(l2_var_mean_test, var_mean_dtype_float32_dim_0_keepdim_true) {
  auto selfDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{0});
  int64_t correction = 1;
  bool keepdim = true;
  auto varOutDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto meanOutDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnVarMean, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(varOutDesc, meanOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

 // ut.TestPrecision();
}

// 正常场景_float32_dim为1_keepdim为false
TEST_F(l2_var_mean_test, var_mean_dtype_float32_dim_1_keepdim_false) {
  auto selfDesc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{1});
  int64_t correction = 1;
  bool keepdim = false;
  auto varOutDesc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto meanOutDesc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnVarMean, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(varOutDesc, meanOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// 正常场景_float16_dim为2_keepdim为true
TEST_F(l2_var_mean_test, var_mean_dtype_float16_dim_2_keepdim_true) {
  auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{2});
  int64_t correction = 1;
  bool keepdim = true;
  auto varOutDesc = TensorDesc({1, 2, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto meanOutDesc = TensorDesc({1, 2, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnVarMean, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(varOutDesc, meanOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// 正常场景_float32_dim为2_3_keepdim为true
TEST_F(l2_var_mean_test, var_mean_dtype_float32_dim_2_3_keepdim_true) {
  auto selfDesc = TensorDesc({1, 2, 8, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
  int64_t correction = 1;
  bool keepdim = true;
  auto varOutDesc = TensorDesc({1, 2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto meanOutDesc = TensorDesc({1, 2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnVarMean, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(varOutDesc, meanOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}
