/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_equal.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "math/tensor_equal/op_api/aclnn_equal.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_tensor_equal_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "tensor_equal_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "tensor_equal_test TearDown" << endl; }
};

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_float16) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 正常场景 true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal01) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}