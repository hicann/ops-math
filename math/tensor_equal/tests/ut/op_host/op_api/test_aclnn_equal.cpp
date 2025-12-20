/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file tensor_equal.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "aclnn_equal.h"

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
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_float) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_int32) {
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_int8) {
  auto tensor_self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_uint8) {
  auto tensor_self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_bool) {
  auto tensor_self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_double) {
  auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_int64) {
  auto tensor_self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_int16) {
  auto tensor_self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_uint16) {
  auto tensor_self = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_uint32) {
  auto tensor_self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_uint64) {
  auto tensor_self = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
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
  ut.TestPrecision();
}

// 正常场景 false
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal02) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 8, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 FORMAT不相等 true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal03) {
  auto tensor_self = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11, 3, 6, 3, 4, 9, 6, 7, 11, 3, 6});

  auto tensor_other = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11, 3, 6, 3, 4, 9, 6, 7, 11, 3, 6});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 FORMAT不相等 true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal04) {
  auto tensor_self = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11, 3, 6, 3, 4, 9, 6, 7, 11, 3, 6});

  auto tensor_other = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11, 3, 6, 3, 4, 9, 6, 7, 11, 3, 6});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}


// 正常场景 self空   other非空
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal05) {

  auto tensor_self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2,1,3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,2,1});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}


// 正常场景 self空   other空 -> true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal06) {

  auto tensor_self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2,1,3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,2,1});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 size不相等   false
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal07) {

  auto tensor_self = TensorDesc({2,2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3});

  auto tensor_other = TensorDesc({2,1,3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,2,1});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                         .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 size相等  ND,NCHW不等  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal08) {

  auto tensor_self = TensorDesc({2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,1,2,4,3,1,2,4,3,1,2,4,3});

  auto tensor_other = TensorDesc({2,2,2,2}, ACL_FLOAT, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,1,2,4,3,1,2,4,3,1,2,4,3});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                         .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 size相等  ND,NHWC不等  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal09) {

  auto tensor_self = TensorDesc({2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,1,2,4,3,1,2,4,3,1,2,4,3});

  auto tensor_other = TensorDesc({2,2,2,2}, ACL_FLOAT, ACL_FORMAT_NHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,1,2,4,3,1,2,4,3,1,2,4,3});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                         .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 size相等  NCHW,NHWC不等  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal10) {

  auto tensor_self = TensorDesc({2,2,2,2}, ACL_FLOAT, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,1,2,4,3,1,2,4,3,1,2,4,3});

  auto tensor_other = TensorDesc({2,2,2,2}, ACL_FLOAT, ACL_FORMAT_NHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,1,2,4,3,1,2,4,3,1,2,4,3});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                         .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景ACL_FLOAT16  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal11) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景ACL_INT32  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal12) {
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景ACL_INT8  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal13) {
  auto tensor_self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景ACL_UINT8  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal14) {
  auto tensor_self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<uint8_t>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<uint8_t>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景ACL_BOOL  true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal15) {
  auto tensor_self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<bool>{false, false, false, true, true, true});

  auto tensor_other = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<bool>{false, false, false, true, true, true});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}


 // 正常场景ACL_INT16  需要走CPU，期望true
 TEST_F(l2_tensor_equal_test, test_tensor_equal_normal16) {
   auto tensor_self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND)
                          .ValueRange(-10, 10)
                          .Value(vector<int16_t>{3, 4, 9, 6, 7, 11});

   auto tensor_other = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND)
                          .ValueRange(-10, 10)
                          .Value(vector<int16_t>{3, 4, 9, 6, 7, 11});

   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                           .Value(vector<bool>{false});

   auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

   // SAMPLE: precision simulate
   ut.TestPrecision();
 }



// 入参为self为null场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_self_is_null) {
  auto tensor_self = nullptr;

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  EXPECT_EQ(workspace_size, 5UL);
}

// 入参为other为null场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_other_is_null) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 5, 6, 7, 8});

  auto tensor_other = nullptr;

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  EXPECT_EQ(workspace_size, 5UL);
}

// 入参out为空指针的场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_input_out_is_null) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = nullptr;

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  EXPECT_EQ(workspace_size, 5UL);
}




// 入参self数据类型不支持的场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_self_dtype_invalid) {
  auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 5, 6, 7, 8});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}

// 入参oter数据类型不支持的场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_other_dtype_invalid) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 5, 6, 7, 8});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}

// 入参数据类型不一致的场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_input_dtype_not_equal) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// out数据类型不是bool的场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_out_dtype_not_bool) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_input_shape_greater_than_8) {
  auto tensor_self = TensorDesc({2,2,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2,2,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5u);
}

// complex64 不支持
TEST_F(l2_tensor_equal_test, test_tensor_equal_not_support_complex64) {
  auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true, false, false, false, false, false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bf16 不支持
TEST_F(l2_tensor_equal_test, test_tensor_equal_not_support_bfloat16) {
  auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 10, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 9, 7, 11});

  auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true, false, false, false, false, false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//非连续

TEST_F(l2_tensor_equal_test, test_tensor_equal_to_contiguous_false) {
  auto tensor_self=TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1,2}, 0,{4, 2})
    .Value(vector<float>{3, 4, 9, 6, 7, 11, 5, 89});

  auto tensor_other=TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0,{4, 2})
    .Value(vector<float>{3, 4, 8, 6, 5, 11, 5, 80});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut= OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspaceSize=0;
  aclnnStatus getWorkspaceResult=ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult,ACLNN_SUCCESS);
  // ut.TestPrecision();  // comment bcz of timeout in model tests (181469 ms)
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_to_contiguous_true) {
  auto tensor_self=TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1,2}, 0,{4, 2})
    .Value(vector<float>{3, 4, 9, 6, 7, 11, 5, 89});

  auto tensor_other=TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0,{4, 2})
    .Value(vector<float>{3, 4, 9, 6, 5, 11, 5, 89});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut= OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspaceSize=0;
  aclnnStatus getWorkspaceResult=ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult,ACLNN_SUCCESS);
  ut.TestPrecision();
}

//数据范围[-1,1]
TEST_F(l2_tensor_equal_test, test_tensor_equal_date_range_f1_1_false) {
  auto tensor_self=TensorDesc({2,3},ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange( -1,1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});

  auto tensor_other=TensorDesc({2,3},ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange(-1,1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.94537017e-01,9.64608252e-01,9.09211235e-01,-9.51533616e-01});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut=OP_API_UT(aclnnEqual,INPUT( tensor_self, tensor_other),OUTPUT( out_tensor_desc));

  uint64_t workspaceSize=0;
  aclnnStatus getWorkspaceResult =ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult,ACLNN_SUCCESS);

  ut.TestPrecision();
}

//数据范围[-1,1]
TEST_F(l2_tensor_equal_test, test_tensor_equal_date_range_f1_1_true) {
  auto tensor_self=TensorDesc({2,3},ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange( -1,1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});

  auto tensor_other=TensorDesc({2,3},ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange(-1,1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut=OP_API_UT(aclnnEqual,INPUT( tensor_self, tensor_other),OUTPUT( out_tensor_desc));

  uint64_t workspaceSize=0;
  aclnnStatus getWorkspaceResult =ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult,ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_input_empty) {
  auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_dimnum_9) {
  auto tensor_self = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}