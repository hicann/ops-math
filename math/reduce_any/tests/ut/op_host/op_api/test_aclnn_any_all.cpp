/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "level2/aclnn_any.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_any_all_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "any_all_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "any_all_test TearDown" << endl; }
};

// TEST_F(l2_any_all_test, case_bool) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_f16) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, ascend910B2_case_bf16) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_f32) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_int8) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

TEST_F(l2_any_all_test, case_uint8) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT8, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_any_all_test, case_int16) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// TEST_F(l2_any_all_test, case_int32) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_int64) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

TEST_F(l2_any_all_test, case_empty) {
  auto tensor_desc = TensorDesc({2, 0, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1, 1}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_any_all_test, case_empty_out) {
  auto tensor_desc = TensorDesc({2, 0, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  auto out_tensor_desc = TensorDesc({0, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_any_all_test, case_unsupportdtype) {
  auto tensor_desc = TensorDesc({2, 2, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1, 1}, ACL_INT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_any_all_test, case_unsupportdshape) {
  auto tensor_desc = TensorDesc({2, 2, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_any_all_test, case_null) {
  auto tensor_desc = TensorDesc({2, 2, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut1 = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnAny, INPUT(nullptr, dim_desc, false), OUTPUT(out_tensor_desc));

  aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_any_all_test, case_uncontinue) {
  auto tensor_desc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_NCHW, {1, 4}, 0, {5, 4});
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1});
  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_any_all_test, case_NCHW) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_NHWC) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_HWCN) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_HWCN);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_NDHWC) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_NDHWC);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_NCDHW) {
//   auto tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_NCDHW);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_any_all_test, case_dim_0) {
//   auto tensor_desc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_NCDHW);
//   auto dim_desc = IntArrayDesc(vector<int64_t>{0});
//   auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// 空dim
TEST_F(l2_any_all_test, case_empty_dimt) {
  auto tensor_desc = TensorDesc({2, 0, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{});
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_any_all_test, case_unsupportdim) {
  auto tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT16, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
  auto out_tensor_desc = TensorDesc({1, 1}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAny, INPUT(tensor_desc, dim_desc, false), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}