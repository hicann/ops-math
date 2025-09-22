/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "level2/aclnn_add.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_add_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "add_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "add_test TearDown" << endl;
  }
};

TEST_F(l2_add_test, case_1) {
  auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(tensor_desc, tensor_desc, scalar_desc), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_add_test, case_nullptr) {
  auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut =
      OP_API_UT(aclnnAdd, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 计算图三
TEST_F(l2_add_test, case_001) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_add_test, Ascend910B2_case_002) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 100);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 100);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);
  auto scalar_desc = ScalarDesc(1.2f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_add_test, case_003) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_add_test, case_004) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

/*TEST_F(l2_add_test, case_005) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}*/

TEST_F(l2_add_test, case_006) {
  auto self_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_add_test, case_007) {
  auto self_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// BOOL场景
TEST_F(l2_add_test, case_008) {
  auto self_tensor_desc =
      TensorDesc({1, 2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, true, false});
  auto other_tensor_desc =
      TensorDesc({1, 2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, true, false, false, false});
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// TEST_F(l2_add_test, case_009) {
//   auto self_tensor_desc = TensorDesc({6, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
//   auto other_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_BF16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
//   auto out_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_BF16, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);
//   auto scalar_desc = ScalarDesc(static_cast<int64_t>(1));

//   auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

/*TEST_F(l2_add_test, case_0010) {
  auto self_tensor_desc = TensorDesc({6, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_DOUBLE, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_DOUBLE, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(2.4f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}*/

// COMPLEX暂时不支持
TEST_F(l2_add_test, case_011) {
  auto self_tensor_desc = TensorDesc({6, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_COMPLEX64, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_COMPLEX64, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(1.2f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// COMPLEX暂时不支持
TEST_F(l2_add_test, case_012) {
  auto self_tensor_desc = TensorDesc({6, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_desc = ScalarDesc(1.2f);
  auto out_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_COMPLEX128, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(1.2f);

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// COMPLEX暂时不支持
TEST_F(l2_add_test, case_fp32_complex64) {
  auto self_tensor_desc = TensorDesc({6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  float src_value = 1.0f;
  aclScalar *other_desc = aclCreateScalar(&src_value, aclDataType::ACL_COMPLEX64);
  auto out_tensor_desc = TensorDesc({2, 5, 6, 3}, ACL_COMPLEX64, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(1.2f);

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_test, case_013) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_add_test, case_014) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_add_test, case_015) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(2.5f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision(); // comment bcz of timeout in model tests (270107 ms)
}

// 空tensor
TEST_F(l2_add_test, case_16) {
  auto self_tensor_desc = TensorDesc({7, 0, 6}, ACL_INT32, ACL_FORMAT_NHWC);
  auto other_tensor_desc = TensorDesc({7, 1, 6}, ACL_INT32, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_test, case_017) {
  auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_add_test, case_019) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(1.2f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_add_test, case_020) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAdd, INPUT(tensor_desc, tensor_desc, nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto scalar_desc = ScalarDesc(5.9f);
  auto ut_2 = OP_API_UT(aclnnAdd, INPUT(nullptr, tensor_desc, scalar_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnAdd, INPUT(tensor_desc, nullptr, scalar_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_4 = OP_API_UT(aclnnAdd, INPUT(tensor_desc, tensor_desc, scalar_desc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_add_test, case_021) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));

  auto ut = OP_API_UT(aclnnAdd, INPUT(tensor_desc, tensor_desc, scalar_desc), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_add_test, case_022) {
  auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(5.9f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  self_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  out_tensor_desc = TensorDesc({10, 5, 10, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2).Precision(0.0001, 0.0001);
  scalar_desc = ScalarDesc(5.9f);

  auto ut_2 = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType
TEST_F(l2_add_test, case_026) {
  auto self_tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(5.9f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  self_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  other_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  out_tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.0001, 0.0001);
  scalar_desc = ScalarDesc(5.9f);

  auto ut_2 = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_test, case_027) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
  auto out_desc = TensorDesc({10, 1}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAdds, INPUT(tensor_desc, scalar_desc, scalar_desc), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnAdds
TEST_F(l2_add_test, case_029) {
  auto self_tensor_desc =
      TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, true, false});
  auto other_desc = ScalarDesc(static_cast<int64_t>(2));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull
TEST_F(l2_add_test, case_030) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto scalar_desc = ScalarDesc(5.9f);

  auto ut = OP_API_UT(aclnnAdds, INPUT(tensor_desc, scalar_desc, nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_2 = OP_API_UT(aclnnAdds, INPUT(nullptr, scalar_desc, scalar_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnAdds, INPUT(tensor_desc, nullptr, scalar_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_4 = OP_API_UT(aclnnAdds, INPUT(tensor_desc, scalar_desc, scalar_desc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// rank invalid
TEST_F(l2_add_test, case_031) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(1.2f);

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, scalar_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnAdds BOOL场景
TEST_F(l2_add_test, case_032) {
  auto self_tensor_desc =
      TensorDesc({1, 2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, true, false});
  auto other_desc = ScalarDesc(false);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// aclnnAdds empty tensor
TEST_F(l2_add_test, case_033) {
  auto self_tensor_desc = TensorDesc({1, 2, 0, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto other_desc = ScalarDesc(static_cast<int64_t>(2));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// aclnnAdds mul + add
TEST_F(l2_add_test, case_034) {
  auto self_tensor_desc = TensorDesc({1, 2, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto other_desc = ScalarDesc(static_cast<int64_t>(2));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(2));

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// aclnnAdds mul + add
TEST_F(l2_add_test, Ascend910B2_case_034) {
  auto self_tensor_desc = TensorDesc({1, 2, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto other_desc = ScalarDesc(static_cast<double>(2.0));
  auto out_tensor_desc = TensorDesc({1, 2, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto scalar_desc = ScalarDesc(static_cast<double>(2.0));

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// aclnnAdd, fp16+fp32
TEST_F(l2_add_test, Ascend910B2_case_fp16_fp32_mix_dtype) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  /*
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
  */
}

// aclnnAdd, fp32+fp16
TEST_F(l2_add_test, Ascend910B2_case_fp32_fp16_mix_dtype) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();

}

// aclnnAdd, bf16+fp32
TEST_F(l2_add_test, Ascend910B2_case_bf16_fp32_mix_dtype) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  /*
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
  */
}

// aclnnAdd, fp32+bf16
TEST_F(l2_add_test, Ascend910B2_case_fp32_bf16_mix_dtype) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  /*
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
  */
}

// // Ascend910_93
// TEST_F(l2_add_test, Ascend910_93_case_1) {
//   auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
//   auto scalar_desc = ScalarDesc(1.0f);

//   auto ut = OP_API_UT(aclnnAdd, INPUT(tensor_desc, tensor_desc, scalar_desc), OUTPUT(tensor_desc));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // NOTE: actually precision test will be automatically skipped on Ascend910_93.
//   ut.TestPrecision();
// }

// Ascend910_93
TEST_F(l2_add_test, Ascend910_93_case_027) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
  auto out_desc = TensorDesc({10, 1}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAdds, INPUT(tensor_desc, scalar_desc, scalar_desc), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_add_test, Ascend910_9589_case_004) {
//   auto self_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//   auto other_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//   auto out_tensor_desc = TensorDesc(self_tensor_desc);
//   auto scalar_desc = ScalarDesc(2.0f);

//   auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_add_test, Ascend910_9589_case_005) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(static_cast<int32_t>(2.0f));

  auto ut = OP_API_UT(aclnnAdd, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// TEST_F(l2_add_test, Ascend910_9589_case_006) {
//   auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
//   auto other_desc = ScalarDesc(1.0f);
//   auto out_tensor_desc = TensorDesc(self_tensor_desc);
//   auto scalar_desc = ScalarDesc(2.0f);

//   auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_add_test, Ascend910_9589_case_007) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_desc = ScalarDesc(static_cast<int32_t>(2.0f));
  auto out_tensor_desc = TensorDesc(self_tensor_desc);
  auto scalar_desc = ScalarDesc(static_cast<int32_t>(2.0f));

  auto ut = OP_API_UT(aclnnAdds, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// aclnnAdds bool + bool
TEST_F(l2_add_test, Ascend910B2_case_bool_bool) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto scalar_desc = ScalarDesc(static_cast<bool>(1));
  auto out_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAdds, INPUT(tensor_desc, scalar_desc, scalar_desc), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}