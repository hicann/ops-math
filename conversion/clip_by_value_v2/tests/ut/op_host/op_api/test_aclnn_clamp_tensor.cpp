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

#include "aclnn_clamp.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_clamp_tensor_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "clamp_tensor_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "clamp_tensor_test TearDown" << std::endl;
  }
};

TEST_F(l2_clamp_tensor_test, case_uncontinue_self) {
  auto tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 3}, 0, {4, 3}).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_uncontinue_min_max) {
  auto tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 3}, 0, {4, 3}).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 3}, 0, {4, 3}).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_f64) {
  auto tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_int8) {
  auto tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_uint8) {
  auto tensor_desc = TensorDesc({1, 4, 5, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({1, 4, 5, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({1, 4, 5, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({1, 4, 5, 9}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_int16) {
  auto tensor_desc = TensorDesc({1, 6, 7, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({1, 6, 7, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({1, 6, 7, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({1, 6, 7, 9}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_f32) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_int32) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_int64) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_empty) {
  auto tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_all_empty) {
  auto tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_brc_aicore) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 1, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_brc_aicore_max) {
  auto tensor_desc = TensorDesc({2, 3, 4, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 1, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_brc_aicore_all) {
  auto tensor_desc = TensorDesc({2, 3, 4, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 1, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 1, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_brc_aicpu) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 1, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_brc_aicpu_min) {
  auto tensor_desc = TensorDesc({2, 1, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_cannot_brc) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 2, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_tensor_test, case_unsupport_dtype) {
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  // unsupport dtype
  auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
  auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_1 =
      OP_API_UT(aclnnClampTensor, INPUT(tensor_desc_1, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc_1));

  aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);

  // different dtype
  auto tensor_desc_2 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto out_tensor_desc_2 = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_2 =
      OP_API_UT(aclnnClampTensor, INPUT(tensor_desc_2, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc_2));

  aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_tensor_test, case_unsupport_shape) {
  auto min_tensor_desc = TensorDesc({5, 1, 3, 2}, ACL_INT64, ACL_FORMAT_NCHW);
  auto max_tensor_desc = TensorDesc({5, 4, 3, 1}, ACL_INT64, ACL_FORMAT_NCHW);
  uint64_t workspace_size = 0;

  auto tensor_desc_1 = TensorDesc({5, 4, 3, 2}, ACL_INT64, ACL_FORMAT_NCHW);
  auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_1 =
      OP_API_UT(aclnnClampTensor, INPUT(tensor_desc_1, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc_1));

  aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_tensor_test, case_null) {
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  uint64_t workspace_size = 0;

  auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_1 = OP_API_UT(aclnnClampTensor, INPUT(nullptr, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc_1));

  aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_2 = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc_1, min_tensor_desc, max_tensor_desc), OUTPUT(nullptr));

  aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc_1, nullptr, nullptr), OUTPUT(out_tensor_desc_1));

  aclnnStatus aclRet_3 = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_3, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_clamp_tensor_test, case_NCHW) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_NHWC) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_HWCN) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_HWCN).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_HWCN).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_HWCN).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_NDHWC) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_NDHWC).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 1, 5, 6}, ACL_INT64, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 1, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_NCDHW) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(2, 3);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_clamp_tensor_test, case_9dim) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto out_tensor_desc =
      TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_tensor_test, case_min_9dim) {
  auto tensor_desc = TensorDesc({4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto out_tensor_desc = TensorDesc({4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_tensor_test, ascend910_9589_case_min_9dim) {
  auto tensor_desc = TensorDesc({4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto min_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto max_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
  auto out_tensor_desc = TensorDesc({4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnClampTensor, INPUT(tensor_desc, min_tensor_desc, max_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}