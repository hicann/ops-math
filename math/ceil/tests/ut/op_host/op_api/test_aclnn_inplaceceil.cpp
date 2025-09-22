/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "level2/aclnn_ceil.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2InplaceCeilTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "l2InplaceCeilTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2InplaceCeilTest TearDown" << std::endl;
  }
};

// 异常场景：self的数据类型不在支持范围内，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_self_int8) {
  auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}



// 异常场景：空tensor，返回ACLNN_SUCCESS
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_empty_tensor) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 异常场景：入参为nullptr，返回ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_nullptr) {
  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT((aclTensor*)nullptr), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 正常场景：float16、4维，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceCeilTest, l2_ceil_test_fp16) {
  auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 正常场景：float32、2维，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceCeilTest, l2_ceil_test_fp32) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 异常场景：ACL_INT32、2维，ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_int32) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：ACL_UINT8、2维，ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_uint8) {
  auto selfDesc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：ACL_INT16、2维，ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_int16) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：ACL_INT64、2维，ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_int64) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：ACL_BOOL、2维，ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_bool) {
  auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：double、4维、aicpu，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceCeilTest, l2_ceil_test_double) {
  auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常场景：double、4维、ACL_FORMAT_NCHW、aicpu，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceCeilTest, l2_ceil_test_double_nchw) {
  auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常场景：double、4维、ACL_FORMAT_NHWC、aicpu，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceCeilTest, l2_ceil_test_double_nhwc) {
  auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 异常场景：double、4维、ACL_FORMAT_NC1HWC0、aicpu，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_double_nc1hwc0) {
  auto selfDesc = TensorDesc({2, 3, 4, 5, 16}, ACL_DOUBLE, ACL_FORMAT_NC1HWC0);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// Test for non-continuous
 TEST_F(l2InplaceCeilTest, l2_ceil_test_non_continuous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4,5}).ValueRange(-2, 2);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 异常场景：double、9维、返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceCeilTest, l2_ceil_test_err_long_shape) {
  auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceCeil, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}