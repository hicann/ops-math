/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_log2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_inplace_log2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "inplace_log2_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "inplace_log2_test TearDown" << std::endl;
    }
};

TEST_F(l2_inplace_log2_test, case_001_FLOAT)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_log2_test, case_002_FLOAT16)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision(); // cpu不支持float16, 先注释掉
}

TEST_F(l2_inplace_log2_test, case_003_INT64)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 100);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/* 走aicpu，暂时不支持，先注释掉
TEST_F(l2_inplace_log2_test, case_004_INT32) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 走aicpu，暂时不支持，先注释掉
TEST_F(l2_inplace_log2_test, case_005_INT16) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inplace_log2_test, case_006_INT8) {
  auto self_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inplace_log2_test, case_007_UINT8) {
  auto self_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 走aicpu，暂时不支持，先注释掉
TEST_F(l2_inplace_log2_test, case_008_BOOL) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 走aicpu，暂时不支持，先注释掉
TEST_F(l2_inplace_log2_test, case_009_BF16) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inplace_log2_test, case_0011_COMPLEX64) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}


TEST_F(l2_inplace_log2_test, case_0012_COMPLEX128) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inplace_log2_test, case_0010_DOUBLE) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

走aicpu，暂时不支持，先注释掉
*/

TEST_F(l2_inplace_log2_test, case_013_ND)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_inplace_log2_test, case_14_EMPTY)
{
    auto self_tensor_desc = TensorDesc({7, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_log2_test, case_015_CONTINUOUS)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// dim超出限制
TEST_F(l2_inplace_log2_test, case_016_MAX_DIM)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_inplace_log2_test, case_017_NULLPTR)
{
    aclTensor* self_tensor_desc = nullptr;
    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_inplace_log2_test, case_018_DTYPE)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid bf16
TEST_F(l2_inplace_log2_test, case_018_DTYPE_bf16)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(tensor_desc), OUTPUT());

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        // SAMPLE: precision simulate
        ut.TestPrecision();
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// CheckFormat
TEST_F(l2_inplace_log2_test, case_019_FORMAT)
{
    auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnInplaceLog2, INPUT(self_tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    auto tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto ut_2 = OP_API_UT(aclnnInplaceLog2, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}