/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_sinh.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_sinh_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Sinh Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Sinh Test TearDown" << std::endl; }
};

// 空tensor
TEST_F(l2_sinh_test, case_empty)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull self
TEST_F(l2_sinh_test, case_null_self)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(nullptr), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_sinh_test, case_null_out)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(tensor_desc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDim
TEST_F(l2_sinh_test, case_dim9)
{
    auto self_tensor_desc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_sinh_test, case_differ_shape)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 1, 4, 4},  ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckBFloat16
TEST_F(l2_sinh_test, ascend910B2_case_bf16_bf16)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckDtypeInvalid
TEST_F(l2_sinh_test, case_float_int16)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_INT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_sinh_test, case_float_float)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_sinh_test, case_double_double)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_sinh_test, case_complex64_complex64)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_sinh_test, case_float_double)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_sinh_test, case_int32_float)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_INT32, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// not contiguous
TEST_F(l2_sinh_test, case_discontiguous_float)
{
    auto self_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-20, 20);
    auto out_desc = TensorDesc(self_desc);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// AICPU not contiguous
TEST_F(l2_sinh_test, case_discontiguous_complex64)
{
    auto self_desc = TensorDesc({5, 4}, ACL_COMPLEX64, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-20, 20);
    auto out_desc = TensorDesc(self_desc);
    auto ut = OP_API_UT(aclnnSinh, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// self的数据类型不在支持范围内
TEST_F(l2_sinh_test, l2_inplace_Sinh_test_int64) {
  auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceSinh, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}