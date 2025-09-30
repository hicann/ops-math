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

#include "aclnn_clamp.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_clamp_max_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "inplace_clamp_max_tensor_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "inplace_clamp_max_tensor_test TearDown" << std::endl;
    }
};

TEST_F(l2_inplace_clamp_max_tensor_test, case_float32)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_float16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();  // cpu do not support float16
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_double)
{
    auto tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NDHWC).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 2, 3, 1, 5}, ACL_DOUBLE, ACL_FORMAT_NDHWC).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_uint8)
{
    auto tensor_desc = TensorDesc({1, 4, 5, 9}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 4, 1, 9}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_int8)
{
    auto tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_int16)
{
    auto tensor_desc = TensorDesc({1, 6, 7, 9}, ACL_INT16, ACL_FORMAT_NHWC).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 1, 7, 9}, ACL_INT16, ACL_FORMAT_NHWC).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}
TEST_F(l2_inplace_clamp_max_tensor_test, case_int32)
{
    auto tensor_desc = TensorDesc({1, 2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 1, 4}, ACL_INT32, ACL_FORMAT_HWCN).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_int64)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_int32_float32)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_float16_float32)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_empty)
{
    auto tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_all_empty)
{
    auto tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_in_not_contiguous)
{
    auto tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 3}, 0, {4, 3}).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_in_error_shape)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 3, 4, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_out_error_shape)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(2, 3);

    auto ut = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, max_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_in_error_dtype)
{
    auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT64, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    // unsupport dtype
    auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut_1 = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc_1, max_tensor_desc), OUTPUT());

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);

    // different dtype
    auto tensor_desc_2 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut_2 = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc_2, max_tensor_desc), OUTPUT());

    aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_null)
{
    auto max_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);
    uint64_t workspace_size = 0;

    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut_1 = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(nullptr, max_tensor_desc), OUTPUT());

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, nullptr), OUTPUT());

    aclnnStatus aclRet_3 = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_3, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_inplace_clamp_max_tensor_test, case_input_error_shape_len)
{
    auto tensor_desc = TensorDesc({4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
    auto tensor_desc_9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
    uint64_t workspace_size = 0;

    auto ut_1 = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc_9, tensor_desc), OUTPUT());

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);

    auto ut_2 = OP_API_UT(aclnnInplaceClampMaxTensor, INPUT(tensor_desc, tensor_desc_9), OUTPUT());

    aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_INVALID);
}