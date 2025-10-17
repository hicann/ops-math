/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_searchsorted.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

class l2_searchsorted_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_searchsorted_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_searchsorted_test TearDown" << endl;
    }
};

TEST_F(l2_searchsorted_test, searchsorted_3_4_input_nullptr)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, nullptr, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_output_nullptr)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_sorted_sequence_invalid_dtype)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_self_invalid_dtype)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_out_invalid_dtype)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_searchsorted_test, searchsorted_sorted_sequence_invalid_shape)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc =
        TensorDesc({3, 4, 2, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_searchsorted_test, searchsorted_self_invalid_shape)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({3, 4, 2, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_searchsorted_test, searchsorted_self_out_shape_not_equal)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({3, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_searchsorted_test, searchsorted_empty_tensor)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_float)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_float16)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_int16)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_int8)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_uint8)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_int32)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_int64)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_out_int64)
{
    const bool out_int32 = false;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_3_4_int64_scalar)
{
    const bool out_int32 = false;
    const bool right = false;

    auto sorted_sequence_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto self_tensor_desc = ScalarDesc(static_cast<int64_t>(1));
    auto out_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorteds, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_searchsorted_test, searchsorted_2_4_6_int64_transpose)
{
    const bool out_int32 = true;
    const bool right = false;

    auto sorted_sequence_tensor_desc =
        TensorDesc({2, 4, 6}, ACL_INT64, ACL_FORMAT_ND, {24, 4, 1}, 0, {2, 4, 6}).ValueRange(0, 2);
    auto self_tensor_desc = TensorDesc({2, 4, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 4, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnSearchSorted, INPUT(sorted_sequence_tensor_desc, self_tensor_desc, out_int32, right, (aclTensor*)nullptr),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}
