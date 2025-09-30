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
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_min_dim.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

class l2_argmin_with_value_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_argmin_with_value_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_argmin_with_value_test TearDown" << endl;
    }
};

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_negtive_nd)
{
    const int64_t dim = -2;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_0_float_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indices_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_4_float_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indices_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nd_checkdim)
{
    const int64_t dim = 5;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_6_float16_nchw_0_nchw)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 6}, ACL_INT64, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_2_4_10_float16_nhwc_1_nhwc)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({2, 4, 10}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({2, 10}, ACL_INT64, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({2, 10}, ACL_FLOAT16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_float_hwcn_1_hwcn)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_HWCN).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_double_ndhwc_0_ndhwc)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_DOUBLE, ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_DOUBLE, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_int8_ncdhw_1_ncdhw)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_INT8, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_int16_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_INT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_int32_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_int64_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, ascend910B2_aclnnMinDim_5_4_2_int64_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, ascend310P_aclnnMinDim_5_4_2_int64_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_uint8_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_bool_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// Ascend910 不支持bf16
TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_bfloat16_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Ascend910B2 支持bf16
TEST_F(l2_argmin_with_value_test, ascend910B2_aclnnMinDim_3_9_2_bfloat16_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 9, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_complex64_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_complex128_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_5_4_2_string_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({5, 4, 2}, ACL_STRING, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({5, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({5, 2}, ACL_STRING, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_0_4_float_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_9D_float_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({9, 11, 3, 4, 6, 9, 2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nhwc)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nd_error_shape)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nd_self_is_null)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = nullptr;
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nd_out_is_null)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = nullptr;
    auto out_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_4_float_nd_0_nd_indices_is_null)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_tensor_desc = nullptr;

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_argmin_with_value_test, aclnnMinDim_3_5_7_6_float_nd_1_nd_transpose)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc =
        TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {3, 5, 6, 7}).ValueRange(0, 2);
    auto indices_tensor_desc =
        TensorDesc({5, 7, 6}, ACL_INT64, ACL_FORMAT_ND, {42, 1, 7}, 0, {5, 6, 7}).Precision(0.001, 0.001);
    auto out_tensor_desc =
        TensorDesc({5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {42, 1, 7}, 0, {5, 6, 7}).Precision(0.001, 0.001);

    auto ut =
        OP_API_UT(aclnnMinDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}