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
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../../op_host/op_api/aclnn_logical_not.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_logical_not_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "logical_not_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "logical_not_test TearDown" << endl;
    }
};

// 测试所有支持的类型
TEST_F(l2_logical_not_test, case_1_dtype_all_support)
{
    vector<aclDataType> dtype_list{ACL_UINT8,   ACL_INT8,   ACL_INT16, ACL_INT32,     ACL_INT64,     ACL_FLOAT,
                                   ACL_FLOAT16, ACL_DOUBLE, ACL_BOOL,  ACL_COMPLEX64, ACL_COMPLEX128};
    for (auto dtype : dtype_list) {
        cout << "+++++++++++++++++++++++ start to test dtype: " << String(dtype) << endl;
        auto input_tensor_desc = TensorDesc({3, 5}, dtype, ACL_FORMAT_ND)
                                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        auto out_tensor_desc =
            TensorDesc({3, 5}, dtype, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
        auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        if (dtype == ACL_COMPLEX128 || dtype == ACL_COMPLEX64) {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(aclRet, ACL_SUCCESS);
            ut.TestPrecision();
        }
    }
}

TEST_F(l2_logical_not_test, case_2_different_dtype)
{
    auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc =
        TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_logical_not_test, case_3_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                  ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        cout << "+++++++++++++++++++++++ start to test format: " << format << endl;
        auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT32, format)
                                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        auto out_tensor_desc =
            TensorDesc({3, 5}, ACL_INT32, format).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
        auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    }
}

TEST_F(l2_logical_not_test, case_4_different_format)
{
    auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc =
        TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_NCHW).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_logical_not_test, case_5_different_shape)
{
    auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc =
        TensorDesc({2, 5}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logical_not_test, case_6_nullptr)
{
    auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc =
        TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    auto ut1 = OP_API_UT(aclnnLogicalNot, INPUT(nullptr), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(nullptr));
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_logical_not_test, case_7_empty)
{
    auto input_tensor_desc = TensorDesc({0, 3, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({0, 3, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_logical_not_test, case_8_non_contiguous)
{
    auto input_tensor_desc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND, {5, 1}, 0, {5, 5});
    auto out_tensor_desc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND, {5, 1}, 0, {5, 5});
    auto ut = OP_API_UT(aclnnLogicalNot, INPUT(input_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_logical_not_test, case_9_inplace)
{
    auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto ut = OP_API_UT(aclnnInplaceLogicalNot, INPUT(input_tensor_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}