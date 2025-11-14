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
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_fill_diagonal.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_inplace_fill_diagonal_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_inplace_fill_diagonal_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_inplace_fill_diagonal_test TearDown" << endl;
    }
};

TEST_F(l2_inplace_fill_diagonal_test, case_normal)
{
    auto tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_inplace_fill_diagonal_test, case_format_nd_normal)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_NCHW, ACL_FORMAT_NHWC, ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    int length = ValidList.size();
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    for (int i = 0; i < length; i++) {
        auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ValidList[i]).ValueRange(-1, 1);
        auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
    }
}

TEST_F(l2_inplace_fill_diagonal_test, case_float_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_inplace_fill_diagonal_test, case_float16_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_inplace_fill_diagonal_test, case_double_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}