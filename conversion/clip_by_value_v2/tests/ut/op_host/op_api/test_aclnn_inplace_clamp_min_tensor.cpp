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

#include "aclnn_clamp.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_clamp_min_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_inplace_clamp_min_tensor_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_inplace_clamp_min_tensor_test TearDown" << endl;
    }
};

TEST_F(l2_inplace_clamp_min_tensor_test, case_f64)
{
    auto tensor_desc = TensorDesc({1, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({1, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_min_tensor_test, case_int16)
{
    auto tensor_desc = TensorDesc({1, 9, 19}, ACL_INT16, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({1, 9, 19}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_min_tensor_test, case_int8)
{
    auto tensor_desc = TensorDesc({1, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({1, 3, 4}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_min_tensor_test, case_uint8)
{
    auto tensor_desc = TensorDesc({1, 19, 20}, ACL_UINT8, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({1, 19, 20}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_min_tensor_test, case_1)
{
    auto tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_min_tensor_test, case_2)
{
    auto tensor_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplace_clamp_min_tensor_test, case_3)
{
    auto tensor_desc = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto min_tensor_desc = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceClampMinTensor, INPUT(tensor_desc, min_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}