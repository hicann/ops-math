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

#include "level2/aclnn_trunc.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_inplacetrunc_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Inplace Trunc Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Inplace Trunc Test TearDown" << std::endl; }
};


TEST_F(l2_inplacetrunc_test, case_shape1D)
{
    auto tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplacetrunc_test, case_shape_2D)
{
    auto tensor_desc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplacetrunc_test, case_shape_3D)
{
    auto tensor_desc = TensorDesc({3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplacetrunc_test, case_shape_4D)
{
    auto tensor_desc = TensorDesc({3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplacetrunc_test, case_shape_5D)
{
    auto tensor_desc = TensorDesc({3,3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplacetrunc_test, case_shape_8D)
{
    auto tensor_desc = TensorDesc({3,3,3,3,3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2)
        .Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inplacetrunc_test, case_invalid_shape_9D)
{
    auto tensor_desc = TensorDesc({3,3,3,3,3,3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto ut = OP_API_UT(aclnnInplaceTrunc, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
