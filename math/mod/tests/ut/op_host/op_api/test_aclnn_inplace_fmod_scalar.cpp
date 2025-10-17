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

#include "level2/aclnn_fmod_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_fmod_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "fmod_scalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "fmod_scalar_test TearDown" << endl;
    }
};

// aclnnInplaceFmodScalar_002:fmod.Scalar_out输入支持DOUBLE
// 走AICPU
TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_002_double)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, -1).Precision(0.0001, 0.0001);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// aclnnInplaceFmodScalar_003:fmod.Scalar_out支持输入为空Tensor
TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_003_input_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceFmodScalar_004:fmod.Scalar_out支持输入为非连续Tensor
TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_004_input_not_contiguous)
{
    auto tensor_self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-10, -1);
    auto scalar_other = ScalarDesc(1.2f);

    auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// aclnnInplaceFmodScalar_005：fmod.Scalar_out空指针场景测试
TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_005_null_pointer)
{
    auto scalar_other = ScalarDesc(1.2f);

    auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT((aclTensor*)nullptr, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 其他用例
// 入参为other为null场景
TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_006_test_lt_scalar_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(tensor_self, nullptr), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}