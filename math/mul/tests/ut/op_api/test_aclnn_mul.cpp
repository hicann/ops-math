/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_mul.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_mul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "mul_test TearDown" << endl;
    }
};

TEST_F(l2_mul_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMul, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mul_test, case_001)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_002)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_003)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_004)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_005)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_006)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_007)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_008)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
