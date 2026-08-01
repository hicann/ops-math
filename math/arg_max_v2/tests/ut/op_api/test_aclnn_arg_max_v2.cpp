/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_argmax.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_arg_max_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "arg_max_v2_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "arg_max_v2_test TearDown" << endl; }
};

// test FLOAT with INT32 output
TEST_F(l2_arg_max_v2_test, case_arg_max_v2_float_int32)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test FLOAT with INT64 output
TEST_F(l2_arg_max_v2_test, case_arg_max_v2_float_int64)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test FLOAT16
TEST_F(l2_arg_max_v2_test, case_arg_max_v2_float16)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test keepdim=true
TEST_F(l2_arg_max_v2_test, case_arg_max_v2_keepdim)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3, 1}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = true;
    auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 1d tensor
TEST_F(l2_arg_max_v2_test, case_arg_max_v2_1d)
{
    auto self_tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 0;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test nullptr
TEST_F(l2_arg_max_v2_test, case_arg_max_v2_nullptr)
{
    int64_t dim = 0;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMax, INPUT((aclTensor*)nullptr, dim, keepdim), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
