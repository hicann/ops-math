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
#include "../../../../op_host/op_api/aclnn_ne_scalar.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_inplace_ne_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "inplace_ne_scalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "inplace_ne_scalar_test TearDown" << endl;
    }
};

// 正常场景 int8
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_int8)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 uint8
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_uint8)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 int16
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_int16)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 int32
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_int32)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);
    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 int64
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_int64)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_complex64)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_complex128)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 float16
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_float16)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 float
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_float)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 double
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_double)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景 bf16
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_not_support_bf16)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景910B SocVersion bf16
TEST_F(l2_inplace_ne_scalar_test, ascend910B2_support_bf16_910B)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 bool
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_bool)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, true, false, true, false, false});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 format_not_equal   HWCN
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_format_notequal)
{
    auto tensor_self_ref =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 ACL_FORMAT_NCDHW
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_NCDHW)
{
    auto tensor_self_ref = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                               .ValueRange(-10, 10)
                               .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 ACL_FORMAT_NDHWC
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_NDHWC)
{
    auto tensor_self_ref = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                               .ValueRange(-10, 10)
                               .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 ACL_FORMAT_HWCN
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_HWCN)
{
    auto tensor_self_ref = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_HWCN)
                               .ValueRange(-10, 10)
                               .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 ACL_FORMAT_NHWC
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_NHWC)
{
    auto tensor_self_ref = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC)
                               .ValueRange(-10, 10)
                               .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 ACL_FORMAT_NCHW
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_support_NCHW)
{
    auto tensor_self_ref = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW)
                               .ValueRange(-10, 10)
                               .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_emptytensor)
{
    auto tensor_self_ref =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 入参为self为null场景
TEST_F(l2_inplace_ne_scalar_test, test_inplace_ne_scalar_nullptr)
{
    auto tensor_self_ref = nullptr;
    auto scalar_ohter = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(tensor_self_ref, scalar_ohter), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非连续测试
TEST_F(l2_inplace_ne_scalar_test, case_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);
    auto scalar_desc = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnInplaceNeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
