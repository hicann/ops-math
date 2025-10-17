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
#include "level2/aclnn_ones.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_ones_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_ones_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_ones_test TearDown" << std::endl;
    }
};

// self为空tensor
TEST_F(l2_ones_test, l2_ones_test_self_empty)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 用例不支持，complex64
TEST_F(l2_ones_test, l2_ones_test_dtype_complex64)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，complex128
TEST_F(l2_ones_test, l2_ones_test_dtype_complex128)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试bfloat16
TEST_F(l2_ones_test, l2_ones_test_dtype_bfloat16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    } else {
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
}

// 用例不支持，shape=9
TEST_F(l2_ones_test, l2_ones_test_shape9)
{
    auto selfDesc = TensorDesc({2, 3, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float16，aicore
TEST_F(l2_ones_test, l2_ones_test_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，float32，aicore
TEST_F(l2_ones_test, l2_ones_test_dtype_float32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，int8，aicore
TEST_F(l2_ones_test, l2_ones_test_dtype_int8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，int32，aicore
TEST_F(l2_ones_test, l2_ones_test_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，uint8，aicore
TEST_F(l2_ones_test, l2_ones_test_dtype_uint8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，double，aicpu
TEST_F(l2_ones_test, l2_ones_test_dtype_double)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，int16，aicpu
TEST_F(l2_ones_test, l2_ones_test_dtype_int16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，int64，aicpu
TEST_F(l2_ones_test, l2_ones_test_dtype_int64)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，bool，aicpu
TEST_F(l2_ones_test, l2_ones_test_dtype_bool)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceOne, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}
