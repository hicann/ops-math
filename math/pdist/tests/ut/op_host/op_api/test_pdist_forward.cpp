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
#include "../../../../op_host/op_api/aclnn_pdist_forward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_pdist_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_pdist_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_pdist_test TearDown" << std::endl;
    }
};

// 正常场景_float32
TEST_F(l2_pdist_test, pdist_dtype_float32)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float32_p为0
TEST_F(l2_pdist_test, pdist_dtype_float32_p_0)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_pdist_test, pdist_empty)
{
    auto selfDesc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 异常场景_p小于0
TEST_F(l2_pdist_test, pdist_p_lt_0)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(-1.0f);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_self的维度不是二维
TEST_F(l2_pdist_test, pdist_self_not_2)
{
    auto selfDesc = TensorDesc({3, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景_self的第0维为1
TEST_F(l2_pdist_test, pdist_0_dim_1)
{
    auto selfDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull_1
TEST_F(l2_pdist_test, pdist_self_nullptr)
{
    auto pDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(nullptr, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_pdist_test, pdist_out_nullptr)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(3.0f);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_pdist_test, pdist_dtype_int16)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({6}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_pdist_test, pdist_dtype_int32)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_3
TEST_F(l2_pdist_test, pdist_dtype_complex64)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({6}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_pdist_test, pdist_out_shape_invalid)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({1}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnPdistForward, INPUT(selfDesc, pDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
