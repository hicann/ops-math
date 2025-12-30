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
#include "random/stateless_random_normal_v2/op_api/aclnn_normal_out.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_normal_float_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_normal_float_tensor SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "l2_normal_float_tensor TearDown" << std::endl;
    }
};

// float_ND 场景
TEST_F(l2_normal_float_tensor_test, case_float_ND_001)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float_NCHW 场景
TEST_F(l2_normal_float_tensor_test, case_float_NCHW_002)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float mean = 1.8;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float_NHWC 场景
TEST_F(l2_normal_float_tensor_test, case_float_NHWC_003)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
    float mean = 1.2;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float_HWCN 场景
TEST_F(l2_normal_float_tensor_test, case_float_NHWC_004)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float_NDHWC 场景
TEST_F(l2_normal_float_tensor_test, case_float_NDHWC_005)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float_NCDHW 场景
TEST_F(l2_normal_float_tensor_test, case_float_NCDHW_006)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16_ND 场景
TEST_F(l2_normal_float_tensor_test, case_float16_ND_007)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype不一致的场景
TEST_F(l2_normal_float_tensor_test, case_float_float64_ND_010)
{
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 入参shape不一致的场景
TEST_F(l2_normal_float_tensor_test, case_float_float64_ND_011)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 指定输出shape的场景
TEST_F(l2_normal_float_tensor_test, case_float_float64_ND_012)
{
    auto stdDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim维度等于8维的场景
TEST_F(l2_normal_float_tensor_test, case_8dim_ND_013)
{
    auto stdDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim维度大于8维的异常场景
TEST_F(l2_normal_float_tensor_test, case_9dim_ND_014)
{
    auto stdDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// std为nullptr的异常场景
TEST_F(l2_normal_float_tensor_test, case_mean_nullptr_ND_015)
{
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, (aclTensor*)nullptr, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out为nullptr的异常场景
TEST_F(l2_normal_float_tensor_test, case_out_nullptr_ND_016)
{
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// mean取值范围是(-1, 1)的场景
TEST_F(l2_normal_float_tensor_test, case_mean_1_1_ND_017)
{
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 含1个入参的空tensor的场景
TEST_F(l2_normal_float_tensor_test, case_empty1_ND_018)
{
    auto stdDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// std和mean的format不一致的场景
TEST_F(l2_normal_float_tensor_test, case_NCHW_ND_format_019)
{
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float mean = 1.5;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatTensor, INPUT(mean, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
