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
#include "../../../../op_host/op_api/aclnn_pdist.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_pdist_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_pdist SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "l2_pdist TearDown" << std::endl;
    }
};

// float 场景
TEST_F(l2_pdist_test, case_float_ND_001)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_pdist_test, case_float_NCHW_002)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_pdist_test, case_float_HWCN_003)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// self取值范围是(-1, 1)的场景
TEST_F(l2_pdist_test, case_mean_N1_1_004)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 异常数据类型bfloat16 场景
TEST_F(l2_pdist_test, case_bfloat16_anormal_005)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常数据类型int 场景
TEST_F(l2_pdist_test, case_int32_anormal_006)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常数据类型bool 场景
TEST_F(l2_pdist_test, case_bool_anormal_007)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常数据类型COMPELEX 场景
TEST_F(l2_pdist_test, case_complex_anormal_008)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 未定义数据类型场景
TEST_F(l2_pdist_test, case_undefined_datatype_009)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self为nullptr
TEST_F(l2_pdist_test, case_self_nullptr_010)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT((aclTensor*)nullptr, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dim维度异常场景：不等于2维的异常场景
TEST_F(l2_pdist_test, case_dim3_anormal_011)
{
    auto selfDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常空tensor的场景1
TEST_F(l2_pdist_test, case_emptytensor03_normal_012)
{
    auto selfDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常空tensor的场景2
TEST_F(l2_pdist_test, case_emptytensor30_normal_013)
{
    auto selfDesc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常输出空tensor的场景
TEST_F(l2_pdist_test, case_emptytensor13_normal_014)
{
    auto selfDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 异常空tensor的场景
TEST_F(l2_pdist_test, case_emptytensor_anormal_015)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// p的取值
// 正常场景：p=3 整型
TEST_F(l2_pdist_test, case_p3_normal_016)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 3;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景：p=10.91 浮点型
TEST_F(l2_pdist_test, case_p10d91_normal_017)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 10.91;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景：p=0 特殊值
TEST_F(l2_pdist_test, case_p0_special_vaule_018)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 0;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 异常场景：p=NAN
TEST_F(l2_pdist_test, case_nan_anormal_019)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = std::numeric_limits<float>::quiet_NaN();
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 p=-1
TEST_F(l2_pdist_test, case_N1_anormal_020)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = -1;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out的shape异常
TEST_F(l2_pdist_test, case_pdist_invalid_out_shape)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 3;
    auto ut = OP_API_UT(aclnnPdist, INPUT(selfDesc, p), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}
