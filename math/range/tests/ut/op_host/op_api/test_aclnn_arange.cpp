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
#include <array>
#include <vector>

#include "level2/aclnn_arange.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_arange_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "arange_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "arange_test TearDown" << endl;
    }
};

// 输入bool
TEST_F(l2_arange_test, input_bool)
{
    auto start = ScalarDesc(static_cast<bool>(1));
    auto end = ScalarDesc(static_cast<float>(17));
    auto step = ScalarDesc(static_cast<int16_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入int8
TEST_F(l2_arange_test, input_int8)
{
    auto start = ScalarDesc(static_cast<int8_t>(1));
    auto end = ScalarDesc(static_cast<int8_t>(17));
    auto step = ScalarDesc(static_cast<int8_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入uint8
TEST_F(l2_arange_test, input_uint8)
{
    auto start = ScalarDesc(static_cast<uint8_t>(1));
    auto end = ScalarDesc(static_cast<uint8_t>(17));
    auto step = ScalarDesc(static_cast<uint8_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入int16
TEST_F(l2_arange_test, input_int16)
{
    auto start = ScalarDesc(static_cast<int16_t>(1));
    auto end = ScalarDesc(static_cast<int16_t>(17));
    auto step = ScalarDesc(static_cast<int16_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入uint16
TEST_F(l2_arange_test, input_uint16)
{
    auto start = ScalarDesc(static_cast<uint16_t>(1));
    auto end = ScalarDesc(static_cast<uint16_t>(17));
    auto step = ScalarDesc(static_cast<uint16_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入int32
TEST_F(l2_arange_test, input_int32)
{
    auto start = ScalarDesc(static_cast<int32_t>(1));
    auto end = ScalarDesc(static_cast<int32_t>(17));
    auto step = ScalarDesc(static_cast<int32_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入uint32
TEST_F(l2_arange_test, input_uint32)
{
    auto start = ScalarDesc(static_cast<uint32_t>(1));
    auto end = ScalarDesc(static_cast<uint32_t>(17));
    auto step = ScalarDesc(static_cast<uint32_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入int64
TEST_F(l2_arange_test, input_int64)
{
    auto start = ScalarDesc(static_cast<int64_t>(1));
    auto end = ScalarDesc(static_cast<int64_t>(17));
    auto step = ScalarDesc(static_cast<int64_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入uint64_t
TEST_F(l2_arange_test, input_uint64)
{
    auto start = ScalarDesc(static_cast<uint64_t>(1));
    auto end = ScalarDesc(static_cast<uint64_t>(17));
    auto step = ScalarDesc(static_cast<uint64_t>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入float
TEST_F(l2_arange_test, input_float)
{
    auto start = ScalarDesc(static_cast<float>(1));
    auto end = ScalarDesc(static_cast<float>(17));
    auto step = ScalarDesc(static_cast<float>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入float,step 是负数
TEST_F(l2_arange_test, input_float_step_neg)
{
    auto start = ScalarDesc(static_cast<float>(17.11));
    auto end = ScalarDesc(static_cast<float>(1.1));
    auto step = ScalarDesc(static_cast<float>(-1.11));

    auto outTensor = TensorDesc({15}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入int,step 是负数
TEST_F(l2_arange_test, input_int_step_neg)
{
    auto start = ScalarDesc(static_cast<float>(17));
    auto end = ScalarDesc(static_cast<float>(1));
    auto step = ScalarDesc(static_cast<float>(-1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入float
TEST_F(l2_arange_test, input_float_2)
{
    auto start = ScalarDesc(static_cast<float>(3.1415));
    auto end = ScalarDesc(static_cast<float>(9.1647));
    auto step = ScalarDesc(static_cast<float>(1.111));

    auto outTensor = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// start 空指针
TEST_F(l2_arange_test, start_nullptr)
{
    auto start = ScalarDesc(static_cast<int>(3));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<float>(1.111));

    auto outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(nullptr, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// end 空指针
TEST_F(l2_arange_test, end_nullptr)
{
    auto start = ScalarDesc(static_cast<int>(3));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<float>(1.111));

    auto outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, nullptr, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// step 空指针
TEST_F(l2_arange_test, step_nullptr)
{
    auto start = ScalarDesc(static_cast<int>(3));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<float>(1.111));

    auto outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, nullptr), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// step = 0
TEST_F(l2_arange_test, step_0)
{
    auto start = ScalarDesc(static_cast<int>(3));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<float>(0));

    auto outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// step = 1, start > end
TEST_F(l2_arange_test, start_greater_than_end)
{
    auto start = ScalarDesc(static_cast<int>(37));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<float>(1));

    auto outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// step = -1, start < end
TEST_F(l2_arange_test, start_less_than_end)
{
    auto start = ScalarDesc(static_cast<int>(3));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<float>(-1));

    auto outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输出ACL_FLOAT16
TEST_F(l2_arange_test, output_float16)
{
    auto start = ScalarDesc(static_cast<int>(1));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<int>(1));

    auto outTensor = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输出ACL_BF16
TEST_F(l2_arange_test, output_bfloat16)
{
    auto start = ScalarDesc(static_cast<int>(1));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<int>(1));

    auto outTensor = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 输出ACL_FLOAT64
TEST_F(l2_arange_test, output_double)
{
    auto start = ScalarDesc(static_cast<int>(1));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<int>(1));

    auto outTensor = TensorDesc({16}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输出ACL_INT64
TEST_F(l2_arange_test, output_int64)
{
    auto start = ScalarDesc(static_cast<int>(1));
    auto end = ScalarDesc(static_cast<int>(17));
    auto step = ScalarDesc(static_cast<int>(1));

    auto outTensor = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(start, end, step), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}
