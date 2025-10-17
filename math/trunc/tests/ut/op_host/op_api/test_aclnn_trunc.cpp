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

#include "level2/aclnn_trunc.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_trunc_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Trunc Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Trunc Test TearDown" << std::endl;
    }
};

TEST_F(l2_trunc_test, case_dtype_float16)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_dtype_float32)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_invalid_self_dtype_int32)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_invalid_out_dtype_int32)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_invalid_dtype_float_float16)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_shape1D)
{
    auto tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_shape_2D)
{
    auto tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_shape_3D)
{
    auto tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_shape_4D)
{
    auto tensor_desc = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_shape_5D)
{
    auto tensor_desc = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_shape_8D)
{
    auto tensor_desc = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_trunc_test, case_invalid_shape_9D)
{
    auto tensor_desc = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_input_not_contiguous)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// CheckNotNull self
TEST_F(l2_trunc_test, case_null_self)
{
    auto out_tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(nullptr), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_trunc_test, case_null_out)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensor
TEST_F(l2_trunc_test, case_2)
{
    auto self_tensor_desc = TensorDesc({1, 0, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnTrunc, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat
TEST_F(l2_trunc_test, case_8)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat
TEST_F(l2_trunc_test, case_9)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShape
TEST_F(l2_trunc_test, case_invalide_shape_1)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_trunc_test, case_invalide_shape_2)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, ascend910B2_case_dtype_bfloat16)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}