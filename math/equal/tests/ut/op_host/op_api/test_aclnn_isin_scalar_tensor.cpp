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
#include "level2/aclnn_isin.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_isin_scalar_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_isin_scalar_tensor_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_isin_scalar_tensor_test TearDown" << std::endl;
    }
};

TEST_F(l2_isin_scalar_tensor_test, case_nullptr)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(nullptr, testElements, false, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, nullptr, false, false), OUTPUT(outDesc));
    workspaceSize = 0;
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(nullptr));
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_isin_scalar_tensor_test, case_null_tensor)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_null_tensor_invert)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim1_float_nd)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.0f, 1.2f, 2.1f, 2.3f, 5.0f});
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// CPU不支持float16,注释精度对比
TEST_F(l2_isin_scalar_tensor_test, case_dim2_float16_nd)
{
    auto element = ScalarDesc(1.0f, ACL_FLOAT16);
    auto testElements = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim3_double_nd)
{
    auto element = ScalarDesc(1.0);
    auto testElements =
        TensorDesc({2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim4_uint8_nchw)
{
    auto element = ScalarDesc(static_cast<uint8_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim4_int8_nhwc)
{
    auto element = ScalarDesc(static_cast<int8_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_NHWC);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim4_int16_hwcn)
{
    auto element = ScalarDesc(static_cast<int16_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim5_int32_ndhwc)
{
    auto element = ScalarDesc(static_cast<int32_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim5_int64_ncdhw)
{
    auto element = ScalarDesc(static_cast<int64_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NCDHW);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim6_float_nd)
{
    auto element = ScalarDesc(0.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim7_float_nd)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_dim8_float_nd)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_error_shape)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_error_dtype)
{
    auto element = ScalarDesc(static_cast<uint32_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2}, ACL_UINT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_error_out_dtype)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_can_not_promote_type)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_diff_dtype)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, case_not_contiguous)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_isin_scalar_tensor_test, ascend910B2_case_dim6_bfloat16_nd)
{
    auto element = ScalarDesc(0.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}