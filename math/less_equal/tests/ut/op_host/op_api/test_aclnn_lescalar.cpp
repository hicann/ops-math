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

#include "level2/aclnn_le_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_le_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "le_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "le_scalar_test TearDown" << std::endl;
    }
};

// 正常场景
TEST_F(l2_le_scalar_test, test_le_scalar_int8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_uint8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<uint8_t>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_int16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<int16_t>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_int64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<int64_t>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_float32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_double)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, ascend910B2_le_scalar_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, ascend910_le_scalar_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_scalar_test, test_le_scalar_support_bool)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, true, false, true, false, false});

    auto scalar_desc = ScalarDesc(static_cast<bool>(0));

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_notsupport_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_scalar_test, test_le_scalar_notsupport_complex64)
{
    auto tensor_self = TensorDesc({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_scalar_test, test_le_scalar_notsupport_complex128)
{
    auto tensor_self = TensorDesc({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_scalar_test, test_le_scalar_int8_int16_cast)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<int8_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_INT16, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_output_cast)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 0}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 异常场景 shape不一致
TEST_F(l2_le_scalar_test, test_le_scalar_diffshape)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 空指针
TEST_F(l2_le_scalar_test, test_le_scalar_nullptr)
{
    auto tensor_self = nullptr;

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非连续测试
TEST_F(l2_le_scalar_test, case_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, test_le_scalar_error_shape)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_scalar_test, Ascend910_9589_case_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, Ascend910_9589_case_norm_float32_with_double)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(static_cast<double>(0.5));

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_le_scalar_test, Ascend910_9589_case_norm_float16_with_double)
{
    auto tensor_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(static_cast<double>(2.0));

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}
