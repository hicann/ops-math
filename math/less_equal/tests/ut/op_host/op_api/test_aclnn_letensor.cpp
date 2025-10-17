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
#include "level2/aclnn_le_tensor.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_le_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "le_tensor_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "le_tensor_test TearDown" << endl;
    }
};

// 正常场景 int8
TEST_F(l2_le_tensor_test, test_le_tensor_support_int8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_uint8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_int16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_uint16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint16_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint16_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_int64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_float32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_double)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, ascend910B2_le_tensor_support_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, ascend910_le_tensor_support_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景 HWCN
TEST_F(l2_le_tensor_test, test_le_tensor_format_HWCN)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_HWCN).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_HWCN).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景 ACL_FORMAT_NCDHW
TEST_F(l2_le_tensor_test, test_le_tensor_support_NCDHW)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1, 1}, ACL_BOOL, ACL_FORMAT_NCDHW)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景 ACL_FORMAT_NDHWC
TEST_F(l2_le_tensor_test, test_le_tensor_support_NDHWC)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1, 1}, ACL_BOOL, ACL_FORMAT_NDHWC)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景 ACL_FORMAT_NHWC
TEST_F(l2_le_tensor_test, test_le_tensor_support_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1}, ACL_BOOL, ACL_FORMAT_NHWC)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景 ACL_FORMAT_NCHW
TEST_F(l2_le_tensor_test, test_le_tensor_support_NCHW)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1}, ACL_BOOL, ACL_FORMAT_NCHW)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景 做broadcast
TEST_F(l2_le_tensor_test, test_le_tensor_broadcast)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(vector<bool>{
                                   false, false, false, false, false, false, false, false, false, false, false, false,
                                   false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景 做broadcast & dtype cast
TEST_F(l2_le_tensor_test, test_le_tensor_broadcast_cast)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(vector<bool>{
                                   false, false, false, false, false, false, false, false, false, false, false, false,
                                   false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景 做broadcast & int16 走cpu
TEST_F(l2_le_tensor_test, test_le_tensor_cpu_broadcast)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(vector<bool>{
                                   false, false, false, false, false, false, false, false, false, false, false, false,
                                   false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 入参无法broadcast的场景
TEST_F(l2_le_tensor_test, test_le_tensor_input_cannot_broadcast)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_error_shape)
{
    auto tensor_self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_support_bool)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, true, false, true, false, false});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, true, false, true, false, false});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, test_le_tensor_notsupport_bf16)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_notsupport_complex64)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_notsupport_complex128)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参out数据类型不支持的场景
TEST_F(l2_le_tensor_test, test_le_tensor_out_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_le_tensor_test, test_le_tensor_emptytensor)
{
    auto tensor_self =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 入参为self为null场景
TEST_F(l2_le_tensor_test, test_le_tensor_nullptr)
{
    auto tensor_self = nullptr;
    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非连续测试
TEST_F(l2_le_tensor_test, case_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);
    auto scalar_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_le_tensor_test, Ascend910_9589_test_le_tensor_support_int16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}