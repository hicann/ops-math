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

#include "../../../op_host/op_api/aclnn_slice.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_slice_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "slice_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "slice_test TearDown" << endl;
    }
};

TEST_F(l2_slice_test, aclnnSlice_float32)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_float64)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_out_float64)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_float16)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_int64)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_int32)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_int16)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_int8)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_uint8)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_bool)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_self_nullptr)
{
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT((aclTensor*)nullptr, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slice_test, aclnnSlice_out_nullptr)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT((aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slice_test, aclnnSlice_large_shape)
{
    const vector<int64_t>& self_shape = {6, 4, 2, 2, 2, 2, 2, 2, 2};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_error_dim)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 5;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_error_dim1)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = -5;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_error_step)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = -2;
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_dim1)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 5;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({6, 3}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_dim01)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 5;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({6, 3}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_dim2)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = -1;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({6, 3}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_start_end1)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    int64_t start = -2;
    int64_t end = -1;
    int64_t step = 1;
    auto out = TensorDesc({6, 1}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_start_end2)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    int64_t start = 4;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({6, 0}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, ascend910B2_case_BF16_910b)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_slice_test, ascend910_95_case_BF16_910d) {
//   const vector<int64_t>& self_shape = {6, 4};
//   auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//   int64_t dim = 0;
//   int64_t start = 1;
//   int64_t end = 4;
//   int64_t step = 1;
//   auto out = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

//   auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_slice_test, case_BF16_910)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 4;
    int64_t step = 1;
    auto out = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_empty_float32)
{
    const vector<int64_t>& self_shape = {5, 0};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = 1;
    int64_t end = 3;
    int64_t step = 3;
    auto out = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_out_empty_float32)
{
    const vector<int64_t>& self_shape = {5};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    int64_t start = -1;
    int64_t end = 3;
    int64_t step = 3;
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSlice, INPUT(self_desc, dim, start, end, step), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
