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

#include "level2/aclnn_split_tensor.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_split_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "split_tensor_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "split_tensor_test TearDown" << endl;
    }
};

TEST_F(l2_split_tensor_test, aclnnSplitTensor_8_4_float32_split_2_dim_0)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 4;
    int64_t dim = 0;
    auto out1 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_99_float64_split_25_dim_0)
{
    const vector<int64_t>& self_shape = {99};
    auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 25;
    int64_t dim = 0;

    auto out1 = TensorDesc({25}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({25}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({25}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({24}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_99_complex128_split_25_dim_0)
{
    const vector<int64_t>& self_shape = {99};
    auto self_desc = TensorDesc(self_shape, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 25;
    int64_t dim = 0;

    auto out1 = TensorDesc({25}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({25}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({25}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({24}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_99_complex64_split_25_dim_0)
{
    const vector<int64_t>& self_shape = {99};
    auto self_desc = TensorDesc(self_shape, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 25;
    int64_t dim = 0;

    auto out1 = TensorDesc({25}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({25}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({25}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({24}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_20_4_float32_dim_0)
{
    const vector<int64_t>& self_shape = {20, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 15;
    int64_t dim = 0;

    auto out1 = TensorDesc({15, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_8_160_float16_dim_1)
{
    const vector<int64_t>& self_shape = {8, 160};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(80, 2);
    uint64_t split_sections = 2;
    int64_t dim = 1;

    auto out = TensorDesc({8, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(80, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// AiCpu support bfloat16
TEST_F(l2_split_tensor_test, aclnnSplitTensor_2_3_4_5_bfloat16_dim_2)
{
    const vector<int64_t>& self_shape = {2, 3, 4, 5};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    uint64_t split_sections = 3;
    int64_t dim = 2;

    auto out1 = TensorDesc({2, 3, 3, 5}, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 3, 1, 5}, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_7_8_9_11_int64_dim_3)
{
    const vector<int64_t>& self_shape = {7, 8, 9, 11};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    uint64_t split_sections = 4;
    int64_t dim = 3;

    auto out1 = TensorDesc({7, 8, 9, 4}, ACL_INT64, ACL_FORMAT_NHWC);
    auto out2 = TensorDesc({7, 8, 9, 4}, ACL_INT64, ACL_FORMAT_NHWC);
    auto out3 = TensorDesc({7, 8, 9, 3}, ACL_INT64, ACL_FORMAT_NHWC);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_15_13_20_8_int32_dim_0)
{
    const vector<int64_t>& self_shape = {15, 13, 20, 8};
    auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    uint64_t split_sections = 4;
    int64_t dim = 0;

    auto out1 = TensorDesc({4, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out2 = TensorDesc({4, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out3 = TensorDesc({4, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out4 = TensorDesc({3, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_27_11_6_5_3_int16_dim_1)
{
    const vector<int64_t>& self_shape = {27, 11, 6, 5, 3};
    auto self_desc = TensorDesc(self_shape, ACL_INT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    uint64_t split_sections = 3;
    int64_t dim = 1;

    auto out1 = TensorDesc({27, 3, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out2 = TensorDesc({27, 3, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out3 = TensorDesc({27, 3, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out4 = TensorDesc({27, 2, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_2_10_111_3_1_int8_dim_2)
{
    const vector<int64_t>& self_shape = {2, 10, 111, 3, 1};
    auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    uint64_t split_sections = 30;
    int64_t dim = 2;

    auto out1 = TensorDesc({2, 10, 30, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out2 = TensorDesc({2, 10, 30, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out3 = TensorDesc({2, 10, 30, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out4 = TensorDesc({2, 10, 21, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_32_23_uint8_dim_0)
{
    const vector<int64_t>& self_shape = {32, 23};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 11;
    int64_t dim = 0;

    auto out1 = TensorDesc({11, 23}, ACL_UINT8, ACL_FORMAT_ND);
    auto out2 = TensorDesc({11, 23}, ACL_UINT8, ACL_FORMAT_ND);
    auto out3 = TensorDesc({10, 23}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_16_17_bool_dim_1)
{
    const vector<int64_t>& self_shape = {16, 17};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 16;
    int64_t dim = 1;

    auto out1 = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    auto out2 = TensorDesc({16, 1}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_self_nullptr)
{
    uint64_t split_sections = 3;
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT((aclTensor*)nullptr, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_out_list_nullptr)
{
    const vector<int64_t>& self_shape = {2, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 3;
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT((aclTensorList*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_error_input_dtype)
{
    const vector<int64_t>& self_shape = {2, 43};
    auto self_desc = TensorDesc(self_shape, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 21;
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 21}, ACL_UINT64, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 21}, ACL_UINT64, ACL_FORMAT_ND);
    auto out3 = TensorDesc({2, 1}, ACL_UINT64, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_error_output_dtype)
{
    const vector<int64_t>& self_shape = {2, 43};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 21;
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 21}, ACL_UINT64, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 21}, ACL_UINT64, ACL_FORMAT_ND);
    auto out3 = TensorDesc({2, 1}, ACL_UINT64, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_max_self_len_error)
{
    const vector<int64_t>& self_shape = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 5;
    int64_t dim = 4;

    auto out1 = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_min_self_len_error)
{
    const vector<int64_t>& self_shape = {};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 0;
    int64_t dim = 0;

    auto out1 = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_max_output_len_error)
{
    const vector<int64_t>& self_shape = {8, 7, 6, 5, 4, 3, 2, 1};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 8;
    int64_t dim = 0;

    auto out1 = TensorDesc({8, 7, 6, 5, 4, 3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_dim_index_out_of_range)
{
    const vector<int64_t>& self_shape = {2, 3, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 4;
    int64_t dim = 3;

    auto out1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_split_sections_bigger_than_dim_value)
{
    const vector<int64_t>& self_shape = {2, 2, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 3;
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({
        out1,
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_2_0_3_float32_dim_1_split_empty)
{
    const vector<int64_t>& self_shape = {2, 0, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 0;
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, ascend910B2_aclnnSplitTensor_200_3_bfloat16_dim_0_slice_branch)
{
    const vector<int64_t>& self_shape = {200, 3};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(100, 2);
    uint64_t split_sections = 2;
    int64_t dim = 0;

    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(100, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_200_3_float16_dim_0_slice_branch)
{
    const vector<int64_t>& self_shape = {200, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(100, 2);
    uint64_t split_sections = 2;
    int64_t dim = 0;

    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(100, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_200_3_float16_dim_0_out_overwrite)
{
    const vector<int64_t>& self_shape = {200, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(100, 2);
    uint64_t split_sections = 2;
    int64_t dim = 0;

    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(32, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_split_tensor_test, aclnnSplitTensor_200_3_float16_dim_0_index_overwrite)
{
    const vector<int64_t>& self_shape = {200, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(100, 2);
    uint64_t split_sections = 20;
    int64_t dim = 0;

    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(4, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}