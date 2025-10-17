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

#include "level2/aclnn_split_with_size.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_split_with_size_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "split_with_size_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "split_with_size_test TearDown" << endl;
    }
};

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_100_float64_dim_0)
{
    const vector<int64_t>& self_shape = {100};
    auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{10, 20, 30, 40});
    int64_t dim = 0;

    auto out1 = TensorDesc({10}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({20}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({30}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({40}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_100_complex128_dim_0)
{
    const vector<int64_t>& self_shape = {100};
    auto self_desc = TensorDesc(self_shape, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{10, 20, 30, 40});
    int64_t dim = 0;

    auto out1 = TensorDesc({10}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({20}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({30}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({40}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_100_complex64_dim_0)
{
    const vector<int64_t>& self_shape = {100};
    auto self_desc = TensorDesc(self_shape, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{10, 20, 30, 40});
    int64_t dim = 0;

    auto out1 = TensorDesc({10}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({20}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({30}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({40}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_20_4_float32_dim_0)
{
    const vector<int64_t>& self_shape = {20, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{15, 5});
    int64_t dim = 0;

    auto out1 = TensorDesc({15, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_8_160_float16_dim_1)
{
    const vector<int64_t>& self_shape = {8, 160};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(80, 2);
    auto split_size = IntArrayDesc(cut_vector);
    int64_t dim = 1;

    auto out = TensorDesc({8, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(80, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// AiCpu support bfloat16
TEST_F(l2_split_with_size_test, aclnnSplitWithSize_2_3_4_5_bfloat16_dim_2)
{
    const vector<int64_t>& self_shape = {2, 3, 4, 5};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{1, 3});
    int64_t dim = 2;

    auto out1 = TensorDesc({2, 3, 1, 5}, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 3, 3, 5}, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_7_8_9_11_int64_dim_3)
{
    const vector<int64_t>& self_shape = {7, 8, 9, 11};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{2, 3, 6});
    int64_t dim = 3;

    auto out1 = TensorDesc({7, 8, 9, 2}, ACL_INT64, ACL_FORMAT_NHWC);
    auto out2 = TensorDesc({7, 8, 9, 3}, ACL_INT64, ACL_FORMAT_NHWC);
    auto out3 = TensorDesc({7, 8, 9, 6}, ACL_INT64, ACL_FORMAT_NHWC);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_15_13_20_8_int32_dim_0)
{
    const vector<int64_t>& self_shape = {15, 13, 20, 8};
    auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{1, 2, 3, 4, 5});
    int64_t dim = 0;

    auto out1 = TensorDesc({1, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out2 = TensorDesc({2, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out3 = TensorDesc({3, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out4 = TensorDesc({4, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out5 = TensorDesc({5, 13, 20, 8}, ACL_INT32, ACL_FORMAT_HWCN);
    auto out_list = TensorListDesc({out1, out2, out3, out4, out5});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_27_11_6_5_3_int16_dim_1)
{
    const vector<int64_t>& self_shape = {27, 11, 6, 5, 3};
    auto self_desc = TensorDesc(self_shape, ACL_INT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3, 3, 3, 2});
    int64_t dim = 1;

    auto out1 = TensorDesc({27, 3, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out2 = TensorDesc({27, 3, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out3 = TensorDesc({27, 3, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out4 = TensorDesc({27, 2, 6, 5, 3}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_2_10_111_3_1_int8_dim_2)
{
    const vector<int64_t>& self_shape = {2, 10, 111, 3, 1};
    auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{10, 1, 60, 40});
    int64_t dim = 2;

    auto out1 = TensorDesc({2, 10, 10, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out2 = TensorDesc({2, 10, 1, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out3 = TensorDesc({2, 10, 60, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out4 = TensorDesc({2, 10, 40, 3, 1}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_32_23_uint8_dim_0)
{
    const vector<int64_t>& self_shape = {32, 23};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{8, 16, 8});
    int64_t dim = 0;

    auto out1 = TensorDesc({8, 23}, ACL_UINT8, ACL_FORMAT_ND);
    auto out2 = TensorDesc({16, 23}, ACL_UINT8, ACL_FORMAT_ND);
    auto out3 = TensorDesc({8, 23}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_16_16_bool_dim_1)
{
    const vector<int64_t>& self_shape = {16, 16};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{16});
    int64_t dim = 1;

    auto out1 = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_self_nullptr)
{
    auto split_size = IntArrayDesc(vector<int64_t>{3});
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT((aclTensor*)nullptr, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_splitSize_nullptr)
{
    const vector<int64_t>& self_shape = {2, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, (aclIntArray*)nullptr, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_out_list_nullptr)
{
    const vector<int64_t>& self_shape = {2, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3});
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT((aclTensorList*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_error_input_dtype)
{
    const vector<int64_t>& self_shape = {2, 43};
    auto self_desc = TensorDesc(self_shape, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3, 10, 30});
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 10}, ACL_UINT64, ACL_FORMAT_ND);
    auto out3 = TensorDesc({2, 30}, ACL_UINT64, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_error_output_dtype)
{
    const vector<int64_t>& self_shape = {2, 43};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3, 10, 30});
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 10}, ACL_UINT64, ACL_FORMAT_ND);
    auto out3 = TensorDesc({2, 30}, ACL_UINT64, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_max_self_len_error)
{
    const vector<int64_t>& self_shape = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{5});
    int64_t dim = 4;

    auto out1 = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_min_self_len_error)
{
    const vector<int64_t>& self_shape = {};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{});
    int64_t dim = 0;

    auto out1 = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_max_output_len_error)
{
    const vector<int64_t>& self_shape = {8, 7, 6, 5, 4, 3, 2, 1};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{8});
    int64_t dim = 0;

    auto out1 = TensorDesc({8, 7, 6, 5, 4, 3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_dim_index_out_of_range)
{
    const vector<int64_t>& self_shape = {2, 3, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{4});
    int64_t dim = 3;

    auto out1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_dim_and_split_size_value_sum_diff_error)
{
    const vector<int64_t>& self_shape = {2, 2, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{1, 2});
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_splitSize_num_diff_with_out_num)
{
    const vector<int64_t>& self_shape = {2, 3, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{1, 2});
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out3 = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_2_0_3_float32_dim_1_split_empty)
{
    const vector<int64_t>& self_shape = {2, 0, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{0});
    int64_t dim = 1;

    auto out1 = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, ascend910B2_aclnnSplitWithSize_200_3_bfloat16_dim_0_slice_branch)
{
    const vector<int64_t>& self_shape = {200, 3};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(100, 2);
    auto split_size = IntArrayDesc(cut_vector);
    int64_t dim = 0;

    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(100, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_200_3_float16_dim_0_slice_branch)
{
    const vector<int64_t>& self_shape = {200, 3};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    const vector<int64_t> cut_vector(100, 2);
    auto split_size = IntArrayDesc(cut_vector);
    int64_t dim = 0;

    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    const vector<TensorDesc> out_res(100, out);
    auto out_list = TensorListDesc(out_res);

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
