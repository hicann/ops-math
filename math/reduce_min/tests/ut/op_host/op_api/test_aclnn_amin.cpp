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

#include "../../../../op_host/op_api/aclnn_amin.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_amin_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "amin_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "amin_test TearDown" << endl;
    }
};

// test self nullptr
TEST_F(l2_amin_test, aclnnAmin_001_test_nullptr_self)
{
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(nullptr, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test dim nullptr
TEST_F(l2_amin_test, aclnnAmin_002_test_nullptr_dim)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, nullptr, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test out nullptr
TEST_F(l2_amin_test, aclnnAmin_003_test_nullptr_out)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self为空Tensor
TEST_F(l2_amin_test, aclnnAmin_004_test_empty_self)
{
    auto input = TensorDesc({2, 16, 4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常路径, flaot16
TEST_F(l2_amin_test, aclnnAmin_005_test_dtype_float16)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32
TEST_F(l2_amin_test, aclnnAmin_006_test_dtype_float32)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// Aicpu, flaot64
TEST_F(l2_amin_test, aclnnAmin_007_test_dtype_float64)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, uint8
TEST_F(l2_amin_test, aclnnAmin_008_test_dtype_uint8)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_UINT8, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, int8
TEST_F(l2_amin_test, aclnnAmin_009_test_dtype_int8)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_INT8, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// Aicpu, int16
TEST_F(l2_amin_test, aclnnAmin_010_test_dtype_int16)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, int32
TEST_F(l2_amin_test, aclnnAmin_011_test_dtype_int32)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// Aicpu, int64
TEST_F(l2_amin_test, aclnnAmin_012_test_dtype_int64)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, bool
TEST_F(l2_amin_test, aclnnAmin_013_test_dtype_bool)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1, keepdim为true
TEST_F(l2_amin_test, aclnnAmin_014_test_dtype_float32_dim_1_keepDim_true)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为-1
TEST_F(l2_amin_test, aclnnAmin_015_test_dtype_float32_dim_neg1)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{-1});
    bool keepDim = true;
    auto out = TensorDesc({2, 16, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为-1, 0
TEST_F(l2_amin_test, aclnnAmin_016_test_dtype_float32_dim_neg1_0)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{-1, 0});
    bool keepDim = true;
    auto out = TensorDesc({1, 16, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1,1,2
TEST_F(l2_amin_test, aclnnAmin_017_test_dtype_float32_dim_1_1_2)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1, 1, 2});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径, flaot32, dim为空
TEST_F(l2_amin_test, aclnnAmin_018_test_dtype_float32_dim_empty)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{});
    bool keepDim = true;
    auto out = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 测试超过8维tensor
TEST_F(l2_amin_test, aclnnAmin_019_test_dim_9)
{
    auto input = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_BOOL, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({1, 1, 3, 4, 5, 6, 7, 8, 9}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self与out 类型不一致
TEST_F(l2_amin_test, aclnnAmin_020_test_self_dtype_float32_out_dtype_int16)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self不在支持类型内
TEST_F(l2_amin_test, aclnnAmin_021_test_dtype_complex64)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// reduce的dim size为0
TEST_F(l2_amin_test, aclnnAmin_022_test_reduce_dim_size_zero)
{
    auto input = TensorDesc({2, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim越界
TEST_F(l2_amin_test, aclnnAmin_023_test_dim_out_range)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{5});
    bool keepDim = true;
    auto out = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out的shape不符合infer shape推导
TEST_F(l2_amin_test, aclnnAmin_024_test_out_shape_invalid)
{
    auto input = TensorDesc({2, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 0维Tensor
TEST_F(l2_amin_test, aclnnAmin_025_test_zero_dim_tensor)
{
    auto input = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keepDim = false;
    auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试非连续支持
TEST_F(l2_amin_test, aclnnAmin_026_test_NonContiguous)
{
    auto input = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto out = TensorDesc({1, 4}, ACL_BOOL, ACL_FORMAT_ND, {4, 1}, 0, {4, 1});

    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keepDim = true;

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1, format为nchw
TEST_F(l2_amin_test, aclnnAmin_027_test_dtype_float32_dim_1_format_nchw)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1, format为nhwc
TEST_F(l2_amin_test, aclnnAmin_028_test_dtype_float32_dim_1_format_nhwc)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1, format为hwcn
TEST_F(l2_amin_test, aclnnAmin_029_test_dtype_float32_dim_1_format_hwcn)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1, format为ndhwc
TEST_F(l2_amin_test, aclnnAmin_030_test_dtype_float32_dim_1_format_ndhwc)
{
    auto input = TensorDesc({2, 16, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常路径, flaot32, dim为1, format为ncdhw
TEST_F(l2_amin_test, aclnnAmin_031_test_dtype_float32_dim_1_format_ncdhw)
{
    auto input = TensorDesc({2, 16, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = true;
    auto out = TensorDesc({2, 1, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 特殊值 Nan
//  TEST_F(l2_amin_test, aclnnAmin_032_test_dtype_float32_special_value_nan) {
//      auto input = TensorDesc({6, 1}, ACL_FLOAT, ACL_FORMAT_ND)
//                              .Value(vector<float>{1, 0, NAN, 0, 1, -1});
//      auto dim = IntArrayDesc(vector<int64_t>{0});
//      bool keepDim = true;
//      auto out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

//     auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     ut.TestPrecision();
// }

// 特殊值 inf、-inf
//  TEST_F(l2_amin_test, aclnnAmin_032_test_dtype_float32_special_value_inf) {
//      auto input = TensorDesc({6, 1}, ACL_FLOAT, ACL_FORMAT_ND)
//                              .Value(vector<float>{INFINITY, -INFINITY, 0, 0, 1, -1});
//      auto dim = IntArrayDesc(vector<int64_t>{0});
//      bool keepDim = true;
//      auto out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

//     auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     ut.TestPrecision();
// }

// 正常路径, bfloat16
TEST_F(l2_amin_test, ascend910B2_aclnnAmin_033_test_dtype_bfloat16)
{
    auto input = TensorDesc({2, 16, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    auto out = TensorDesc({2, 4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAmin, INPUT(input, dim, keepDim), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}