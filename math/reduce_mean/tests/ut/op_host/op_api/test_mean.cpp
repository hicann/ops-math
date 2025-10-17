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

#include "../../../../op_host/op_api/aclnn_mean.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_mean_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mean_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "mean_test TearDown" << endl;
    }
};

TEST_F(l2_mean_test, case_1)
{
    auto input_tensor_desc = TensorDesc({1, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    aclDataType dtype_desc = ACL_FLOAT;
    bool keepDim = false;
    auto out_desc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mean_test, case_2)
{
    auto input_tensor_desc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT16;
    auto out_desc = TensorDesc({1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mean_test, case_3)
{
    auto input_tensor_desc = TensorDesc({2, 8, 2, 2, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, 2});
    bool keepDim = true;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 1, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_mean_test, case_4)
{
    auto input_tensor_desc = TensorDesc({41, 0, 1, 10, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = true;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_desc = TensorDesc(input_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_mean_test, case_5)
{
    auto tensor_desc = TensorDesc({10, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // auto input_tensor_desc = TensorDesc({1, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    // auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    // aclDataType dtype_desc = ACL_FLOAT;
    // bool keepDim = false;
    // auto out_desc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    // auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 所有维度都做reduce
TEST_F(l2_mean_test, case_6)
{
    auto tensor_desc = TensorDesc({10, 24, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dim_desc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 通过空dim对所有维度reduce
TEST_F(l2_mean_test, case_7)
{
    auto tensor_desc = TensorDesc({10, 24, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dim_desc = IntArrayDesc(vector<int64_t>{});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// fp16
TEST_F(l2_mean_test, case_8)
{
    auto input_tensor_desc = TensorDesc({4, 3, 2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT16;
    auto out_desc = TensorDesc({4, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_mean_test, case_9)
{
    auto input_tensor_desc = TensorDesc({1, 8, 2, 3, 6}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT16;
    auto out_desc = TensorDesc({1, 8, 3, 6}, ACL_FLOAT16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_mean_test, ascend910B2_case_self_bf16)
{
    auto input_tensor_desc = TensorDesc({1, 8, 2, 3, 6}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_BF16;
    auto out_desc = TensorDesc({1, 8, 3, 6}, ACL_BF16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mean_test, case_self_bf16)
{
    auto input_tensor_desc = TensorDesc({1, 8, 2, 3, 6}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_BF16;
    auto out_desc = TensorDesc({1, 8, 3, 6}, ACL_BF16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    // ut.TestPrecision();
}

TEST_F(l2_mean_test, case_10)
{
    auto input_tensor_desc = TensorDesc({1, 8, 2, 3, 6}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto dim_desc = IntArrayDesc(vector<int64_t>{3});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_desc = TensorDesc({1, 8, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_mean_test, case_repeat_dim)
{
    auto input_tensor_desc = TensorDesc({8, 2, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_desc = TensorDesc({1, 8, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mean_test, case_repeat_negative_dim)
{
    auto input_tensor_desc = TensorDesc({8, 2, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, -3});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_desc = TensorDesc({8, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mean_test, case_double)
{
    auto tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_DOUBLE;
    auto out_tensor_desc = TensorDesc({10, 2, 10}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_mean_test, case_scalar)
{
    auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{0});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull
TEST_F(l2_mean_test, case_11)
{
    auto tensor_desc = TensorDesc({64, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, 3, 4});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;

    auto ut = OP_API_UT(aclnnMean, INPUT(nullptr, dim_desc, keepDim, dtype_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnMean, INPUT(tensor_desc, nullptr, keepDim, dtype_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_mean_test, case_12)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_UINT32;
    auto out_tensor_desc = TensorDesc({10}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mean_test, case_13)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT64, ACL_FORMAT_ND);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_UINT64;
    auto out_tensor_desc = TensorDesc({10}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_mean_test, case_14)
{
    auto tensor_desc = TensorDesc({10, 24, 3, 5}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({10, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_mean_test, case_15)
{
    auto tensor_desc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, 2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({10, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType
TEST_F(l2_mean_test, case_17)
{
    auto tensor_desc = TensorDesc({10, 5, 2, 10, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, 2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_INT32;
    auto out_tensor_desc = TensorDesc({10, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    tensor_desc = TensorDesc({10, 5, 2, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    dim_desc = IntArrayDesc(vector<int64_t>{1, 2});
    keepDim = false;
    dtype_desc = ACL_FLOAT;
    out_tensor_desc = TensorDesc({10, 5, 10, 22, 42, 30, 24}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut_2 = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid
TEST_F(l2_mean_test, case_18)
{
    auto tensor_desc = TensorDesc({10, 5, 2, 10, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto dim_desc = IntArrayDesc(vector<int64_t>{1, 6});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({10, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mean_test, case_19)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dim_desc = IntArrayDesc(vector<int64_t>{-3});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_FLOAT;
    auto out_tensor_desc = TensorDesc({10}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// // AICPU
// TEST_F(l2_mean_test, case_20) {
//     auto tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_DOUBLE, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{1});
//     bool keepDim = false;
//     aclDataType dtype_desc = ACL_DOUBLE;
//     auto out_tensor_desc = TensorDesc({10, 2, 10}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_21) {
//     auto tensor_desc = TensorDesc({10, 64, 2, 10}, ACL_BF16, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{1, 2});
//     bool keepDim = false;
//     aclDataType dtype_desc = ACL_BF16;
//     auto out_tensor_desc = TensorDesc({10, 10}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_22) {
//     auto tensor_desc = TensorDesc({2, 2, 4}, ACL_INT32, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{2});
//     bool keepDim = true;
//     aclDataType dtype_desc = ACL_INT32;
//     auto out_tensor_desc = TensorDesc({2, 2, 1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_23) {
//     auto tensor_desc = TensorDesc({24, 20, 4}, ACL_INT64, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{1, 2});
//     bool keepDim = true;
//     aclDataType dtype_desc = ACL_INT64;
//     auto out_tensor_desc = TensorDesc({24, 1, 1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_24) {
//     auto tensor_desc = TensorDesc({2, 2, 4, 8}, ACL_INT16, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{2});
//     bool keepDim = true;
//     aclDataType dtype_desc = ACL_INT16;
//     auto out_tensor_desc = TensorDesc({2, 2, 1, 8}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_25) {
//     auto tensor_desc = TensorDesc({2, 2, 4}, ACL_INT8, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{2});
//     bool keepDim = false;
//     aclDataType dtype_desc = ACL_INT8;
//     auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_26) {
//     auto tensor_desc = TensorDesc({2, 2, 48, 24}, ACL_UINT8, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{2});
//     bool keepDim = true;
//     aclDataType dtype_desc = ACL_UINT8;
//     auto out_tensor_desc = TensorDesc({2, 2, 1, 24}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_19) {
//     auto tensor_desc = TensorDesc({2, 2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{2});
//     bool keepDim = true;
//     aclDataType dtype_desc = ACL_COMPLEX64;
//     auto out_tensor_desc = TensorDesc({2, 2, 1}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// TEST_F(l2_mean_test, case_19) {
//     auto tensor_desc = TensorDesc({2,2,4}, ACL_COMPLEX128, ACL_FORMAT_ND);
//     auto dim_desc = IntArrayDesc(vector<int64_t>{2});
//     bool keepDim = true;
//     aclDataType dtype_desc = ACL_COMPLEX64;
//     auto out_tensor_desc = TensorDesc({2, 2, 1}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnMean, INPUT(tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_tensor_desc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// dim为空，noopWithEmptyAxes为false
TEST_F(l2_mean_test, case_27)
{
    auto selfDesc = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dimDesc = IntArrayDesc(vector<int64_t>{});
    auto outDesc = TensorDesc({1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnMeanV2, INPUT(selfDesc, dimDesc, keepDim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// dim为空tensor，noopWithEmptyAxes为true
TEST_F(l2_mean_test, case_28)
{
    auto selfDesc = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dimDesc = IntArrayDesc(vector<int64_t>{});
    auto outDesc = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDim = true;
    bool noopWithEmptyAxes = true;

    auto ut = OP_API_UT(aclnnMeanV2, INPUT(selfDesc, dimDesc, keepDim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// dim不为空，dtype为float，noopWithEmptyAxes为true
TEST_F(l2_mean_test, case_29)
{
    auto selfDesc = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDim = true;
    bool noopWithEmptyAxes = true;

    auto ut = OP_API_UT(aclnnMeanV2, INPUT(selfDesc, dimDesc, keepDim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// dim不为空，dtype为float，noopWithEmptyAxes为false
TEST_F(l2_mean_test, case_30)
{
    auto selfDesc = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDim = false;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnMeanV2, INPUT(selfDesc, dimDesc, keepDim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}