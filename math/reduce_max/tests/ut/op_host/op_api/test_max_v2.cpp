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
#include "../../../../op_host/op_api/aclnn_max_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_max_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_max_v2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_max_v2_test TearDown" << endl;
    }
};

// self为空指针
TEST_F(l2_max_v2_test, l2_max_v2_test_nullptr_self)
{
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(nullptr, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// dim为空指针
TEST_F(l2_max_v2_test, l2_max_v2_test_nullptr_dim)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, nullptr, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// out为空指针
TEST_F(l2_max_v2_test, l2_max_v2_test_nullptr_out)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// self为空tensor
TEST_F(l2_max_v2_test, l2_max_v2_test_empty_self)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float16
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_float32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

//  正常路径，double, aicpu
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_double)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int64, aicpu
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_int64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{1});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// 正常路径，dim为1
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_1)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{1});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，dim为-1
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_neg1)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{-1});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，dim为-1,0
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_neg1_0)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{-1, 0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，dim为空
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_empty)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{});
    bool keepDims = true;
    bool noopWithEmptyDims = true;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，dim为空
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_empty_and_keep_dims_true)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 异常路径，dim非法
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_invalid)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0, 4});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，keep_dim为false
TEST_F(l2_max_v2_test, l2_max_v2_test_keepdim_false)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{-1, 0});
    bool keepDims = false;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

//  用例不支持，最大维度超过8
TEST_F(l2_max_v2_test, l2_max_v2_test_dimension)
{
    auto selfDesc = TensorDesc({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，0维Tensor
TEST_F(l2_max_v2_test, l2_max_v2_test_zero_dim_tensor)
{
    auto selfDesc = TensorDesc({}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;
    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
