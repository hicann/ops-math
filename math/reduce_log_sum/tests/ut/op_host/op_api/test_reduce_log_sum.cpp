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

#include <float.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_reduce_log_sum.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include <cstdlib>
#include <ctime>

using namespace std;

class l2_reduce_log_sum_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_reduce_log_sum_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_reduce_log_sum_test TearDown" << endl;
    }
};

// data为空指针
TEST_F(l2_reduce_log_sum_test, case_1)
{
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmpty = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(nullptr, dim, keep_dim, noopWithEmpty), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// axes为空指针
TEST_F(l2_reduce_log_sum_test, case_2)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, nullptr, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// reduce为空指针
TEST_F(l2_reduce_log_sum_test, case_3)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// 数据为int64
TEST_F(l2_reduce_log_sum_test, case_4)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 数据为int8
TEST_F(l2_reduce_log_sum_test, case_5)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
// 数据为bool
TEST_F(l2_reduce_log_sum_test, case_6)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
// 数据为double
TEST_F(l2_reduce_log_sum_test, case_7)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;

    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// float16
TEST_F(l2_reduce_log_sum_test, case_8)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT16;
    const vector<int64_t>& outShape = {1, 4};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(4);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float32
TEST_F(l2_reduce_log_sum_test, case_9)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {1, 4};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(4);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
// float32 dim为-1
TEST_F(l2_reduce_log_sum_test, case_10)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{-1});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {2, 1};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(2);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float32 keep_dim为false
TEST_F(l2_reduce_log_sum_test, case_11)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{-1, 0});
    bool keep_dim = false;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {3};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(3);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
// float32 dim为空，noopWithEmptyAxes为true
TEST_F(l2_reduce_log_sum_test, case_12)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{});
    bool keep_dim = false;
    bool noopWithEmptyAxes = true;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {2, 4};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(8);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float32 dim为空，noopWithEmptyAxes为false
TEST_F(l2_reduce_log_sum_test, case_13)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{});
    bool keep_dim = false;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {1};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim重复
TEST_F(l2_reduce_log_sum_test, case_14)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{1, 1});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim超出范围
TEST_F(l2_reduce_log_sum_test, case_15)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{1, 3});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 维度超出范围
TEST_F(l2_reduce_log_sum_test, case_16)
{
    auto xDesc = TensorDesc({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {1, 2, 1, 1, 1, 1, 1, 1, 1, 1};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self为空tensor
TEST_F(l2_reduce_log_sum_test, case_17)
{
    auto xDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {1, 0};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self为0维tensor
TEST_F(l2_reduce_log_sum_test, case_18)
{
    auto xDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    aclDataType dType = ACL_FLOAT;
    const vector<int64_t>& outShape = {};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
