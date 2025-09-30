/**
 * This program is free software, you can redistribute it and/or modify.
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

#include "level2/aclnn_trans_matmul_weight.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_trans_matmul_weight_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_trans_matmul_weight_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_trans_matmul_weight_test TearDown" << endl;
    }
};

TEST_F(l2_trans_matmul_weight_test, ascend910B2_test_normal_dim2_input)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    //   SAMPLE: precision simulate
    //   ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_test_normal_dim3_input)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    //   SAMPLE: precision simulate
    //   ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_test_empty)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_dim_larger_than_3)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_invalid_format)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_test_nullptr)
{
    uint64_t weightSize = 0;
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSize(nullptr, &weightSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_test_nullptr_2)
{
    aclIntArray* tensorShape = nullptr;
    vector<int64_t> tensorShapeVec = {32, 16};
    tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSize(tensorShape, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_matmul_weight_test, ascend910B2_test_invalid)
{
    aclIntArray* tensorShape = nullptr;
    vector<int64_t> tensorShapeVec = {16, 16, 32, 16};
    tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSize(tensorShape, &workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_matmul_weight_test, ascend910_95_dim_larger_than_2)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend310P_test_normal_input_int8)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransMatmulWeight, INPUT(x1_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    //   SAMPLE: precision simulate
    //   ut.TestPrecision();
}

TEST_F(l2_trans_matmul_weight_test, ascend310P_test_nullptr)
{
    uint64_t weightSize = 0;
    aclDataType dataType = aclDataType::ACL_INT8;
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSizeV2(nullptr, dataType, &weightSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_matmul_weight_test, ascend310P_test_nullptr_2)
{
    aclIntArray* tensorShape = nullptr;
    vector<int64_t> tensorShapeVec = {32, 16};
    aclDataType dataType = aclDataType::ACL_INT8;
    tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSizeV2(tensorShape, dataType, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_matmul_weight_test, ascend310P_test_invalid_shape)
{
    aclIntArray* tensorShape = nullptr;
    uint64_t weightSize = 0;
    vector<int64_t> tensorShapeVec = {0, 16};
    aclDataType dataType = aclDataType::ACL_INT8;
    tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSizeV2(tensorShape, dataType, &weightSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_matmul_weight_test, ascend310P_test_invalid_shape_dim)
{
    aclIntArray* tensorShape = nullptr;
    uint64_t weightSize = 0;
    vector<int64_t> tensorShapeVec = {2, 2, 2, 2, 2, 32, 16};
    aclDataType dataType = aclDataType::ACL_FLOAT16;
    tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
    aclnnStatus aclRet = aclnnCalculateMatmulWeightSizeV2(tensorShape, dataType, &weightSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
