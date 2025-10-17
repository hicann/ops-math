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

#include "aclnn_clamp.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_clamp_max_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "clamp_max_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "clamp_max_test TearDown" << std::endl;
    }
};

// 基础用例
TEST_F(l2_clamp_max_test, aclnnClampMax_base_case_1)
{
    // self input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = {2, 4};
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
    auto maxScalar = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0});

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_base_case_2)
{
    // self input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = {2, 4};
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto maxScalar = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Value(vector<float>{0, 0, 0, 0, 0, 0, 0, 0});

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

/* 各元素基本类型覆盖用例
 * 维度：1-8维
 * 基本数据类型：FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64
 * 数据格式：ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 */

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_5_6_7_8_float_nd)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(3.0f);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_5_6_7_float16_nd)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(1.0f);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_int32_nchw)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_NCHW;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto maxScalar = ScalarDesc(1.0f);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_int8_nchw)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_NCHW;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<int8_t>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_uint8_nhwc)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4};
    aclDataType selfDtype = ACL_UINT8;
    aclFormat selfFormat = ACL_FORMAT_NHWC;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<int8_t>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_5_double_ncdhw)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_NCDHW;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_NCDHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<int32_t>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_3_4_5_6_7_int16_nd)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<int16_t>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_clamp_max_test, aclnnClampMax_1_2_int64_nd)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<int64_t>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 各元素特殊类型覆盖用例
// 空tensor   self ,out为空时，空进空出
TEST_F(l2_clamp_max_test, aclnnClampMax_float_nd_empty_tensor)
{
    // self input
    const vector<int64_t>& selfShape = {0};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = {0};
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 边界值
TEST_F(l2_clamp_max_test, aclnnClampMax_float32_nd_boundary_value)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input3
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 不连续
TEST_F(l2_clamp_max_test, aclnnClampMax_5_4_float_nd_not_contiguous)
{
    // self input
    const vector<int64_t>& selfShape = {5, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& selfViewDim = {1, 5};
    int64_t selfOffset = 0;
    const vector<int64_t>& selfStorageDim = {4, 5};

    // out input
    const vector<int64_t>& outShape = {5, 4};
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;
    const vector<int64_t>& outViewDim = {1, 5};
    int64_t outOffset = 0;
    const vector<int64_t>& outStorageDim = {4, 5};

    auto selfTensorDesc =
        TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);

    auto maxScalar = ScalarDesc(static_cast<float>(2));

    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat, outViewDim, outOffset, outStorageDim);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 报错类型覆盖用例
// 空指针
TEST_F(l2_clamp_max_test, aclnnClampMax_input_nullptr)
{
    auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxScalar = ScalarDesc(static_cast<float>(2));

    auto ut_self_nullptr = OP_API_UT(aclnnClampMax, INPUT((aclTensor*)nullptr, maxScalar), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_max_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    aclRet = ut_max_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_out_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, maxScalar), OUTPUT((aclTensor*)nullptr));

    aclRet = ut_out_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self类型不满足
TEST_F(l2_clamp_max_test, aclnnClampMax_self_dtype_error)
{
    // self input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_BF16;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self类型和out不一致
TEST_F(l2_clamp_max_test, aclnnClampMax_self_and_out_dtype_non_consistent)
{
    // self input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self形状大于8
TEST_F(l2_clamp_max_test, aclnnClampMax_self_shape_out_of_8)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_max_test, ascend910_9589_aclnnClampMax_input_diff)
{
    auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxScalar = ScalarDesc(static_cast<float>(2));

    auto ut_self_nullptr = OP_API_UT(aclnnClampMax, INPUT((aclTensor*)nullptr, maxScalar), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_max_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    aclRet = ut_max_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_out_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, maxScalar), OUTPUT((aclTensor*)nullptr));

    aclRet = ut_out_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_normal = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, maxScalar), OUTPUT(tensor_desc));

    aclRet = ut_normal.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}