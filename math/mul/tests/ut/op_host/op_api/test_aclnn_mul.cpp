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

#include "level2/aclnn_mul.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_mul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "mul_test TearDown" << endl;
    }
};

TEST_F(l2_mul_test, aclnnMul_1_9_6_5_1_float_nd_3_1_1_5_5_float_nd)
{
    // left input
    const vector<int64_t>& selfShape = {1, 9, 6, 5, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {3, 1, 1, 5, 5};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {3, 9, 6, 5, 5};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMuls_1_1_9_6_5_1_float_nd_scalar)
{
    // left input
    const vector<int64_t>& selfShape = {1, 1, 9, 6, 5, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<float>(2));
    // output
    const vector<int64_t>& outShape = {1, 1, 9, 6, 5, 1};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_2_3_7_7_float16_nchw_2_1_1_7_float16_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3, 7, 7};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& otherShape = {2, 1, 1, 7};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_NCHW;
    // output
    const vector<int64_t>& outShape = {2, 3, 7, 7};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMuls_2_3_7_7_float16_nchw_scalar)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3, 7, 7};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<float>(5));
    // output
    const vector<int64_t>& outShape = {2, 3, 7, 7};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_11_2_1_12_float64_nchw_11_1_1_12_float64_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {11, 2, 1, 12};
    aclDataType selfDtype = ACL_DOUBLE;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& otherShape = {11, 1, 1, 12};
    aclDataType otherDtype = ACL_DOUBLE;
    aclFormat otherFormat = ACL_FORMAT_NCHW;
    // output
    const vector<int64_t>& outShape = {11, 2, 1, 12};
    aclDataType outDtype = ACL_DOUBLE;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_6_3_3_6_uint8_nhwc_6_1_1_1_uint8_nhwc_float16)
{
    // left input
    const vector<int64_t>& selfShape = {6, 3, 3, 6};
    aclDataType selfDtype = ACL_UINT8;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& otherShape = {6, 1, 1, 1};
    aclDataType otherDtype = ACL_UINT8;
    aclFormat otherFormat = ACL_FORMAT_NHWC;
    // output
    const vector<int64_t>& outShape = {6, 3, 3, 6};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_1_1_6_6_int8_hwcn_7_2_1_1_int8_hwcn_int32)
{
    // left input
    const vector<int64_t>& selfShape = {1, 1, 6, 6};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // right input
    const vector<int64_t>& otherShape = {7, 2, 1, 1};
    aclDataType otherDtype = ACL_INT8;
    aclFormat otherFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {7, 2, 6, 6};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_1_5_1_6_3_int16_ndhwc_3_5_2_6_1_int32_ndhwc_float32)
{
    // left input
    const vector<int64_t>& selfShape = {1, 5, 1, 6, 3};
    aclDataType selfDtype = ACL_INT16;
    aclFormat selfFormat = ACL_FORMAT_NDHWC;
    // right input
    const vector<int64_t>& otherShape = {3, 5, 2, 6, 1};
    aclDataType otherDtype = ACL_INT32;
    aclFormat otherFormat = ACL_FORMAT_NDHWC;
    // output
    const vector<int64_t>& outShape = {3, 5, 2, 6, 3};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_NDHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_3_7_2_5_1_int64_ncdhw_1_1_2_5_3_int64_ncdhw)
{
    // left input
    const vector<int64_t>& selfShape = {3, 7, 2, 5, 1};
    aclDataType selfDtype = ACL_INT64;
    aclFormat selfFormat = ACL_FORMAT_NCDHW;
    // right input
    const vector<int64_t>& otherShape = {1, 1, 2, 5, 3};
    aclDataType otherDtype = ACL_INT64;
    aclFormat otherFormat = ACL_FORMAT_NCDHW;
    // output
    const vector<int64_t>& outShape = {3, 7, 2, 5, 3};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_NCDHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_23_71_complex64_nd_23_1_complex64_nd)
{
    // left input
    const vector<int64_t>& selfShape = {23, 71};
    aclDataType selfDtype = ACL_COMPLEX64;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {23, 1};
    aclDataType otherDtype = ACL_COMPLEX64;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {23, 71};
    aclDataType outDtype = ACL_COMPLEX64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_1_6_23_complex128_nd_5_1_1_complex128_nd)
{
    // left input
    const vector<int64_t>& selfShape = {1, 6, 23};
    aclDataType selfDtype = ACL_COMPLEX128;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {5, 1, 1};
    aclDataType otherDtype = ACL_COMPLEX128;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {5, 6, 23};
    aclDataType outDtype = ACL_COMPLEX128;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_1_11_21_6_float32_nd_12_1_1_1_float32_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {1, 11, 21, 16};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {12, 1, 1, 1};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_NCHW;
    // output
    const vector<int64_t>& outShape = {12, 11, 21, 16};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// Bool dtype
TEST_F(l2_mul_test, aclnnMul_9_nd_1_bool_nd)
{
    // left input
    const vector<int64_t>& selfShape = {9};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1};
    aclDataType otherDtype = ACL_BOOL;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {9};
    aclDataType outDtype = ACL_BOOL;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// Bool dtype
TEST_F(l2_mul_test, aclnnMuls_2_2_1_7_3_1_6_nd_scalar)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 1, 7, 3, 1, 6};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<bool>(5));
    // output
    const vector<int64_t>& outShape = {2, 2, 1, 7, 3, 1, 6};
    aclDataType outDtype = ACL_BOOL;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_0_2_float32_nd_1_2_float32_nd_empty_tensor)
{
    // left input
    const vector<int64_t>& selfShape = {0, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1, 2};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {0, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, aclnnMuls_0_2_float32_nd_1_2_float32_nd_empty_tensor)
{
    // left input
    const vector<int64_t>& selfShape = {0, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<float>(1));
    // output
    const vector<int64_t>& outShape = {0, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, aclnnMul_100_int8_100_int8_in_overflow)
{
    // left input
    const vector<int64_t>& selfShape = {100};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {100};
    aclDataType otherDtype = ACL_INT8;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {100};
    aclDataType outDtype = ACL_INT8;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(257, 300);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 1);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_100_int32_100_int32_out_overflow)
{
    // left input
    const vector<int64_t>& selfShape = {100};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {100};
    aclDataType otherDtype = ACL_INT32;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {100};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(65536, 66000);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(65536, 66000);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_1_2_float16_2_1_float16_boundary_value)
{
    // left input
    const vector<int64_t>& selfShape = {1, 2};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 1};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 2};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<float>{1, -1});
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// not contiguous
TEST_F(l2_mul_test, aclnnMul_5_4_float32_not_contiguous)
{
    // left input
    const vector<int64_t>& selfShape = {5, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& selfViewDim = {1, 5};
    int64_t selfOffset = 0;
    const vector<int64_t>& selfStorageDim = {4, 5};

    // right input
    const vector<int64_t>& otherShape = {5, 4};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    const vector<int64_t>& otherViewDim = {1, 5};
    int64_t otherOffset = 0;
    const vector<int64_t>& otherStorageDim = {4, 5};
    // output
    const vector<int64_t>& outShape = {5, 4};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc =
        TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);
    auto otherTensorDesc =
        TensorDesc(otherShape, otherDtype, otherFormat, otherViewDim, otherOffset, otherStorageDim).ValueRange(-2, 2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_input_output_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut_l = OP_API_UT(aclnnMul, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnMul, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_r.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(aclnnMul, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_o.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mul_test, aclnnMuls_input_output_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(static_cast<float>(1));

    auto ut_l = OP_API_UT(aclnnMuls, INPUT((aclTensor*)nullptr, scalarDesc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnMuls, INPUT(tensor_desc, (aclScalar*)nullptr), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_r.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalarDesc), OUTPUT((aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_o.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mul_test, aclnnMul_input_error_shape)
{
    // left input
    const vector<int64_t>& selfShape = {123, 11, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {123, 8, 2};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {123, 11, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mul_test, aclnnMul_input_output_diff_shape)
{
    // left input
    const vector<int64_t>& selfShape = {123, 11, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {123, 11, 2};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {123, 8, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mul_test, aclnnMuls_input_output_diff_shape)
{
    // left input
    const vector<int64_t>& selfShape = {123, 11, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<float>(2));
    // output
    const vector<int64_t>& outShape = {123, 8, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mul_test, aclnnMul_error_input_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {6, 12, 11, 16};
    aclDataType selfDtype = ACL_UINT64;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& otherShape = {6, 12, 11, 1};
    aclDataType otherDtype = ACL_UINT64;
    aclFormat otherFormat = ACL_FORMAT_NHWC;
    // output
    const vector<int64_t>& outShape = {6, 12, 11, 16};
    aclDataType outDtype = ACL_UINT64;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mul_test, aclnnMuls_error_input_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {6, 12, 11, 16};
    aclDataType selfDtype = ACL_UINT64;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<float>(1));
    // output
    const vector<int64_t>& outShape = {6, 12, 11, 16};
    aclDataType outDtype = ACL_UINT64;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mul_test, aclnnMul_bool_int16_for_kron)
{
    // left input
    const vector<int64_t>& selfShape = {2, 1, 2, 1, 2, 1};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1, 2, 1, 2, 1, 2};
    aclDataType otherDtype = ACL_INT16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 2, 2, 2, 2, 2};
    aclDataType outDtype = ACL_INT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMuls_bool_tensor_double_scalar)
{
    // left input
    const vector<int64_t>& selfShape = {1, 724};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<double>(100000.000000));
    // output
    const vector<int64_t>& outShape = {1, 724};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_float16_float32_mix_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1, 3};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 3};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// ascend910B complex64 support aicore broadcast
TEST_F(l2_mul_test, ascend910B2_aclnnMul_COMPLEX64_mul_aicore_broadcast)
{
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, aclnnMul_float32_float16_mix_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1, 3};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 3};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// ascend910B complex64 support aicore
TEST_F(l2_mul_test, ascend910B2_aclnnMul_COMPLEX64_mul_aicore)
{
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMul, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// aclnnMul, fp16+fp32
TEST_F(l2_mul_test, Ascend910B2_case_fp16_fp32_mix_dtype)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// aclnnMul, fp32+fp16
TEST_F(l2_mul_test, Ascend910B2_case_fp32_fp16_mix_dtype)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// aclnnMul, bf16+fp32
TEST_F(l2_mul_test, Ascend910B2_case_bf16_fp32_mix_dtype)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    /*
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
    */
}

// aclnnMul, fp32+bf16
TEST_F(l2_mul_test, Ascend910B2_case_fp32_bf16_mix_dtype)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    /*
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
    */
}

// aclnnMuls, bfloat16 tensor + double scalar
TEST_F(l2_mul_test, Ascend910B2_aclnnMuls_bflot16_tensor_double_scalar)
{
    // left input
    const vector<int64_t>& selfShape = {6};
    aclDataType selfDtype = ACL_BF16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<double>(0.5081327481546147));
    // output
    const vector<int64_t>& outShape = {6};
    aclDataType outDtype = ACL_BF16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// aclnnMuls, complex tensor + double scalar
TEST_F(l2_mul_test, Ascend910B2_aclnnMuls_complex64_tensor_double_scalar)
{
    // left input
    const vector<int64_t>& selfShape = {23, 71};
    aclDataType selfDtype = ACL_COMPLEX64;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<double>(0.5081327481546147));
    // output
    const vector<int64_t>& outShape = {23, 71};
    aclDataType outDtype = ACL_COMPLEX64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_mul_test, Ascend910_9589_l2_muls_inferDtype_001)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 1};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<int32_t>(5));
    // output
    const vector<int64_t>& outShape = {2, 2, 1};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, Ascend910_9589_l2_muls_inferDtype_002)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<int32_t>(5));
    // output
    const vector<int64_t>& outShape = {2, 2, 1};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, Ascend910_9589_l2_muls_inferDtype_003)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 1};
    aclDataType selfDtype = ACL_BF16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<int32_t>(5));
    // output
    const vector<int64_t>& outShape = {2, 2, 1};
    aclDataType outDtype = ACL_BF16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_mul_test, Ascend910_9589_l2_muls_inferDtype_004)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 1};
    aclDataType selfDtype = ACL_DOUBLE;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<int32_t>(5));
    // output
    const vector<int64_t>& outShape = {2, 2, 1};
    aclDataType outDtype = ACL_UINT8;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mul_test, Ascend910_9589_l2_muls_inferDtype_005)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 1};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    auto scalarDesc = ScalarDesc(static_cast<double>(1.0000000000000001e39));
    // output
    const vector<int64_t>& outShape = {2, 2, 1};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMuls, INPUT(selfTensorDesc, scalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}
