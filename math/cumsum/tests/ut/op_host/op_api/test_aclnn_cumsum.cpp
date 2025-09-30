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

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_cumsum.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_cumsum_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "cumsum_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "cumsum_test TearDown" << endl;
    }
};

// 正常场景_FLOAT_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_FLOAT_ND)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_FLOAT_ND_CUBE实现
TEST_F(l2_cumsum_test, ascend910B2_l2_cumsum_normal_FLOAT_ND)
{
    auto selfDesc = TensorDesc({32, 16 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({32, 16 * 256}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_FLOAT16_NCHW
TEST_F(l2_cumsum_test, l2_cumsum_normal_FLOAT16_NCHW)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT16;
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_FLOAT16_NCHW_CUBE实现
TEST_F(l2_cumsum_test, ascend910B2_l2_cumsum_normal_FLOAT16_ND)
{
    auto selfDesc = TensorDesc({32, 16, 256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT16;
    auto outDesc = TensorDesc({32, 16, 256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 大shape场景
TEST_F(l2_cumsum_test, ascend910B2_l2_cumsum_large_FLOAT32_ND)
{
    auto selfDesc = TensorDesc({256, 256, 70000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 2;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({256, 256, 70000}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
}

// 正常场景_INT32_NHWC
TEST_F(l2_cumsum_test, l2_cumsum_normal_INT32_NHWC)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NHWC);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_INT32;
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_DOUBLE_HWCN
TEST_F(l2_cumsum_test, l2_cumsum_normal_DOUBLE_HWCN)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_HWCN);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_DOUBLE;
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_UINT8_NDHWC
TEST_F(l2_cumsum_test, l2_cumsum_normal_UINT8_NDHWC)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NDHWC);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_UINT8;
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_INT8_NCDHW
TEST_F(l2_cumsum_test, l2_cumsum_normal_INT8_NCDHW)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_NCDHW);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_INT8;
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_INT16_NCHW_NHWC
TEST_F(l2_cumsum_test, l2_cumsum_normal_INT16_NCHW_NHWC)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NCHW);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_INT16;
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_INT64_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_INT64_ND)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_INT64;
    auto outDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_COMPLEX64_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_COMPLEX64_ND)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_COMPLEX64;
    auto outDesc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_COMPLEX128_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_COMPLEX128_ND)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_COMPLEX128, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_COMPLEX128;
    auto outDesc = TensorDesc({2, 2}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_BFLOAT16_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_dtype_BFLOAT16_ND)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_BF16;
    auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

        // // precision simulate
        // ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 0维场景
TEST_F(l2_cumsum_test, l2_cumsum_normal_0dim_tensor)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_cumsum_test, l2_cumsum_normal_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckNotNull_self_nullptr
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_self_nullptr)
{
    auto selfDesc = nullptr;
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out_nullptr
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_out_nullptr)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_BOOL
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dtype_BOOL)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_BOOL;
    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dtype_UNDEFINED)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_DT_UNDEFINED;
    auto outDesc = TensorDesc({2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType_FLOAT_to_DOUBLE
TEST_F(l2_cumsum_test, l2_cumsum_normal_dtype_FLOAT_to_DOUBLE)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_DOUBLE;
    auto outDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckPromoteType_FLOAT_to_INT8
TEST_F(l2_cumsum_test, l2_cumsum_normal_dtype_FLOAT_to_INT8)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_INT8;
    auto outDesc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckPromoteType_COMPLEX64_to_FLOAT
TEST_F(l2_cumsum_test, l2_cumsum_normal_dtype_COMPLEX64_to_FLOAT)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckShape
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_shape_unequal)
{
    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDim = 1
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dim_correct)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 1;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDim_1
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dim_incorrect_1)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 2;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDim_2
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dim_incorrect_2)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = -3;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_shape_10D)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_cumsum_test, l2_cumsum_normal_valuerange)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 非连续
TEST_F(l2_cumsum_test, l2_cumsum_normal_uncontiguous)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    int64_t dimDesc = 0;
    aclDataType dtypeDesc = ACL_UINT8;
    auto outDesc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckExclusiveReverse
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_exclusive_reverse)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool exclusive = true;
    bool reverse = true;

    auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckExclusive
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_exclusive)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool exclusive = true;
    bool reverse = false;

    auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // precision simulate
    ut.TestPrecision();
}

// CheckReverse
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_reverse)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool exclusive = false;
    bool reverse = true;

    auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor场景
TEST_F(l2_cumsum_test, l2_cumsumv2_normal_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 0;
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    bool exclusive = false;
    bool reverse = true;

    auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dim = 1场景
TEST_F(l2_cumsum_test, l2_cumsumv2_normal_dim_one_tensor)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 1;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool exclusive = false;
    bool reverse = true;

    auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}