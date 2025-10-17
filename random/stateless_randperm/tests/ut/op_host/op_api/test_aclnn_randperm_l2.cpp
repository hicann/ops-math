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
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_randperm.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_randerpm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_randerpm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_randerpm_test TearDown" << endl;
    }
};

TEST_F(l2_randerpm_test, aclnnRandperm_float_nd)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_FLOAT;
    aclFormat out_format = ACL_FORMAT_ND;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_int64_t_nd)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_INT64;
    aclFormat out_format = ACL_FORMAT_ND;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_float16_nd)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    // output
    const vector<int64_t>& outShape = {100};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_float64_ndhwc)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_DOUBLE;
    aclFormat out_format = ACL_FORMAT_NDHWC;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_int32_t_hwch)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_INT32;
    aclFormat out_format = ACL_FORMAT_HWCN;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_5_4_6_int16_t_ncdhw)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_INT16;
    aclFormat out_format = ACL_FORMAT_NCDHW;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_5_4_6_uint8_t_nhwc)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_UINT8;
    aclFormat out_format = ACL_FORMAT_NHWC;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_5_4_6_8_5_2_1_4_int8_t_hwcn)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {100};
    aclDataType out_dtype = ACL_INT8;
    aclFormat out_format = ACL_FORMAT_HWCN;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_5_4_float32_not_contiguous)
{
    const vector<int64_t>& outShape = {100};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;
    const vector<int64_t>& outViewDim = {1};
    int64_t outOffset = 0;
    const vector<int64_t>& outStorageDim = {1};

    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat, outViewDim, outOffset, outStorageDim);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(outTensorDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// StatelessRandperm算子不支持
TEST_F(l2_randerpm_test, aclnnRandperm_bf16_nd)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {5, 4, 6};
    aclDataType out_dtype = ACL_BF16;
    aclFormat out_format = ACL_FORMAT_ND;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
}

// StatelessRandperm算子不支持
TEST_F(l2_randerpm_test, aclnnRandperm_5_4_6_complex64_nd)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {5, 4, 6};
    aclDataType out_dtype = ACL_COMPLEX64;
    aclFormat out_format = ACL_FORMAT_ND;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
}

// StatelessRandperm算子不支持
TEST_F(l2_randerpm_test, aclnnRandperm_5_4_6_complex128_nd)
{
    int64_t n = 100;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {5, 4, 6};
    aclDataType out_dtype = ACL_COMPLEX128;
    aclFormat out_format = ACL_FORMAT_ND;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
}

// StatelessRandperm算子不支持
TEST_F(l2_randerpm_test, aclnnRandperm_5_4_6_bool_hwcn)
{
    int64_t n = 1;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {5, 4, 6};
    aclDataType out_dtype = ACL_BOOL;
    aclFormat out_format = ACL_FORMAT_HWCN;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
}

TEST_F(l2_randerpm_test, aclnnRandperm_exception_1)
{
    int64_t n = -3;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {5, 4, 6};
    aclDataType out_dtype = ACL_INT32;
    aclFormat out_format = ACL_FORMAT_HWCN;

    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnRandperm, INPUT(n, seed, offset), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
