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

#include "../../../../op_host/op_api/aclnn_aminmax.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_aminmax_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "aminmax_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "aminmax_test TearDown" << std::endl;
    }
};

TEST_F(l2_aminmax_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(nullptr, dim_desc, true), OUTPUT(minOut, maxOut));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, nullptr, true), OUTPUT(minOut, maxOut));
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(nullptr, maxOut));
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, nullptr));
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_aminmax_test, case_null_tensor_no_dim)
{
    auto tensor_desc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {0, 1, 2};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_aminmax_test, case_null_tensor_dim_is_empty)
{
    auto tensor_desc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {2};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_aminmax_test, case_null_tensor)
{
    auto tensor_desc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// TEST_F(l2_aminmax_test, case_dim1_float_nd) {
//   auto tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//   vector<int64_t> dim = {0};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto maxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// cpu不支持float16，不需要对比精度
// TEST_F(l2_aminmax_test, case_dim2_float16_nd) {
//   auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
//   vector<int64_t> dim = {0};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
//   auto maxOut = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

TEST_F(l2_aminmax_test, case_dim3_double_nd)
{
    auto tensor_desc = TensorDesc({2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_aminmax_test, case_dim4_uint8_nchw)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2, 2}, ACL_UINT8, ACL_FORMAT_NCHW);
    auto maxOut = TensorDesc({2, 2, 2}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, false), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_aminmax_test, case_dim4_int8_nhwc)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_NHWC);
    vector<int64_t> dim = {1};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 1, 2, 2}, ACL_INT8, ACL_FORMAT_NHWC);
    auto maxOut = TensorDesc({2, 1, 2, 2}, ACL_INT8, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_aminmax_test, case_dim4_int16_hwcn)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_HWCN);
    vector<int64_t> dim = {2};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2, 1, 2}, ACL_INT16, ACL_FORMAT_HWCN);
    auto maxOut = TensorDesc({2, 2, 1, 2}, ACL_INT16, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// TEST_F(l2_aminmax_test, case_dim5_int32_ndhwc) {
//   auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NDHWC);
//   vector<int64_t> dim = {3};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({2, 2, 2, 1, 2}, ACL_INT32, ACL_FORMAT_NDHWC);
//   auto maxOut = TensorDesc({2, 2, 2, 1, 2}, ACL_INT32, ACL_FORMAT_NDHWC);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// 放开精度对比后，流水线挂住，先注释掉
TEST_F(l2_aminmax_test, case_dim5_int64_ncdhw)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NCDHW);
    vector<int64_t> dim = {4};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NCDHW);
    auto maxOut = TensorDesc({2, 2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 放开精度对比后，流水线挂住，先注释掉
TEST_F(l2_aminmax_test, case_dim6_bool_nd)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    vector<int64_t> dim = {5};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2, 2, 2, 2, 1}, ACL_BOOL, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 2, 2, 2, 2, 1}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_aminmax_test, case_dim6_double_nd)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    vector<int64_t> dim = {0, 1, 2, 3, 4, 5};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({1, 1, 1, 1, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({1, 1, 1, 1, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// cpu不支持float16，不需要对比精度
TEST_F(l2_aminmax_test, case_dim7_float16_nd)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> dim = {4};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, false), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// TEST_F(l2_aminmax_test, case_dim8_float_nd) {
//   auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//   vector<int64_t> dim = {7};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto maxOut = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

TEST_F(l2_aminmax_test, case_error_shape)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> dim = {8};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_aminmax_test, case_error_dtype)
{
    auto tensor_desc = TensorDesc({2, 2}, ACL_UINT32, ACL_FORMAT_ND);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({1, 2}, ACL_UINT32, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({1, 2}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_aminmax_test, case_diff_dtype)
{
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_aminmax_test, case_error_dim)
{
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {2};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_aminmax_test, case_dim_is_neg) {
//   auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//   vector<int64_t> dim = {-1};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto maxOut = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

TEST_F(l2_aminmax_test, case_dim0_float_no_keepdim)
{
    auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> dim = {0};
    auto dim_desc = IntArrayDesc(dim);
    auto minOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, false), OUTPUT(minOut, maxOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// TEST_F(l2_aminmax_test, case_not_contiguous) {
//   auto tensor_desc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
//   vector<int64_t> dim = {1};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({3, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto maxOut = TensorDesc({3, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

// TEST_F(l2_aminmax_test, ascend910B2_case_dim8_bfloat16_nd) {
//   auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_ND);
//   vector<int64_t> dim = {7};
//   auto dim_desc = IntArrayDesc(dim);
//   auto minOut = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_BF16, ACL_FORMAT_ND);
//   auto maxOut = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_BF16, ACL_FORMAT_ND);

//   auto ut = OP_API_UT(aclnnAminmax, INPUT(tensor_desc, dim_desc, true), OUTPUT(minOut, maxOut));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }