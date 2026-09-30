/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>

#include "conversion/pack/op_api/aclnn_stack.h"
#include "gtest/gtest.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_stack_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_stack_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_stack_test TearDown" << endl; }

    void TearDown() override { op::SetPlatformNpuArch(NpuArch::DAV_2201); }
};

// 输入为空指针
TEST_F(l2_stack_test, l2_stack_test_nullptr_input)
{
    int64_t dim = 0;
    auto out_tensor_desc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStack, INPUT(nullptr, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// out为空指针
TEST_F(l2_stack_test, l2_stack_test_nullptr_out)
{
    auto tensor_1_desc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = 0;
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 空tensors
TEST_F(l2_stack_test, l2_stack_test_empty_tensors)
{
    auto tensor_1_desc = TensorDesc({2, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = 0;
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto out_tensor_desc = TensorDesc({2, 2, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，float16
TEST_F(l2_stack_test, l2_stack_test_dtype_float16)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_stack_test, l2_stack_test_dtype_float32)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，double
TEST_F(l2_stack_test, l2_stack_test_dtype_double)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_DOUBLE, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_DOUBLE, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int8
TEST_F(l2_stack_test, l2_stack_test_dtype_int8)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int16
TEST_F(l2_stack_test, l2_stack_test_dtype_int16)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_INT16, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_INT16, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int32
TEST_F(l2_stack_test, l2_stack_test_dtype_int32)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int64
TEST_F(l2_stack_test, l2_stack_test_dtype_int64)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_INT64, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_INT64, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint8
TEST_F(l2_stack_test, l2_stack_test_dtype_uint8)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_UINT8, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_UINT8, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint16
TEST_F(l2_stack_test, l2_stack_test_dtype_uint16)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_UINT16, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_UINT16, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint32
TEST_F(l2_stack_test, l2_stack_test_dtype_uint32)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_UINT32, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_UINT32, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint64
TEST_F(l2_stack_test, l2_stack_test_dtype_uint64)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_UINT64, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_UINT64, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，bool
TEST_F(l2_stack_test, l2_stack_test_dtype_bool)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_BOOL, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，complex64
TEST_F(l2_stack_test, l2_stack_test_dtype_complex64)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_stack_test, ascend910B2_l2_stack_test_dtype_bfloat16)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常路径，hifloat8（仅950支持）
TEST_F(l2_stack_test, ascend950_l2_stack_test_dtype_hifloat8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常路径，float8_e5m2（仅950支持）
TEST_F(l2_stack_test, ascend950_l2_stack_test_dtype_float8_e5m2)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常路径，float8_e4m3fn（仅950支持）
TEST_F(l2_stack_test, ascend950_l2_stack_test_dtype_float8_e4m3fn)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 用例不支持，float8_e8m0不支持
TEST_F(l2_stack_test, l2_stack_test_dtype_float8_e8m0_not_support)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，float8_e4m3fn在910B不支持
TEST_F(l2_stack_test, l2_stack_test_dtype_float8_e4m3fn_910b_not_support)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，float32转int8
TEST_F(l2_stack_test, l2_stack_test_dtype_cast)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入tensors维度数量不同
TEST_F(l2_stack_test, l2_stack_test_different_dim_num)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入tensors维度不同
TEST_F(l2_stack_test, l2_stack_test_different_shape)
{
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，输入1个tensor
TEST_F(l2_stack_test, l2_stack_test_tensors_1)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_list_desc = TensorListDesc(1, tensor_1_desc);
    auto out_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，输入200个tensors
TEST_F(l2_stack_test, l2_stack_test_tensors_200)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_list_desc = TensorListDesc(200, tensor_1_desc);
    auto out_tensor_desc = TensorDesc({200, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 用例不支持，输入tensor维度超过8
TEST_F(l2_stack_test, l2_stack_test_9dim)
{
    auto tensor_1_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim测试
TEST_F(l2_stack_test, l2_stack_test_different_dim)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = -3;
    ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = 3;
    ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    dim = -4;
    ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入1个tensor且为空
TEST_F(l2_stack_test, l2_stack_test_one_empty_tensor)
{
    auto tensor_1_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc});
    auto out_tensor_desc = TensorDesc({1, 5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_stack_test, ascend310P_l2_stack_test_dtype_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2002);
    auto tensor_1_desc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND)
                             .Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 非连续输入用例 ====================
// 构造说明: view {5, 64} stride {65, 1} 即 [5, 65] 存储上每行取前 64 列的切片视图,
// 尾轴 stride=1、dim0 轴非连续; stack dim=1 后 view 为 [5, 1, 64], strides [65, 64, 1],
// 非连续轴(strideDim=0)恰好落在允许位置, 满足 aclnnCat 同款门控, 走 ConcatD 非连续路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_dim1)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 尾插轴 stack(dim=2) + 跨步视图: view {5, 64} stride {4096, 64}, 尾轴自身非连续(strideDim=1),
// 其余轴连续, 满足门控
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_dim_end)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {4096, 64}, 0, {5, 4084}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 64, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim=0 不满足非连续门控, 回退 Contiguous 常规路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_dim0)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({2, 5, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非 RegBase 平台(DAV_2201)不支持非连续直通, 回退常规路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_910b)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 33 个非连续 tensor: 超过 32 仍满足门控(单批 ConcatD 非连续支持 [2, 64])
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_33_tensors)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc(33, tensor_desc);
    auto out_tensor_desc = TensorDesc({5, 33, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 65 个非连续 tensor: 超过单批上限 64, 需按 64 分批两级 ConcatD
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_65_tensors)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc(65, tensor_desc);
    auto out_tensor_desc = TensorDesc({5, 65, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// complex128 不在底层 ConcatD AICore 非连续 tiling 支持列表内, 排除后回退常规路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_complex128)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_COMPLEX128, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_COMPLEX128, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入含 0 尺寸(空 tensor)时 shape 全一致即全空, 不满足门控, 回退常规路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_all_empty)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 断点轴 stride 超过 uint32 上限: 门控拒绝直通, 回退常规路径物化成功。
// 注: 超限 stride 放在 extent=1 的轴上(stride 永不被解引用), 视图才合法——
// 若放在 extent>=2 的轴上, 合法视图需 >=2^32 元素(约17GB)存储, UT 无法构造。
// 本例断点@轴1(stride=2^32, extent=1), dim=0 走 gather 族, 段长(64)在预算内,
// 门控在 stride 上限校验处拒绝(int64 域比较), 回退 Contiguous+Pack 成功
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_stride_exceed_uint32)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({2, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND, {64, 4294967296, 1}, 0, {2, 1, 64});
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({2, 2, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ==================== 非连续泛化用例 (dtype/rank/offset/混合连续性/边界) ====================
// dtype 不一致时门控拒绝, 回退常规路径(promote 后 Pack)
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_mixed_dtype)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_1_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_2_desc = TensorDesc({5, 64}, ACL_FLOAT16, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// rank-1 gather 视图 {64} stride {2}, 尾插轴后非连续轴落在 strideDim, 小包条件放行
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_rank1_gather)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 0, {128}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// view_offset 非零的非连续视图
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_offset)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 66, {6, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 一个连续 + 一个非连续混合输入, 门控要求至少一个非连续且其余校验通过
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_mixed_contiguity)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_1_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {64, 1}, 0, {5, 64}).ValueRange(-1, 1);
    auto tensor_2_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 尾轴 stride 非 1(多轴非连续), 门控在末轴校验处拒绝, 回退常规路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_multiaxis)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {130, 2}, 0, {6, 130}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// fp16 小包边界: 尾轴大包与 stride 大包均不满足, 靠总量小包放行
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_fp16_smalltotal)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {9, 1}, 0, {5, 9}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 128 个非连续 tensor: 64+64 两级 ConcatD
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_128_tensors)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {65, 1}, 0, {5, 65}).ValueRange(-1, 1);
    auto tensor_list_desc = TensorListDesc(128, tensor_desc);
    auto out_tensor_desc = TensorDesc({5, 128, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 0-dim 输入: dim 只能取 0(或归一化为 0), 门控必拒, 走未改动的回退常规路径
TEST_F(l2_stack_test, l2_stack_test_non_contiguous_zero_dim_tensor)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_desc, tensor_desc});
    auto out_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
