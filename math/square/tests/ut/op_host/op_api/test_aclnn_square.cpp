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
#include "level2/aclnn_square.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"
#include "opdev/platform.h"

using namespace std;

class l2_square_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_square_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_square_test TearDown" << endl;
    }
};

// test nullptr
TEST_F(l2_square_test, ascend910_95_case_nullptr)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(nullptr), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(nullptr));
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test dtype: float32
TEST_F(l2_square_test, ascend910_95_case_dtype_fp32)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector{1, 2, 3, 4});
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// test dtype: bfloat16
TEST_F(l2_square_test, ascend910_95_case_dtype_bf16)
{
    auto self = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test dtype: int64
TEST_F(l2_square_test, ascend910_95_case_dtype_int64)
{
    auto self = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非法dype：不支持的dtype
TEST_F(l2_square_test, ascend910_95_case_dtype_invalid_0)
{
    auto self = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype：dtype不一致
TEST_F(l2_square_test, ascend910_95_case_dtype_invalid_1)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format：format超8维
TEST_F(l2_square_test, ascend910_95_case_format_invalid_0)
{
    auto self = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format：format不一致
TEST_F(l2_square_test, ascend910_95_case_format_invalid_1)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape：shape不一致
TEST_F(l2_square_test, ascend910_95_case_shape_invalid)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}