/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"

#include "math/floor_mod/op_api/aclnn_remainder.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

class l2_remainder_scalar_tensor_ascend950_test : public testing::Test {
protected:
    void TearDown() override { SetPlatformNpuArch(NpuArch::DAV_2201); }
};

TEST_F(l2_remainder_scalar_tensor_ascend950_test, double_scalar_int32_tensor_to_float16)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);

    auto self = ScalarDesc(2049.0);
    auto other = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(std::vector<int32_t>{2});
    auto out = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0, 0.0);
    auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, integer_scalar_unsupported_compute_dtype)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    for (auto otherDtype : {ACL_INT8, ACL_UINT8, ACL_INT16}) {
        for (auto outDtype : {ACL_INT32, ACL_FLOAT}) {
            SCOPED_TRACE(static_cast<int>(otherDtype));
            SCOPED_TRACE(static_cast<int>(outDtype));
            auto self = ScalarDesc(int64_t{7});
            auto other = TensorDesc({4}, otherDtype, ACL_FORMAT_ND).ValueRange(2, 5);
            auto out = TensorDesc({4}, outDtype, ACL_FORMAT_ND);
            auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
            uint64_t workspaceSize = 0;
            EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
        }
    }
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, float_scalar_promotes_int8_tensor)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self = ScalarDesc(7.0f);
    auto other = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(2, 5);
    auto out = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, non_regbase_widens_int8_tensor_to_output_dtype)
{
    SetPlatformNpuArch(NpuArch::DAV_2201);
    auto self = ScalarDesc(int64_t{7});
    auto other = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(2, 5);
    auto out = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, empty_tensor_rejects_unsupported_compute_dtype)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self = ScalarDesc(int64_t{7});
    auto other = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, supported_compute_dtypes)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    for (auto dtype : {ACL_INT32, ACL_INT64, ACL_FLOAT16, ACL_FLOAT, ACL_DOUBLE, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto self = ScalarDesc(int64_t{7});
        auto other = TensorDesc({4}, dtype, ACL_FORMAT_ND).ValueRange(2, 5);
        auto out = TensorDesc({4}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, empty_tensor_supported_compute_dtype)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self = ScalarDesc(int64_t{7});
    auto other = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(l2_remainder_scalar_tensor_ascend950_test, unsupported_output_dtype)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self = ScalarDesc(int64_t{7});
    auto other = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(2, 5);
    auto out = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRemainderScalarTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
