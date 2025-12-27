/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "math/floor_mod/op_api/aclnn_remainder.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "acl/acl.h"

using namespace op;
using namespace std;

class l2_remainder_inplace_tensor_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "remainder_inplace_tensor_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "remainder_inplace_tensor_scalar_test TearDown" << std::endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        ScalarDesc other)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat)
                        .ValueRange(selfRange[0], selfRange[1])
                        .Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(self, other), OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        ScalarDesc other)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat)
                        .ValueRange(selfRange[0], selfRange[1])
                        .Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(self, other), OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// self + other + out: int32
TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_01)
{
    int32_t value = 3;
    auto other_desc = ScalarDesc(value);
    test_run({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);
}

// self + other + out: 不支持self和other都为bool、uint8、int8、int16、bfloat16、complex64、complex128
TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_06)
{
    bool value_bool = true;
    uint8_t value_uint8 = 3;
    int8_t value_int8 = 3;
    int16_t value_int16 = 3;
    auto other_desc_bool = ScalarDesc(value_bool);
    auto other_desc_uint8 = ScalarDesc(value_uint8);
    auto other_desc_int8 = ScalarDesc(value_int8);
    auto other_desc_int16 = ScalarDesc(value_int16);

    test_run_invalid({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, other_desc_bool);
    test_run_invalid({2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
    test_run_invalid({2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, other_desc_int8);
    test_run_invalid({2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, other_desc_int16);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
}

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_09)
{
    uint64_t workspaceSize = 0;
    auto other_desc = ScalarDesc(7.86f);
    auto self = TensorDesc({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.00001, 0.00001);

    auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(nullptr, other_desc), OUTPUT());
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(self, nullptr), OUTPUT());
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_10)
{
    // self emtpy, self.dtype = promoteType
    int32_t value_int32 = 7;
    auto other_desc = ScalarDesc(value_int32);
    test_run({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);

    // self emtpy, self.dtype != promoteType
    int64_t value_int64 = 7;
    other_desc = ScalarDesc(value_int64);
    test_run({2, 0, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);
}