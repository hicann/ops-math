/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_bitwise_and_dtype_contract.cpp
 * \brief Dtype capability contract of BitwiseAnd aclnn APIs.
 *
 * The expectations here are the contract shared by the operator docs and the business code
 * (op_api/aclnn_bitwise_and_tensor.cpp, aclnn_bitwise_and_scalar.cpp, aclnn_bitwiseand.cpp),
 * for RegBase (Ascend 950 family) products:
 *  - aclnnBitwiseAndTensorOut / aclnnInplaceBitwiseAndTensorOut:
 *      BOOL + 8 integer dtypes (INT8/16/32/64, UINT8/16/32/64).
 *  - aclnnBitwiseAndTensor / aclnnInplaceBitwiseAndTensor:
 *      BOOL + 8 integer dtypes (INT8/16/32/64, UINT8/16/32/64).
 *  - aclnnBitwiseAndScalar / aclnnInplaceBitwiseAndScalar:
 *      BOOL + 8 integer dtypes (INT8/16/32/64, UINT8/16/32/64).
 */

#include <cstdint>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_bitwise_and_tensor.h"
#include "../../../op_api/aclnn_bitwise_and_scalar.h"
#include "../../../op_api/aclnn_bitwiseand.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_bitwise_and_dtype_contract : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "bitwise_and dtype contract SetUp" << endl; }
    static void TearDownTestCase() { cout << "bitwise_and dtype contract TearDown" << endl; }
};

#define EXPECT_TENSOR_OK(api, DT)                             \
    do {                                                      \
        auto d = TensorDesc({2, 3}, DT, ACL_FORMAT_ND);       \
        auto ut = OP_API_UT(api, INPUT(d, d), OUTPUT(d));     \
        uint64_t ws = 0;                                      \
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS); \
    } while (0)

#define EXPECT_INPLACE_TENSOR_OK(api, DT)                     \
    do {                                                      \
        auto d = TensorDesc({2, 3}, DT, ACL_FORMAT_ND);       \
        auto ut = OP_API_UT(api, INPUT(d, d), OUTPUT());      \
        uint64_t ws = 0;                                      \
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS); \
    } while (0)

#define EXPECT_SCALAR_OK(api, DT, SVAL)                       \
    do {                                                      \
        auto d = TensorDesc({2, 3}, DT, ACL_FORMAT_ND);       \
        auto s = ScalarDesc(SVAL);                            \
        auto ut = OP_API_UT(api, INPUT(d, s), OUTPUT(d));     \
        uint64_t ws = 0;                                      \
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS); \
    } while (0)

#define EXPECT_SCALAR_INVALID(api, DT, SVAL)                              \
    do {                                                                  \
        auto d = TensorDesc({2, 3}, DT, ACL_FORMAT_ND);                   \
        auto s = ScalarDesc(SVAL);                                        \
        auto ut = OP_API_UT(api, INPUT(d, s), OUTPUT(d));                 \
        uint64_t ws = 0;                                                  \
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID); \
    } while (0)

#define EXPECT_INPLACE_SCALAR_OK(api, DT, SVAL)               \
    do {                                                      \
        auto d = TensorDesc({2, 3}, DT, ACL_FORMAT_ND);       \
        auto s = ScalarDesc(SVAL);                            \
        auto ut = OP_API_UT(api, INPUT(d, s), OUTPUT());      \
        uint64_t ws = 0;                                      \
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS); \
    } while (0)

// ---- Tensor APIs: BOOL + 8 integer dtypes ----
TEST_F(l2_bitwise_and_dtype_contract, tensor_all_supported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_BOOL);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_INT8);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_UINT8);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_INT16);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_UINT16);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_INT32);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_UINT32);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_INT64);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensor, ACL_UINT64);
}

TEST_F(l2_bitwise_and_dtype_contract, inplace_tensor_all_supported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_BOOL);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_INT8);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_UINT8);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_INT16);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_UINT16);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_INT32);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_UINT32);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_INT64);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_UINT64);
}

// ---- Out APIs: BOOL + 8 integer dtypes (same as the tensor APIs) ----
TEST_F(l2_bitwise_and_dtype_contract, tensor_out_contract)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_BOOL);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_INT8);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_UINT8);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_INT16);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_UINT16);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_INT32);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_INT64);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_UINT32);
    EXPECT_TENSOR_OK(aclnnBitwiseAndTensorOut, ACL_UINT64);
}

TEST_F(l2_bitwise_and_dtype_contract, inplace_tensor_out_contract)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_BOOL);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_INT8);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_UINT8);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_INT16);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_UINT16);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_INT32);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_INT64);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_UINT32);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensorOut, ACL_UINT64);
}

// ---- Scalar APIs on RegBase (Ascend 950): UINT32/UINT64 additionally supported ----
TEST_F(l2_bitwise_and_dtype_contract, scalar_regbase_contract)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    EXPECT_SCALAR_OK(aclnnBitwiseAndScalar, ACL_UINT32, uint32_t(1));
    EXPECT_SCALAR_OK(aclnnBitwiseAndScalar, ACL_UINT64, uint64_t(1));
    EXPECT_INPLACE_SCALAR_OK(aclnnInplaceBitwiseAndScalar, ACL_UINT32, uint32_t(1));
    EXPECT_INPLACE_SCALAR_OK(aclnnInplaceBitwiseAndScalar, ACL_UINT64, uint64_t(1));
}

// ---- RegBase (Ascend 950) rejection paths ----
TEST_F(l2_bitwise_and_dtype_contract, regbase_mixed_rejections)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    // Scalar: on RegBase the only rejected input combinations are BOOL self with a UINT16/32/64 scalar
    EXPECT_SCALAR_INVALID(aclnnBitwiseAndScalar, ACL_BOOL, uint16_t(1));
    EXPECT_SCALAR_INVALID(aclnnBitwiseAndScalar, ACL_BOOL, uint32_t(1));
    EXPECT_SCALAR_INVALID(aclnnBitwiseAndScalar, ACL_BOOL, uint64_t(1));
    // Inplace tensor: selfRef=BOOL may only pair with a BOOL other
    {
        auto a = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnInplaceBitwiseAndTensor, INPUT(a, b), OUTPUT());
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    // Inplace tensor: selfRef=UINT32 pairs with UINT32 (RegBase accepts UINT32/UINT64 inputs)
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_UINT32);
    EXPECT_INPLACE_TENSOR_OK(aclnnInplaceBitwiseAndTensor, ACL_UINT64);
    // Inplace scalar: selfRef=BOOL may only pair with a BOOL scalar
    {
        auto a = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto s = ScalarDesc(int8_t(1));
        auto ut = OP_API_UT(aclnnInplaceBitwiseAndScalar, INPUT(a, s), OUTPUT());
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    // A dtype outside the support list (FLOAT) is rejected
    {
        auto a = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, a), OUTPUT(a));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
}

// ---- Shape / dim / format contract ----
TEST_F(l2_bitwise_and_dtype_contract, shape_dim_format_contract)
{
    {
        // <= 8 dims supported
        auto d8 = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(d8, d8), OUTPUT(d8));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS);
    }
    {
        // > 8 dims rejected
        auto d9 = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(d9, d9), OUTPUT(d9));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    {
        // private format rejected
        auto d = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NC1HWC0);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(d, d), OUTPUT(d));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    {
        // non-broadcastable shapes rejected
        auto a = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
        auto b = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    {
        // scalar: self and out shape must be equal
        auto a = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
        auto o = TensorDesc({3, 2}, ACL_INT32, ACL_FORMAT_ND);
        auto s = ScalarDesc(int32_t(1));
        auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(a, s), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
}

// ---- Mixed dtype promotion and out-dtype cast rules ----
TEST_F(l2_bitwise_and_dtype_contract, mixed_promote_and_out_cast)
{
    {
        auto a = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS);
    }
    {
        auto a = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS);
    }
    {
        // uint16/uint32/uint64 only combine with the same dtype -> promote is DT_UNDEFINED
        auto a = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    {
        // int -> float out is allowed by CanCast
        auto a = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS);
    }
    {
        // int -> bool out is rejected
        auto a = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
    {
        // bool promote -> float out is allowed by CanCast
        auto a = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACL_SUCCESS);
    }
    {
        // bool x uint16 cannot promote (uint16 combines only with the same dtype)
        auto a = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        auto b = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
        auto o = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(a, b), OUTPUT(o));
        uint64_t ws = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
    }
}
