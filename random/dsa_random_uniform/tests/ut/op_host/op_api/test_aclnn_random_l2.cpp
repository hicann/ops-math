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

#include "../../../../op_host/op_api/aclnn_random.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

class l2_random_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_random_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_random_test TearDown" << endl;
    }
};

TEST_F(l2_random_test, aclnnInplaceRandom_3_4_float_nd)
{
    const vector<int64_t>& self_shape = {3, 4};
    aclDataType self_dtype = ACL_FLOAT;
    aclFormat self_format = ACL_FORMAT_ND;

    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {3, 4};
    aclDataType out_dtype = ACL_FLOAT;
    aclFormat out_format = ACL_FORMAT_ND;

    auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(-5, 5);
    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_random_test, ascend910B2_aclnnInplaceRandom_3_4_bfloat16_nd)
{
    const vector<int64_t>& self_shape = {3, 4};
    aclDataType self_dtype = ACL_BF16;
    aclFormat self_format = ACL_FORMAT_ND;

    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& out_shape = {3, 4};
    aclDataType out_dtype = ACL_BF16;
    aclFormat out_format = ACL_FORMAT_ND;

    auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(-5, 5);
    auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_random_test, aclnnInplaceRandom_1_11_34_22_19_6_5_1_float16_nchw_23_11_34_22_1_1_5_5_float16_nchw)
{
    const vector<int64_t>& selfShape = {1, 11};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;

    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    // output
    const vector<int64_t>& outShape = {1, 11};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT(selfTensorDesc, from, to, seed, offset), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_random_test, aclnnInplaceRandom_5_4_float32_not_contiguous)
{
    const vector<int64_t>& selfShape = {5, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& selfViewDim = {1, 5};
    int64_t selfOffset = 0;
    const vector<int64_t>& selfStorageDim = {4, 5};

    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& outShape = {5, 4};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT(selfTensorDesc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_random_test, aclnnInplaceRandom_9_float32_)
{
    const vector<int64_t>& selfShape = {5, 4, 5, 4, 5, 4, 5, 4, 5};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& outShape = {5, 4, 5, 4, 5, 4, 5, 4, 5};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT(selfTensorDesc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_random_test, aclnnInplaceRandom_case14)
{
    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& outShape = {5, 4};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT((aclTensor*)nullptr, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_random_test, ascend910B2_aclnnInplaceRandom_5_4_float32_not_contiguous)
{
    const vector<int64_t>& selfShape = {5, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& selfViewDim = {1, 5};
    int64_t selfOffset = 0;
    const vector<int64_t>& selfStorageDim = {4, 5};

    int64_t from = 2;
    int64_t to = 4;
    int64_t seed = 0;
    int64_t offset = 0;

    const vector<int64_t>& outShape = {5, 4};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnInplaceRandom, INPUT(selfTensorDesc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}