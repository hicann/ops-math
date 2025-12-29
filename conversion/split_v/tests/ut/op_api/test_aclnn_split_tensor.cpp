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
#include "gtest/gtest.h"

#include "conversion/split_v/op_api/aclnn_split_tensor.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_split_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "split_tensor_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "split_tensor_test TearDown" << endl;
    }
};

TEST_F(l2_split_tensor_test, aclnnSplitTensor_8_4_float32_split_2_dim_0)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t split_sections = 4;
    int64_t dim = 0;
    auto out1 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(self_desc, split_sections, dim), OUTPUT(out_list));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}