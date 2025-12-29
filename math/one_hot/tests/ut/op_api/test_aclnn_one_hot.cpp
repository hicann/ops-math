/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_one_hot.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_one_hot.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_one_hot_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_one_hot_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_one_hot_test TearDown" << std::endl;
    }
};

TEST_F(l2_one_hot_test, ascend910B2_l2_one_hot_test_input_invalid_1)
{
    auto selfDesc = nullptr;
    auto outDesc = TensorDesc({2, 2, 11}, ACL_INT64, ACL_FORMAT_ND);
    int64_t numClasses = 11;
    auto onValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    onValue.ValueRange(1, 2);
    auto offValue = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    offValue.ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnOneHot, INPUT(selfDesc, numClasses, onValue, offValue, -1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}
