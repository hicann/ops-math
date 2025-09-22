/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "index.h"
#include "common/utils/ut_op_common.h"
#include "util/util.h"

using namespace ge;

class CoalesceSparseTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CoalesceSparseTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CoalesceSparseTest TearDown" << std::endl;
    }
};

TEST_F(CoalesceSparseTest, coalesce_sparse_test_success)
{
    ge::op::CoalesceSparse op;
    op.UpdateInputDesc("unique_len", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
    op.UpdateInputDesc("unique_indices", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({5, 2}, ge::DT_INT32, ge::FORMAT_ND, {5, 2}, ge::FORMAT_ND));
    op.UpdateInputDesc("values", create_desc_with_ori({5, 3}, ge::DT_INT32, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND));
    std::vector<int64_t> expected_output_shape0 = {4, 2};
    std::vector<int64_t> expected_output_shape1 = {4, 3};
    Runtime2TestParam coalesce_sparse{{}};
    EXPECT_EQ(InferShapeTest(op, coalesce_sparse), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDesc(0);
    auto output1_desc = op.GetOutputDesc(1);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape0);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output_shape1);
}

TEST_F(CoalesceSparseTest, coalesce_sparse_infer_datatype_success)
{
    ge::op::CoalesceSparse op;
    op.UpdateInputDesc("unique_len", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
    op.UpdateInputDesc("unique_indices", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({5, 2}, ge::DT_INT32, ge::FORMAT_ND, {5, 2}, ge::FORMAT_ND));
    op.UpdateInputDesc("values", create_desc_with_ori({5, 3}, ge::DT_INT32, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND));
    std::vector<int64_t> expected_output_shape0 = {4, 2};
    std::vector<int64_t> expected_output_shape1 = {4, 3};
    Runtime2TestParam coalesce_sparse{{}};
    EXPECT_EQ(InferDataTypeTest(op, coalesce_sparse), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDesc(0);
    auto output1_desc = op.GetOutputDesc(1);
    EXPECT_EQ(output0_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(output1_desc.GetDataType(), ge::DT_INT32);
}
