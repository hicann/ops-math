/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file non_finite_check.cpp
 * \brief
 */
 
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "experiment_ops.h"     // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class NonFiniteCheck : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "NonFiniteCheck SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NonFiniteCheck TearDown" << std::endl;
    }
};

TEST_F(NonFiniteCheck, NonFiniteCheck_InferDataType)
{
    ge::op::NonFiniteCheck op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
    auto tensor_desc =
        create_desc_shape_range({2, 100, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND, shape_range);
    op.create_dynamic_input_tensor_list(3);
    op.UpdateDynamicInputDesc("tensor_list", 0, tensor_desc);
    op.UpdateDynamicInputDesc("tensor_list", 1, tensor_desc);
    op.UpdateDynamicInputDesc("tensor_list", 2, tensor_desc);
    auto ret = InferDataTypeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("found_flag");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}