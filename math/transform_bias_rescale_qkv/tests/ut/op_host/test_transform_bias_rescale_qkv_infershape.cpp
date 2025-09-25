/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "../../../op_graph/transform_bias_rescale_qkv_proto.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class TransformBiasRescaleQkv : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransformBiasRescaleQkv SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TransformBiasRescaleQkv TearDown" << std::endl;
    }
};

TEST_F(TransformBiasRescaleQkv, TransformBiasRescaleQkv_infershape_case_0)
{
    ge::op::TransformBiasRescaleQkv op;
    op.UpdateInputDesc("qkv", create_desc({3, 4, 144}, ge::DT_FLOAT16));
    op.UpdateInputDesc("qkv_bias", create_desc({144}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}