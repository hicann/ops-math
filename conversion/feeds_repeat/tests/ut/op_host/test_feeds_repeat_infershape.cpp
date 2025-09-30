/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class FeedsRepeat : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FeedsRepeat Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FeedsRepeat Proto Test TearDown" << std::endl;
    }
};

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_1)
{
    ge::op::FeedsRepeat op;

    op.UpdateInputDesc("feeds", create_desc({4, 5, 6, 7}, ge::DT_FLOAT));
    op.UpdateInputDesc("feeds_repeat_times", create_desc({4}, ge::DT_INT32));
    op.SetAttr("output_feeds_size", 15);
    Runtime2TestParam rt_param;
    rt_param.attrs = {"output_feeds_size"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_y_shape = {15, 5, 6, 7};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_2)
{
    ge::op::FeedsRepeat op;

    op.UpdateInputDesc("feeds", create_desc({1, 50, 65}, ge::DT_FLOAT16));
    op.UpdateInputDesc("feeds_repeat_times", create_desc({1}, ge::DT_INT32));
    op.SetAttr("output_feeds_size", 128);
    Runtime2TestParam rt_param;
    rt_param.attrs = {"output_feeds_size"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_y_shape = {128, 50, 65};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_3)
{
    ge::op::FeedsRepeat op;

    op.UpdateInputDesc("feeds", create_desc({4, 50, 60, 7}, ge::DT_BF16));
    op.UpdateInputDesc("feeds_repeat_times", create_desc({4}, ge::DT_INT32));
    op.SetAttr("output_feeds_size", 15);
    Runtime2TestParam rt_param;
    rt_param.attrs = {"output_feeds_size"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_y_shape = {15, 50, 60, 7};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_101)
{
    ge::op::FeedsRepeat op;

    op.UpdateInputDesc("feeds", create_desc({48, 5, 6, 7}, ge::DT_FLOAT));
    op.UpdateInputDesc("feeds_repeat_times", create_desc({48}, ge::DT_INT64));
    op.SetAttr("output_feeds_size", 100);
    Runtime2TestParam rt_param;
    rt_param.attrs = {"output_feeds_size"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_y_shape = {100, 5, 6, 7};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_102)
{
    ge::op::FeedsRepeat op;

    op.UpdateInputDesc("feeds", create_desc({50, 5, 6, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("feeds_repeat_times", create_desc({50}, ge::DT_INT64));
    op.SetAttr("output_feeds_size", 50);
    Runtime2TestParam rt_param;
    rt_param.attrs = {"output_feeds_size"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_y_shape = {50, 5, 6, 7};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_103)
{
    ge::op::FeedsRepeat op;

    op.UpdateInputDesc("feeds", create_desc({100, 5, 6, 7}, ge::DT_BF16));
    op.UpdateInputDesc("feeds_repeat_times", create_desc({100}, ge::DT_INT64));
    op.SetAttr("output_feeds_size", 101);
    Runtime2TestParam rt_param;
    rt_param.attrs = {"output_feeds_size"};
    rt_param.input_const = {false, true};
    EXPECT_EQ(InferShapeTest(op, rt_param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_y_shape = {101, 5, 6, 7};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}
