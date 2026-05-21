/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

using OpAttr = gert::InfershapeContextPara::OpAttr;

class ReshapeInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReshapeInfershapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ReshapeInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(ReshapeInfershapeTest, reshape_basic_success)
{
    int64_t shape_value[2] = {3, 2};
    gert::InfershapeContextPara infershape_context_para(
        "Reshape",
        {{{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &shape_value}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershape_context_para, ge::GRAPH_SUCCESS, {{3, 2}});
}

TEST_F(ReshapeInfershapeTest, reshape_infer_minus_one_success)
{
    int64_t shape_value[2] = {6, -1};
    gert::InfershapeContextPara infershape_context_para(
        "Reshape",
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &shape_value}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    ExecuteTestCase(infershape_context_para, ge::GRAPH_SUCCESS, {{6, 4}});
}

TEST_F(ReshapeInfershapeTest, reshape_copy_zero_success)
{
    int64_t shape_value[2] = {0, -1};
    gert::InfershapeContextPara infershape_context_para(
        "Reshape",
        {{{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &shape_value}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    ExecuteTestCase(infershape_context_para, ge::GRAPH_SUCCESS, {{2, 3}});
}

TEST_F(ReshapeInfershapeTest, reshape_axis_num_axes_success)
{
    int64_t shape_value[1] = {12};
    gert::InfershapeContextPara infershape_context_para(
        "Reshape",
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &shape_value}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershape_context_para, ge::GRAPH_SUCCESS, {{2, 12}});
}

TEST_F(ReshapeInfershapeTest, reshape_allowzero_success)
{
    int64_t shape_value[2] = {0, 3};
    gert::InfershapeContextPara infershape_context_para(
        "Reshape",
        {{{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &shape_value}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         OpAttr("allowzero", Ops::Math::AnyValue::CreateFrom<int64_t>(1))});
    ExecuteTestCase(infershape_context_para, ge::GRAPH_SUCCESS, {{0, 3}});
}

TEST_F(ReshapeInfershapeTest, reshape_invalid_double_minus_one_failed)
{
    int64_t shape_value[3] = {-1, 2, -1};
    gert::InfershapeContextPara infershape_context_para(
        "Reshape",
        {{{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &shape_value}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershape_context_para, ge::GRAPH_FAILED);
}