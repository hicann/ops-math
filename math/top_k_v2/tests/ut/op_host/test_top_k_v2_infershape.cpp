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
 * \file test_strided_slice_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include <vector>

using namespace std;

namespace {

gert::InfershapeContextPara MakeTopKV2Case(int32_t k, int64_t dim = -1)
{
    static int32_t kValue;
    kValue = k;
    return gert::InfershapeContextPara("TopKV2",
                                       {
                                           {{{2, 32}, {2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, &kValue},
                                       },
                                       {
                                           {{{2, 32}, {2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{2, 32}, {2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
                                       },
                                       {
                                           {"sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)},
                                           {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)},
                                           {"largest", Ops::Math::AnyValue::CreateFrom<bool>(true)},
                                           {"indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)},
                                       });
}

} // namespace

class TopKV2InferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendTopKV2Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AscendTopKV2Test TearDown" << std::endl; }
};

TEST_F(TopKV2InferShape, top_k_v2_infershape_test1)
{
    vector<int64_t> k = {8};

    gert::InfershapeContextPara infershapeContextPara(
        "TopKV2",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 8}, {10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 8}, {10, 8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"largest", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TopKV2InferShape, validation_k_equal_axis_is_accepted)
{
    auto infershapeContextPara = MakeTopKV2Case(32);
    std::cout << "validation_case k=32 expected=GRAPH_SUCCESS" << std::endl;
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 32}, {2, 32}});
}

TEST_F(TopKV2InferShape, validation_k_greater_than_axis_is_rejected)
{
    auto infershapeContextPara = MakeTopKV2Case(33);
    std::cout << "validation_case k=33 expected=GRAPH_FAILED" << std::endl;
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKV2InferShape, validation_negative_k_is_rejected)
{
    auto infershapeContextPara = MakeTopKV2Case(-1);
    std::cout << "validation_case k=-1 expected=GRAPH_FAILED" << std::endl;
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKV2InferShape, validation_dim_below_negative_rank_is_rejected)
{
    auto infershapeContextPara = MakeTopKV2Case(1, -3);
    std::cout << "validation_case dim=-3 expected=GRAPH_FAILED" << std::endl;
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
