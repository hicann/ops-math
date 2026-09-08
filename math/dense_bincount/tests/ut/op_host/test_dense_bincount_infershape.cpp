/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class dense_bincount : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "dense_bincount SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "dense_bincount TearDown" << std::endl; }
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape shape;
    for (const int64_t dimension : dimensions) {
        shape.MutableOriginShape().AppendDim(dimension);
        shape.MutableStorageShape().AppendDim(dimension);
    }
    return shape;
}

TEST_F(dense_bincount, dense_bincount_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara("DenseBincount",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(dense_bincount, dense_bincount_infershape_test2)
{
    std::vector<int64_t> inputSizeValues = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "DenseBincount",
        {
            {{{1, 3}, {1, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(dense_bincount, dense_bincount_unknown_rank)
{
    int32_t sizeValue = 4;
    gert::InfershapeContextPara context(
        "DenseBincount",
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &sizeValue}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-2}});
}

TEST_F(dense_bincount, dense_bincount_empty_input)
{
    int32_t sizeValue = 3;
    gert::StorageShape emptyShape({0}, {0});
    gert::InfershapeContextPara context(
        "DenseBincount",
        {{emptyShape, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &sizeValue}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("binary_output", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3}});
}

TEST_F(dense_bincount, dense_bincount_empty_input_each_axis)
{
    int32_t sizeValue = 3;
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> cases = {
        {{2, 0}, {2, 3}},
        {{0, 2}, {0, 3}},
        {{0, 0}, {0, 3}},
    };
    for (const auto& [inputShape, outputShape] : cases) {
        gert::StorageShape shape = MakeStorageShape(inputShape);
        gert::InfershapeContextPara context(
            "DenseBincount",
            {{shape, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &sizeValue}},
            {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
        ExecuteTestCase(context, ge::GRAPH_SUCCESS, {outputShape});
    }
}

TEST_F(dense_bincount, dense_bincount_zero_bins)
{
    int32_t sizeValue = 0;
    gert::InfershapeContextPara context(
        "DenseBincount",
        {{{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &sizeValue}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 0}});
}

TEST_F(dense_bincount, dense_bincount_rejects_invalid_rank)
{
    int32_t sizeValue = 3;
    gert::InfershapeContextPara context("DenseBincount",
                                        {{{{1, 2, 3}, {1, 2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &sizeValue}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(dense_bincount, dense_bincount_rejects_output_size_overflow)
{
    int64_t sizeValue = std::numeric_limits<int64_t>::max() / sizeof(float) / 2 + 1;
    gert::InfershapeContextPara context(
        "DenseBincount",
        {{{{2, 1}, {2, 1}}, ge::DT_INT64, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &sizeValue}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
