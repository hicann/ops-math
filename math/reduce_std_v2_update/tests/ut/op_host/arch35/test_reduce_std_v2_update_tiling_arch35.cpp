/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

struct ReduceStdV2UpdateCompileInfo {};

class ReduceStdV2UpdateTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdV2UpdateTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceStdV2UpdateTiling TearDown" << std::endl; }
};

TEST_F(ReduceStdV2UpdateTiling, tiling_fp32_2d)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_fp16_all_reduce)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_bf16_negative_dim)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{8, 128, 128}, {8, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 128, 128}, {8, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_empty_r_correction_0)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_empty_r_correction_1)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2U, std::vector<size_t>{16777216});
}
