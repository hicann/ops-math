/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_unfold_grad_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../op_host/unfold_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class UnfoldGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UnfoldGradTilingTest  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "UnfoldGradTilingTest  TearDown" << std::endl;
    }
};

TEST_F(UnfoldGradTilingTest, UnfoldGradTilingData_test_float16_outputshape_8_2_dim_0_size_3_step_2_success_case0)
{
    optiling::Tiling4UnfoldGradCompileInfo compileInfo = {};
    std::vector<int64_t> inputSizeValues = {8, 2};
    gert::TilingContextPara tilingContextPara(
        "UnfoldGrad",
        {
            {{{3, 2, 3}, {3, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
        },
        {
            {{{8, 2}, {8, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("size", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
         gert::TilingContextPara::OpAttr("step", Ops::Math::AnyValue::CreateFrom<int64_t>(2))},
        &compileInfo);
    uint64_t expectTilingKey = 222;
    string expectTilingData = "1 1 1 177 1 52224 104448 16 18 3 2 3200 2 8 48 200 1 2 4 8 3 2 0 3 2 0 2 ";
    std::vector<size_t> expectWorkspaces = {16777280};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(
    UnfoldGradTilingTest,
    UnfoldGradTilingData_test_float16_outputshape_1_3_3_658_658_dim_4_size_20_step_2_success_case0)
{
    optiling::Tiling4UnfoldGradCompileInfo compileInfo = {};
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    gert::TilingContextPara tilingContextPara(
        "UnfoldGrad",
        {
            {{{1, 3, 3, 658, 320, 20}, {1, 3, 3, 658, 320, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
        },
        {
            {{{1, 3, 3, 658, 658}, {1, 3, 3, 658, 658}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(4)),
         gert::TilingContextPara::OpAttr("size", Ops::Math::AnyValue::CreateFrom<int64_t>(20)),
         gert::TilingContextPara::OpAttr("step", Ops::Math::AnyValue::CreateFrom<int64_t>(2))},
        &compileInfo);
    uint64_t expectTilingKey = 221;
    string expectTilingData =
        "5922 93 63 4 64 52224 104448 658 6400 0 6400 25920 5 0 1 0 128000 2 4 8 320 658 4 20 2 0 6400 ";
    std::vector<size_t> expectWorkspaces = {32363920};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(
    UnfoldGradTilingTest,
    UnfoldGradTilingData_test_float32_outputshape_1_3_3_658_658_dim_3_size_2_step_20_success_case0)
{
    optiling::Tiling4UnfoldGradCompileInfo compileInfo = {};
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    gert::TilingContextPara tilingContextPara(
        "UnfoldGrad",
        {
            {{{1, 3, 3, 33, 658, 2}, {1, 3, 3, 33, 658, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
        },
        {
            {{{1, 3, 3, 658, 658}, {1, 3, 3, 658, 658}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
         gert::TilingContextPara::OpAttr("size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("step", Ops::Math::AnyValue::CreateFrom<int64_t>(20))},
        &compileInfo);
    uint64_t expectTilingKey = 312;
    string expectTilingData = "9 1 1 0 9 0 131072 432964 43428 33 658 4096 5 8 16 512 83 4 4 8 33 658 3 2 20 0 658 ";
    std::vector<size_t> expectWorkspaces = {32363920};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}