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
 * \file test_stateless_drop_out_gen_mask_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/stateless_drop_out_gen_mask_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class StatelessDropOutGenMaskTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessDropOutGenMaskTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessDropOutGenMaskTilingTest TearDown" << std::endl;
    }
};

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_0)
{
    optiling::StatelessDropOutGenMaskCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    vector<float> probValue = {1.0};
    vector<int64_t> seedValue = {2};
    vector<int64_t> seed1Value = {0};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seed1Value.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 2048 }, { 2048 } },ge::DT_UINT8, ge::FORMAT_ND},
    }, 
    &compileInfo);
    uint64_t expectTilingKey = 1001;
    string expectTilingData = "32 512 512 2 2 256 2 0 8 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
