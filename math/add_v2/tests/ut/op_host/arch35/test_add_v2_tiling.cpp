/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_v2_tiling.cpp
 * \brief add_v2 tiling UT for ascend950 (arch35)
 */

#include "math/add_v2/op_host/arch35/add_v2_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class AddV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddV2Tiling TearDown" << std::endl; }
};

TEST_F(AddV2Tiling, add_v2_tiling_fp32)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_fp16_broadcast)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_bf16)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_int64)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_complex64)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_invalid_dtype)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
