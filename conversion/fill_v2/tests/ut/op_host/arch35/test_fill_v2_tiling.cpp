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
#include "../../../../op_host/arch35/fill_v2_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_format.h"

using namespace std;
using namespace ge;

class FillV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FillV2TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "FillV2TilingTest TearDown" << std::endl; }
};

TEST_F(FillV2TilingTest, fill_v2_test_fp32)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 2;
    string expectTilingData = "4096 140737488355332 1073741824 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_fp16)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 1;
    string expectTilingData = "4096 281474976710658 16384 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_int32)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 6;
    string expectTilingData = "4096 140737488355332 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_double)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "4096 70368744177672 4611686018427387904 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_int64)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "4096 70368744177672 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_int8)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 4;
    string expectTilingData = "4096 562949953421313 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_int16)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 5;
    string expectTilingData = "4096 281474976710658 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillV2TilingTest, fill_v2_test_invalid_dtype)
{
    optiling::FillV2CompileInfo compile_info = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "FillV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compile_info);

    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
