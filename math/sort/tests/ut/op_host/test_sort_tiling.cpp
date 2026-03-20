/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class SortTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

struct SortCompileInfo {
    int32_t core_num = 64;
    int32_t num_block = 1;
    int32_t num_offset = 0;
    int32_t float_bytes = 4;
};

TEST_F(SortTilingTest, test_sort_basic_tiling_fp32_2x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1024}, {2, 1024}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_1d_small_fp32_512) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_2d_batch_small_fp32_32x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 1024}, {32, 1024}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_big_batch_fp32_128x8192) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{128, 8192}, {128, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 8192}, {128, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 8192}, {128, 8192}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 257;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_descending_fp32_2x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1024}, {2, 1024}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 65792;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_fp16_2x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024}, {2, 1024}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_int32_index_fp32_2x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1024}, {2, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_boundary_4096) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{2, 4096}, {2, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4096}, {2, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096}, {2, 4096}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_boundary_4097) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4097}, {2, 4097}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 257;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_3d_shape_fp32_8x16x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{8, 16, 1024}, {8, 16, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 16, 1024}, {8, 16, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 16, 1024}, {8, 16, 1024}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_bf16_32x1024) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{32, 1024}, {32, 1024}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_big_batch_fp32_64x16384) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{64, 16384}, {64, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 16384}, {64, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 16384}, {64, 16384}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 260;
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SortTilingTest, test_sort_merge_big_batch_fp32_128x32768) {
    SortCompileInfo compileInfo = {64, 1, 0, 4};
    gert::TilingContextPara tilingContextPara(
        "Sort",
        {
            {{{128, 32768}, {128, 32768}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 32768}, {128, 32768}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 32768}, {128, 32768}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("descending", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("stable", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 260;
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}