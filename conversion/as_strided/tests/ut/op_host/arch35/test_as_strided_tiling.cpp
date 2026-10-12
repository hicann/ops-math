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
 * \file test_as_strided_tiling.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/as_strided_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AsStridedTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(AsStridedTiling, as_strided_tiling_test_case1)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{3}, {3}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case2)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case3)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case4)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2, 6, 6};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case5)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2, 6, 6};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{7}, {7}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case6)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2, 6, 6};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case7)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{1000, 1000, 512}, {1000, 1000, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{124, 2, 80}, {124, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case8)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{1000, 1000, 512}, {1000, 1000, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{124, 2, 80}, {124, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case9)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {1};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{3, 3, 3, 3}, {3, 3, 3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
                                               {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()}},
                                              {
                                                  {
                                                      {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// storage元素数超过uint32上限(4294967295)时, 校验不应回绕误报越界, 应正常走MoveAlign模板(fp16: 100+2)
TEST_F(AsStridedTiling, as_strided_tiling_test_storage_size_over_uint32)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {5, 2, 8, 512, 32};
    std::vector<int64_t> strideValues = {540672, 131072, 16384, 32, 1};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4295639040}, {4295639040}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{1310720}, {1310720}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// storage元素数等于2^32边界, 校验不回绕
TEST_F(AsStridedTiling, as_strided_tiling_test_storage_size_equal_2p32)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {5, 2, 8, 512, 32};
    std::vector<int64_t> strideValues = {540672, 131072, 16384, 32, 1};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4294967296}, {4294967296}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{1310720}, {1310720}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// storage元素数等于2^32-1边界, 校验不回绕
TEST_F(AsStridedTiling, as_strided_tiling_test_storage_size_equal_2p32_minus_1)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {5, 2, 8, 512, 32};
    std::vector<int64_t> strideValues = {540672, 131072, 16384, 32, 1};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4294967295}, {4294967295}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{1310720}, {1310720}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 视图跨度恰好等于storage元素数, 应判定越界失败
TEST_F(AsStridedTiling, as_strided_tiling_test_out_of_range_failed)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {5, 2, 8, 512, 32};
    std::vector<int64_t> strideValues = {540672, 131072, 16384, 32, 1};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{2424831}, {2424831}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{1310720}, {1310720}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// stride超过int32上限(2147483647)且字节步长超过MTE 32位限制, 不应走MoveAlign(防截断), 小输出走SIMT
TEST_F(AsStridedTiling, as_strided_tiling_test_stride_over_int32_movealign_fallback)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {2, 64};
    std::vector<int64_t> strideValues = {3000000000, 1};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{3000000064}, {3000000064}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 单轴size超过uint32上限且stride为0(广播), 输出numel超过2^32, tiling全链路int64不回绕,
// gather因UB因子不满足退出后正常回落NDDMA主模板(fp16: 2)
TEST_F(AsStridedTiling, as_strided_tiling_test_broadcast_size_over_uint32_nddma)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {4294967296, 2};
    std::vector<int64_t> strideValues = {0, 1};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{8589934592}, {8589934592}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 输出numel超过2^32且storage较大, 走NDDMA主模板(axisOutTotalFactor/curAxisFactor需int64)
TEST_F(AsStridedTiling, as_strided_tiling_test_output_numel_over_uint32_nddma)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> sizeValues = {4294967296, 2};
    std::vector<int64_t> strideValues = {1, 0};
    std::vector<int64_t> offsetValues = {0};
    gert::TilingContextPara tilingContextPara("AsStrided",
                                              {{{{4294967296}, {4294967296}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValues.data()},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValues.data()},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValues.data()}},
                                              {
                                                  {
                                                      {{{8589934592}, {8589934592}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  },
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
