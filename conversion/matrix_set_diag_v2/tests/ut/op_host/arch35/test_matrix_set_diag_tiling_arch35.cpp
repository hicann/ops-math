/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_kernel/arch35/matrix_set_diag_v2_tilingdata.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MatrixSetDiagV2TilingTest : public testing::Test {
protected:
    MatrixSetDiagCompileInfo compileInfo{};

    static void SetUpTestCase() { std::cout << "MatrixSetDiagV2TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatrixSetDiagV2TilingTest TearDown" << std::endl; }
};

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_int16)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_uint8)
{
    std::vector<int32_t> kValues = {1};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_float)
{
    std::vector<int32_t> kValues = {-1, 1};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_bool)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {{{2, 2, 2}, {2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0b0'0'1'00000010;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_failed_should_have_same_type)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_failed_diag_dim_should_less_input_dim)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_failed_input_dim_from_2)
{
    std::vector<int32_t> kValues = {0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_align)
{
    std::vector<int32_t> kValues = {-31, 0};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiagV2",
        {
            {{{10000, 2, 10000, 32}, {10000, 2, 10000, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10000, 2, 32, 32}, {10000, 2, 32, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{10000, 2, 10000, 32}, {10000, 2, 10000, 32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalCntPerCore, 12500);
    EXPECT_EQ(data->xRowFactor, 256);
    EXPECT_EQ(data->xColFactor, 32);
    EXPECT_EQ(data->input.coreNum, 64);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_not_align)
{
    std::vector<int32_t> kValues = {-30, 0};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiagV2",
        {
            {{{10000, 2, 10000, 31}, {10000, 2, 10000, 31}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{10000, 2, 31, 31}, {10000, 2, 31, 31}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{10000, 2, 10000, 31}, {10000, 2, 10000, 31}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->totalCntPerCore, 5938);
    EXPECT_EQ(data->xRowFactor, 528);
    EXPECT_EQ(data->xColFactor, 31);
    EXPECT_EQ(data->input.coreNum, 64);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_1)
{
    std::vector<int32_t> kValues = {-8, 23};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{23, 144, 138}, {23, 144, 138}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{23, 32, 138}, {23, 32, 138}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{23, 144, 138}, {23, 144, 138}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_2)
{
    std::vector<int32_t> kValues = {-2, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{657, 3, 8}, {657, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{657, 10, 3}, {657, 10, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{657, 3, 8}, {657, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'1'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_3)
{
    std::vector<int32_t> kValues = {-3, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1507, 4, 8}, {1507, 4, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{1507, 11, 4}, {1507, 11, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1507, 4, 8}, {1507, 4, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'1'00000001);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2NoCutTailTilingData));
    MSDV2NoCutTailTilingData* data = reinterpret_cast<MSDV2NoCutTailTilingData*>(tilingInfo.tilingData.get());
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_4)
{
    std::vector<int32_t> kValues = {7, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{83, 192, 3}, {83, 192, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_5)
{
    std::vector<int32_t> kValues = {7, 7};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{83, 192, 3}, {83, 192, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_6)
{
    std::vector<int32_t> kValues = {4, 4};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1252, 3, 4}, {1252, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{1252, 3}, {1252, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1252, 3, 4}, {1252, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_optimize_core_num1)
{
    std::vector<int32_t> kValues = {-1, 4};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{1, 21, 121, 100}, {1, 21, 121, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 21, 6, 100}, {1, 21, 6, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{1, 21, 121, 100}, {1, 21, 121, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 42);
    EXPECT_EQ(data->xRowFactor, 81);
    EXPECT_EQ(data->xColFactor, 100);
    EXPECT_EQ(data->totalCntPerCore, 1);
}

TEST_F(MatrixSetDiagV2TilingTest, test_tiling_cut_tail_optimize_core_num2)
{
    std::vector<int32_t> kValues = {-20532, 0};
    gert::TilingContextPara tilingContextPara("MatrixSetDiagV2",
                                              {
                                                  {{{65538, 2}, {65538, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{20533, 2}, {20533, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                              },
                                              {
                                                  {{{65538, 2}, {65538, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b1'0'0'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MSDV2CutTailTilingData));
    MSDV2CutTailTilingData* data = reinterpret_cast<MSDV2CutTailTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->input.coreNum, 33);
    EXPECT_EQ(data->xRowFactor, 2048);
    EXPECT_EQ(data->xColFactor, 2);
    EXPECT_EQ(data->totalCntPerCore, 1);
}
