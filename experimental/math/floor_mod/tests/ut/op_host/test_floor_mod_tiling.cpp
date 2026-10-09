/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_kernel/floor_mod_tiling_data.h"
#include "../../../op_kernel/floor_mod_tiling_key.h"
#include "floor_mod_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace FloorModNs;
using namespace optiling;

namespace {

const FloorModTilingData* ExecuteFloorModTiling(gert::TilingContextPara& para, TilingInfo& info)
{
    EXPECT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingDataSize, sizeof(FloorModTilingData));
    return reinterpret_cast<const FloorModTilingData*>(info.tilingData.get());
}

} // namespace

TEST(FloorModTilingTest, DenseSameShape)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    const uint64_t key = GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16,
                                            FLOOR_MOD_TPL_PATH_GENERAL);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, key, std::vector<size_t>{0});
    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_DENSE);
    EXPECT_EQ(tiling->dtypeKey, FLOOR_MOD_TPL_FP16);
    EXPECT_EQ(tiling->totalElements, 8192U);
    EXPECT_GT(tiling->denseTile, 0U);
    EXPECT_LE(tiling->denseTile, tiling->totalElements);
    EXPECT_GT(tiling->coreNum, 0U);
    EXPECT_LE(tiling->coreNum, 40U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
    EXPECT_EQ(tiling->outShape[0], 128U);
    EXPECT_EQ(tiling->outShape[1], 64U);
    EXPECT_EQ(tiling->x1Stride[0], 64U);
    EXPECT_EQ(tiling->x2Stride[0], 64U);
}

TEST(FloorModTilingTest, AcceptsFlattenedTorchOutputStorage)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{10, 127, 113}, {10, 127, 113}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{10, 1, 113}, {10, 1, 113}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{143510}, {143510}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalElements, 143510U);
    EXPECT_EQ(tiling->outShape[0], 10U);
    EXPECT_EQ(tiling->outShape[1], 127U);
    EXPECT_EQ(tiling->outShape[2], 113U);
}

TEST(FloorModTilingTest, UsesOriginShapeForFlattenedDoubleInputs)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{1, 25, 89, 113}, {251425}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                  {{{9, 25, 89, 113}, {2262825}}, ge::DT_DOUBLE, ge::FORMAT_ND}},
                                 {{{{9, 25, 89, 113}, {2262825}}, ge::DT_DOUBLE, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalElements, 2262825U);
    EXPECT_EQ(tiling->x1Elements, 251425U);
    EXPECT_EQ(tiling->x2Elements, 2262825U);
    EXPECT_EQ(tiling->outShape[0], 9U);
    EXPECT_EQ(tiling->outShape[1], 25U);
    EXPECT_EQ(tiling->outShape[2], 89U);
    EXPECT_EQ(tiling->outShape[3], 113U);
}

TEST(FloorModTilingTest, RowReuse)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{2, 1, 5}, {2, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_ROW_REUSE);
    EXPECT_EQ(tiling->seedIsX1, 1U);
    EXPECT_EQ(tiling->e1, 4U);
    EXPECT_EQ(tiling->D, 5U);
    EXPECT_EQ(tiling->e2, 1U);
    EXPECT_EQ(tiling->posTotal, 2U);
    EXPECT_EQ(tiling->posDigits, 1U);
    EXPECT_EQ(tiling->posExtent[0], 2U);
    EXPECT_EQ(tiling->seedStride[0], 5U);
    EXPECT_EQ(tiling->denseStride[0], 20U);
    EXPECT_GE(tiling->dTile, 5U);
    EXPECT_GT(tiling->batchRows, 0U);
    EXPECT_GT(tiling->totalTasks, 0U);
}

TEST(FloorModTilingTest, SmallGlobalScalarUsesScalarBroadcastPath)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
                                 {{{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->seedIsX1, 0U);
    EXPECT_EQ(tiling->coreNum, 1U);
    EXPECT_GE(tiling->maxTRowElems, tiling->totalElements);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT32,
                                                 FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST));
}

TEST(FloorModTilingTest, SmallSwappedGlobalScalarUsesScalarBroadcastPath)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod", {{{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}, {{{83, 7}, {83, 7}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{83, 7}, {83, 7}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->seedIsX1, 1U);
    EXPECT_EQ(tiling->swapped, 1U);
    EXPECT_EQ(tiling->coreNum, 1U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_BF16,
                                                 FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST));
}

TEST(FloorModTilingTest, MediumGlobalScalarUsesScalarBroadcastPath)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod", {{{{2049}, {2049}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2049}, {2049}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->coreNum, 0U);
    EXPECT_LT(tiling->coreNum, 40U);
    EXPECT_EQ(tiling->totalTasks, tiling->coreNum);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16,
                                                 FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST));
}

TEST(FloorModTilingTest, OneTilePerCoreGlobalScalarUsesScalarBroadcastPath)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{21, 53, 107}, {21, 53, 107}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{21, 53, 107}, {21, 53, 107}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->coreNum, 1U);
    EXPECT_EQ(tiling->totalTasks, tiling->coreNum);
    EXPECT_EQ(tiling->e2TileCount, tiling->coreNum);
    EXPECT_GE(tiling->maxTRowElems, tiling->e2Tile);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32,
                                                 FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST));
}

TEST(FloorModTilingTest, SubAmortizedGlobalScalarUsesBoundedAnalyticalCoreCount)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{3, 108, 117}, {3, 108, 117}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{3, 108, 117}, {3, 108, 117}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->coreNum, 0U);
    EXPECT_LE(tiling->coreNum, 40U);
    EXPECT_EQ(tiling->totalTasks, tiling->coreNum);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32,
                                                 FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST));
}

TEST(FloorModTilingTest, MultiWaveGlobalScalarUsesBalancedScalarPipeline)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod", {{{{1000000}, {1000000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1000000}, {1000000}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->totalTasks, tiling->coreNum);
    EXPECT_EQ(tiling->coreNum, 40U);
    EXPECT_EQ(tiling->totalTasks % tiling->coreNum, 0U);
    EXPECT_EQ(tiling->e2TileCount, tiling->totalTasks);
    EXPECT_LE(tiling->e2Tile, 255U * 64U + 63U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32,
                                                 FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST));
}

TEST(FloorModTilingTest, ContiguousPositionsUseBatchedRepeat)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{18, 19, 1, 17}, {18, 19, 1, 17}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{18, 19, 66, 17}, {18, 19, 66, 17}}, ge::DT_INT32, ge::FORMAT_ND}},
                                 {{{{18, 19, 66, 17}, {18, 19, 66, 17}}, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_ROW_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_COMPACT_POS_BROADCAST);
    EXPECT_EQ(tiling->posTotal, 18U * 19U);
    EXPECT_GT(tiling->posTile, 1U);
    EXPECT_GE(tiling->totalTasks, 40U);
    EXPECT_LE(tiling->totalTasks, 80U);
    EXPECT_EQ(tiling->coreNum, 40U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, MediumRowReuseUsesHardwareParallelism)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{1, 1, 22}, {1, 1, 22}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{16, 111, 22}, {16, 111, 22}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{16, 111, 22}, {16, 111, 22}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_ROW_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_COMPACT_ROW_BATCH);
    EXPECT_EQ(tiling->coreNum, 20U);
    EXPECT_EQ(tiling->rowPartitions, 20U);
    EXPECT_GE(tiling->batchRows, 89U);
    EXPECT_EQ(tiling->maxPatternElems, 22U);
    EXPECT_GE(tiling->maxTRowElems, tiling->batchRows * 22U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_BF16,
                                                 FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH));
}

TEST(FloorModTilingTest, Int64ShortRowsUseBoundedAnalyticalPartitions)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{7, 65, 4}, {7, 65, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                  {{{1, 1, 4}, {1, 1, 4}}, ge::DT_INT64, ge::FORMAT_ND}},
                                 {{{{7, 65, 4}, {7, 65, 4}}, ge::DT_INT64, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_ROW_REUSE);
    EXPECT_EQ(tiling->D, 4U);
    EXPECT_EQ(tiling->e1, 455U);
    EXPECT_GT(tiling->rowPartitions, 1U);
    EXPECT_LE(tiling->rowPartitions, 40U);
    EXPECT_EQ(tiling->coreNum, tiling->rowPartitions);
    EXPECT_GT(tiling->batchRows, 0U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, CompactRowBatchGeneralizesAcrossWidths)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{1, 37}, {1, 37}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{129, 37}, {129, 37}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{129, 37}, {129, 37}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_ROW_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_COMPACT_ROW_BATCH);
    EXPECT_EQ(tiling->D, 37U);
    EXPECT_EQ(tiling->e1, 129U);
    EXPECT_GT(tiling->batchRows, 1U);
    EXPECT_GE(tiling->maxTRowElems, tiling->batchRows * 37U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16,
                                                 FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH));
}

TEST(FloorModTilingTest, CompactRowBatchUsesPlatformUbBudget)
{
    FloorModCompileInfo compileInfo = {40, 96 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{1, 96}, {1, 96}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{73, 96}, {73, 96}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{73, 96}, {73, 96}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_COMPACT_ROW_BATCH);
    EXPECT_EQ(tiling->D, 96U);
    EXPECT_GT(tiling->rowPartitions, 1U);
    EXPECT_LE(tiling->ubUsedBytes, 95U * 1024U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32,
                                                 FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH));
}

TEST(FloorModTilingTest, SingleRowReuseUsesWideDSlab)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{5, 1, 26, 26, 7, 93}, {5, 1, 26, 26, 7, 93}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{5, 7, 26, 26, 7, 93}, {5, 7, 26, 26, 7, 93}}, ge::DT_INT32, ge::FORMAT_ND}},
                                 {{{{5, 7, 26, 26, 7, 93}, {5, 7, 26, 26, 7, 93}}, ge::DT_INT32, ge::FORMAT_ND}},
                                 &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_ROW_REUSE);
    EXPECT_GT(tiling->dTile, 2040U);
    EXPECT_EQ(tiling->batchRows, 1U);
    EXPECT_EQ(tiling->reuseSchedule, 0U);
    EXPECT_EQ(tiling->coreNum, 40U);
}

TEST(FloorModTilingTest, ExpandReuse)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{2, 4, 1}, {2, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->seedIsX1, 1U);
    EXPECT_EQ(tiling->e1, 1U);
    EXPECT_EQ(tiling->D, 8U);
    EXPECT_EQ(tiling->e2, 5U);
    EXPECT_GT(tiling->dTile, 0U);
    EXPECT_GT(tiling->e2Tile, 0U);
    EXPECT_GT(tiling->batchRows, 0U);
    EXPECT_TRUE(tiling->reuseLayout == FLOOR_MOD_REUSE_PADDED_REPEAT ||
                tiling->reuseLayout == FLOOR_MOD_REUSE_PACKED_BROADCAST ||
                tiling->reuseLayout == FLOOR_MOD_REUSE_DENSE_TAIL_BATCH);
}

TEST(FloorModTilingTest, PackedBroadcastLargeShape)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{17, 17, 23, 1, 19, 1}, {17, 17, 23, 1, 19, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{17, 17, 23, 28, 19, 109}, {17, 17, 23, 28, 19, 109}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{17, 17, 23, 28, 19, 109}, {17, 17, 23, 28, 19, 109}}, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PACKED_BROADCAST);
    EXPECT_EQ(tiling->D, 19U);
    EXPECT_EQ(tiling->e2, 109U);
    EXPECT_EQ(tiling->dTile, 19U);
    EXPECT_EQ(tiling->e2Tile, 109U);
    EXPECT_EQ(tiling->coreNum, 40U);
    EXPECT_GT(tiling->broadcastTmpBytes, 0U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, DenseTailBatchTilesDWhenFullSlabExceedsUb)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{9, 22, 83, 1}, {9, 22, 83, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{9, 22, 83, 49}, {9, 22, 83, 49}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{{{9, 22, 83, 49}, {9, 22, 83, 49}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_DENSE_TAIL_BATCH);
    EXPECT_LT(tiling->dTile, tiling->D);
    EXPECT_EQ(tiling->e2Tile, tiling->e2);
    EXPECT_GT(tiling->broadcastTmpBytes, 0U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, WideDenseTailUsesUbBoundedAnalyticalRoute)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{7717, 1}, {7717, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{7717, 2002}, {7717, 2002}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{7717, 2002}, {7717, 2002}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PADDED_REPEAT);
    EXPECT_EQ(tiling->D, 7717U);
    EXPECT_EQ(tiling->e2, 2002U);
    EXPECT_GT(tiling->coreNum, 0U);
    EXPECT_LE(tiling->coreNum, 40U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_FP32,
                                                 FLOOR_MOD_TPL_PATH_GENERAL));
}

TEST(FloorModTilingTest, DenseTailBatchOwnsContiguousRows)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    const std::vector<gert::Shape> shapes = {{5, 11, 19, 3, 69, 23}, {216315, 23}, {65536, 16}};
    for (const auto& denseShape : shapes) {
        auto seedShape = denseShape;
        const auto rank = denseShape.GetDimNum();
        const auto width = denseShape.GetDim(rank - 1);
        seedShape.SetDim(rank - 1, 1);
        gert::StorageShape denseStorage;
        denseStorage.MutableOriginShape() = denseShape;
        denseStorage.MutableStorageShape() = denseShape;
        gert::StorageShape seedStorage;
        seedStorage.MutableOriginShape() = seedShape;
        seedStorage.MutableStorageShape() = seedShape;
        SCOPED_TRACE(denseShape.GetShapeSize());
        gert::TilingContextPara para(
            "FloorMod", {{denseStorage, ge::DT_INT32, ge::FORMAT_ND}, {seedStorage, ge::DT_INT32, ge::FORMAT_ND}},
            {{denseStorage, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);
        TilingInfo info;
        const auto* tiling = ExecuteFloorModTiling(para, info);
        ASSERT_NE(tiling, nullptr);
        EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
        EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_DENSE_TAIL_BATCH);
        EXPECT_EQ(tiling->D, static_cast<uint64_t>(seedShape.GetShapeSize()));
        EXPECT_EQ(tiling->e2, static_cast<uint32_t>(width));
        EXPECT_GT(tiling->dTile, 1U);
        EXPECT_EQ(tiling->e2Tile, static_cast<uint32_t>(width));
        EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
        EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT32,
                                                     FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH));
    }
}

TEST(FloorModTilingTest, SmallDenseTailUsesOneLargeSingleBufferedSlab)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{7, 76, 32}, {7, 76, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{7, 76, 1}, {7, 76, 1}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{7, 76, 32}, {7, 76, 32}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_DENSE_TAIL_BATCH);
    EXPECT_EQ(tiling->reuseSchedule, 0U);
    EXPECT_EQ(tiling->coreNum, 3U);
    EXPECT_GE(tiling->dTile, 178U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, SmallDenseTailAmortizesWorkAcrossFewerCores)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{66, 1}, {66, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{66, 71}, {66, 71}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{66, 71}, {66, 71}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PACKED_BROADCAST);
    EXPECT_GT(tiling->dTile, 1U);
    EXPECT_LT(tiling->coreNum, 40U);
    EXPECT_EQ(tiling->e2Tile, 71U);
}

TEST(FloorModTilingTest, UnderfilledWideDenseTailUsesBoundedAnalyticalParallelism)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{25, 7, 1}, {25, 7, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{25, 7, 87}, {25, 7, 87}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{25, 7, 87}, {25, 7, 87}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PACKED_BROADCAST);
    EXPECT_EQ(tiling->D, 175U);
    EXPECT_EQ(tiling->e2, 87U);
    EXPECT_GT(tiling->coreNum, 0U);
    EXPECT_LE(tiling->coreNum, 40U);
    EXPECT_GE(tiling->totalTasks, tiling->coreNum);
    EXPECT_EQ(tiling->reuseSchedule, 0U);
    EXPECT_GT(tiling->broadcastTmpBytes, 0U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, PackedBroadcastTilesE2ForScalarSeed)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{3750, 228}, {3750, 228}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{{{3750, 228}, {3750, 228}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PACKED_BROADCAST);
    EXPECT_EQ(tiling->dTile, 1U);
    EXPECT_LT(tiling->e2Tile, tiling->e2);
    EXPECT_EQ(tiling->broadcastTmpBytes, 0U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, CrossedRouteWinsUnifiedCostForAlternatingLargeShape)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{17, 6, 23, 4, 1, 26}, {17, 6, 23, 4, 1, 26}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{17, 6, 23, 1, 90, 26}, {17, 6, 23, 1, 90, 26}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{17, 6, 23, 4, 90, 26}, {17, 6, 23, 4, 90, 26}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_CROSSED);
    EXPECT_EQ(tiling->crossOuter, 2346U);
    EXPECT_EQ(tiling->crossA, 4U);
    EXPECT_EQ(tiling->crossB, 90U);
    EXPECT_EQ(tiling->crossD, 26U);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PADDED_COMPACT);
    EXPECT_GT(tiling->crossDAligned, tiling->crossD);
    EXPECT_EQ(tiling->crossBTile, tiling->crossB);
    EXPECT_EQ(tiling->crossBTileCount, 1U);
    EXPECT_GT(tiling->crossTotalTasks, 0U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, LargeUnitDCrossKeepsEveryPartialSeedSlabAligned)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{66, 8754, 1}, {66, 8754, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{66, 1, 592}, {66, 1, 592}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{66, 8754, 592}, {66, 8754, 592}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_CROSSED);
    EXPECT_EQ(tiling->crossD, 1U);
    if (tiling->crossOuterTile > 1U && tiling->crossATile != tiling->crossA) {
        const uint32_t tailA = tiling->crossA % tiling->crossATile == 0U ? tiling->crossATile :
                                                                           tiling->crossA % tiling->crossATile;
        EXPECT_EQ((tiling->crossATile * tiling->crossM * sizeof(float)) % 32U, 0U);
        EXPECT_EQ((tailA * tiling->crossM * sizeof(float)) % 32U, 0U);
    }
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, UnitDCrossedRouteWinsAfterFillingHardwareWave)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{2, 141, 1}, {2, 141, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{2, 1, 3899}, {2, 1, 3899}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{2, 141, 3899}, {2, 141, 3899}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_CROSSED);
    EXPECT_EQ(tiling->crossM, 1U);
    EXPECT_EQ(tiling->crossD, 1U);
    EXPECT_GE(tiling->crossTotalTasks, 40U);
}

TEST(FloorModTilingTest, Bf16CrossedShortRowsUseCompactOutput)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{9, 29, 2, 3, 19, 1, 30}, {9, 29, 2, 3, 19, 1, 30}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{9, 29, 2, 3, 1, 43, 30}, {9, 29, 2, 3, 1, 43, 30}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{9, 29, 2, 3, 19, 43, 30}, {9, 29, 2, 3, 19, 43, 30}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_CROSSED);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PADDED_COMPACT);
    EXPECT_EQ(tiling->crossOuter, 1566U);
    EXPECT_EQ(tiling->crossA, 19U);
    EXPECT_EQ(tiling->crossM, 1U);
    EXPECT_EQ(tiling->crossB, 43U);
    EXPECT_EQ(tiling->crossD, 30U);
    EXPECT_EQ(tiling->crossBTile, tiling->crossB);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, UnitDCrossedMatricesUseCompactOutput)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{28, 21, 27, 1, 22}, {28, 21, 27, 1, 22}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{28, 21, 27, 58, 1}, {28, 21, 27, 58, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{28, 21, 27, 58, 22}, {28, 21, 27, 58, 22}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_CROSSED);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PADDED_COMPACT);
    EXPECT_EQ(tiling->crossOuter, 15876U);
    EXPECT_EQ(tiling->crossA, 58U);
    EXPECT_EQ(tiling->crossM, 1U);
    EXPECT_EQ(tiling->crossB, 22U);
    EXPECT_EQ(tiling->crossD, 1U);
    EXPECT_EQ(tiling->crossATile, tiling->crossA);
    EXPECT_EQ(tiling->crossBTile, tiling->crossB);
    EXPECT_EQ(tiling->arithmeticMode, FLOOR_MOD_ARITH_FMA_NEG_DENOMINATOR);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, CommonMSpanKeepsPackedReuseRoute)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{27, 3, 69, 1}, {27, 3, 69, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                  {{{1, 3, 69, 108}, {1, 3, 69, 108}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{{{27, 3, 69, 108}, {27, 3, 69, 108}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PACKED_BROADCAST);
}

TEST(FloorModTilingTest, UnderfilledUnitDUsesAnalyticalPackedReuseRoute)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod",
        {{{{23, 1}, {23, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{1, 61}, {1, 61}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{23, 61}, {23, 61}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->reuseLayout, FLOOR_MOD_REUSE_PACKED_BROADCAST);
    EXPECT_EQ(tiling->totalElements, 23U * 61U);
    EXPECT_EQ(tiling->D, 1U);
    EXPECT_EQ(tiling->e2, 61U);
    EXPECT_GT(tiling->coreNum, 0U);
    EXPECT_LE(tiling->coreNum, 40U);
    EXPECT_LE(tiling->ubUsedBytes, 191U * 1024U);
}

TEST(FloorModTilingTest, ScalarBroadcast)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod", {{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->seedIsX1, 1U);
    EXPECT_EQ(tiling->totalElements, 6U);
    EXPECT_EQ(tiling->D, 1U);
    EXPECT_EQ(tiling->e2, 6U);
    EXPECT_EQ(tiling->coreNum, 1U);
}

TEST(FloorModTilingTest, RankZeroScalar)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_DENSE);
    EXPECT_EQ(tiling->rank, 0U);
    EXPECT_EQ(tiling->totalElements, 1U);
    EXPECT_EQ(tiling->coreNum, 1U);
}

TEST(FloorModTilingTest, AlternatingBroadcastUsesReusableSuffix)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para("FloorMod",
                                 {{{{2, 1, 4, 1, 8, 1}, {2, 1, 4, 1, 8, 1}}, ge::DT_INT64, ge::FORMAT_ND},
                                  {{{1, 3, 1, 5, 1, 17}, {1, 3, 1, 5, 1, 17}}, ge::DT_INT64, ge::FORMAT_ND}},
                                 {{{{2, 3, 4, 5, 8, 17}, {2, 3, 4, 5, 8, 17}}, ge::DT_INT64, ge::FORMAT_ND}},
                                 &compileInfo);

    TilingInfo info;
    const auto* tiling = ExecuteFloorModTiling(para, info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->mode, FLOOR_MOD_MODE_EXPAND_REUSE);
    EXPECT_EQ(tiling->seedIsX1, 1U);
    EXPECT_EQ(tiling->swapped, 1U);
    EXPECT_EQ(tiling->D, 1U);
    EXPECT_EQ(tiling->e2, 17U);
    EXPECT_EQ(tiling->posTotal, 960U);
}

TEST(FloorModTilingTest, RejectsInvalidBroadcast)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    gert::TilingContextPara para(
        "FloorMod", {{{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(FloorModTilingTest, RejectsElementCountOverflow)
{
    FloorModCompileInfo compileInfo = {40, 192 * 1024, false};
    const int64_t hugeDim = std::numeric_limits<int64_t>::max();
    gert::TilingContextPara para(
        "FloorMod",
        {{{{hugeDim, 2}, {hugeDim, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{hugeDim, 2}, {hugeDim, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
