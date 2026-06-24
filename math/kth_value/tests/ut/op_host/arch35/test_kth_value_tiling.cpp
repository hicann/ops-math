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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/kth_value_tiling_data.h"

using namespace std;
using namespace ge;

class KthValueTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

struct KthValueCompileInfo {
    int32_t coreNum = 64;
};

namespace {
constexpr size_t WORK_SPACE_SIZE = 16777216;
KthValueCompileInfo g_compileInfo = {64};

gert::TilingContextPara MakeKthValueTilingContext(
    const gert::StorageShape& xShape, const gert::StorageShape& valuesShape, const gert::StorageShape& indicesShape,
    ge::DataType xDtype, int64_t k, int64_t dim = -1)
{
    return gert::TilingContextPara(
        "KthValue",
        {
            {xShape, xDtype, ge::FORMAT_ND},
        },
        {
            {valuesShape, xDtype, ge::FORMAT_ND},
            {indicesShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("k", Ops::Math::AnyValue::CreateFrom<int64_t>(k)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)),
        },
        &g_compileInfo);
}
} // namespace

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_fp32_2x1024)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{2, 1024}, {2, 1024}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 5);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_axis_one_copy_fp32_100x1)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{100, 1}, {100, 1}}, {{100, 1}, {100, 1}}, {{100, 1}, {100, 1}}, ge::DT_FLOAT, 1, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_bf16_8x512)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{8, 512}, {8, 512}}, {{8, 1}, {8, 1}}, {{8, 1}, {8, 1}}, ge::DT_BF16, 100);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_fp32_3d_8x16x1024)
{
    auto tilingContextPara = MakeKthValueTilingContext(
        {{8, 16, 1024}, {8, 16, 1024}}, {{8, 16, 1}, {8, 16, 1}}, {{8, 16, 1}, {8, 16, 1}}, ge::DT_FLOAT, 500, 2);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_axis_one_copy_int32_64x1)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{64, 1}, {64, 1}}, {{64, 1}, {64, 1}}, {{64, 1}, {64, 1}}, ge::DT_INT32, 1, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_axis_one_copy_allows_unsorted_dim_over_uint32)
{
    int64_t largeBatch = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    auto tilingContextPara = MakeKthValueTilingContext(
        {{largeBatch, 1}, {largeBatch, 1}}, {{largeBatch, 1}, {largeBatch, 1}}, {{largeBatch, 1}, {largeBatch, 1}},
        ge::DT_FLOAT, 1, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->unsortedDimNum, largeBatch);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_intra_core_allows_unsorted_dim_over_uint32)
{
    int64_t largeBatch = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    auto tilingContextPara = MakeKthValueTilingContext(
        {{largeBatch, 4097}, {largeBatch, 4097}}, {{largeBatch, 1}, {largeBatch, 1}},
        {{largeBatch, 1}, {largeBatch, 1}}, ge::DT_FLOAT, 64);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 260);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->unsortedDimNum, largeBatch);
    EXPECT_GT(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_fp32_1d_512)
{
    auto tilingContextPara = MakeKthValueTilingContext({{512}, {512}}, {{1}, {1}}, {{1}, {1}}, ge::DT_FLOAT, 256);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_bf16_nonlast_axis1840_radix)
{
    auto tilingContextPara = MakeKthValueTilingContext(
        {{2, 1840, 1}, {2, 1840, 1}}, {{2, 1, 1}, {2, 1, 1}}, {{2, 1, 1}, {2, 1, 1}}, ge::DT_BF16, 920, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 266);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_invalid_dtype)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}}, ge::DT_BOOL, 5);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_empty_shape)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{0, 100}, {0, 100}}, {{0, 1}, {0, 1}}, {{0, 1}, {0, 1}}, ge::DT_FLOAT, 5);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_k_zero)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 0);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_k_exceeds_axis)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{2, 10}, {2, 10}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 11);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_dim_out_of_range)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 5, 3);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_negative_dim_out_of_range)
{
    auto tilingContextPara =
        MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 5, -3);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_values_dtype_mismatch)
{
    gert::TilingContextPara tilingContextPara(
        "KthValue",
        {
            gert::TilingContextPara::TensorDescription({{2, 100}, {2, 100}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND),
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_INT64, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::OpAttr("k", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
        },
        &g_compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_indices_dtype_not_int64)
{
    gert::TilingContextPara tilingContextPara(
        "KthValue",
        {
            gert::TilingContextPara::TensorDescription({{2, 100}, {2, 100}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND),
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::OpAttr("k", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
        },
        &g_compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
