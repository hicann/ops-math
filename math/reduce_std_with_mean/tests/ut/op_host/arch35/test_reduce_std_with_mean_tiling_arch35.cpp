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

struct ReduceStdWithMeanCompileInfo {};

class ReduceStdWithMeanTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdWithMeanTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceStdWithMeanTiling TearDown" << std::endl; }
};

// OpDef 属性顺序：dim(0) / unbiased(1) / keepdim(2) / invert(3) / epsilon(4) / correction(5)
static std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(const std::vector<int64_t>& dim, bool unbiased,
                                                              bool keepdim, bool invert, float epsilon,
                                                              int64_t correction)
{
    return {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(dim)),
            gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(unbiased)),
            gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepdim)),
            gert::TilingContextPara::OpAttr("invert", Ops::Math::AnyValue::CreateFrom<bool>(invert)),
            gert::TilingContextPara::OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
            gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(correction))};
}

static gert::StorageShape ToStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
    return shape;
}

static gert::TilingContextPara MakeCase(const std::vector<int64_t>& xShape, ge::DataType dtype,
                                        const std::vector<int64_t>& dim, bool unbiased = true, bool keepdim = false,
                                        bool invert = false, float epsilon = 0.001f, int64_t correction = 1,
                                        ge::DataType meanDtype = ge::DT_UNDEFINED,
                                        const std::vector<int64_t>& meanShape = {})
{
    static ReduceStdWithMeanCompileInfo compileInfo;
    if (meanDtype == ge::DT_UNDEFINED) {
        meanDtype = dtype;
    }
    const std::vector<int64_t>& mShape = meanShape.empty() ? xShape : meanShape;
    return gert::TilingContextPara(
        "ReduceStdWithMean",
        {gert::TilingContextPara::TensorDescription(ToStorageShape(xShape), dtype, ge::FORMAT_ND),
         gert::TilingContextPara::TensorDescription(ToStorageShape(mShape), meanDtype, ge::FORMAT_ND)},
        {gert::TilingContextPara::TensorDescription(ToStorageShape({1}), dtype, ge::FORMAT_ND)},
        MakeAttrs(dim, unbiased, keepdim, invert, epsilon, correction), &compileInfo);
}

// TilingKey 位布局（templateType=bit0, isEmptyTensor=bit1, isTailR=bit2）：
//   normal tailA=0 / group tailA=1 / empty=2 / normal tailR=4 / group tailR=5
static constexpr uint64_t kKeyNormalTailA = 0;
static constexpr uint64_t kKeyGroupTailA = 1;
static constexpr uint64_t kKeyEmpty = 2;
static constexpr uint64_t kKeyNormalTailR = 4;
static constexpr uint64_t kKeyGroupTailR = 5;
// 纯 Vector kernel 无系统 workspace（lib api 16MB 已去除），normal/empty 路径 ws=0

TEST_F(ReduceStdWithMeanTiling, tiling_fp32_2d_tailR)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {1}), ge::GRAPH_SUCCESS, kKeyNormalTailR, std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_fp32_2d_tailA)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {0}), ge::GRAPH_SUCCESS, kKeyNormalTailA, std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_fp16_all_reduce)
{
    ExecuteTestCase(MakeCase({128, 64}, ge::DT_FLOAT16, {}), ge::GRAPH_SUCCESS, kKeyNormalTailR,
                    std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_bf16_negative_dim)
{
    ExecuteTestCase(MakeCase({8, 128, 128}, ge::DT_BF16, {-1}), ge::GRAPH_SUCCESS, kKeyNormalTailR,
                    std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_scalar_input)
{
    ExecuteTestCase(MakeCase({}, ge::DT_FLOAT, {}), ge::GRAPH_SUCCESS, kKeyNormalTailR, std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_invert_correction_zero)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {1}, false, false, true, 0.01f, 0), ge::GRAPH_SUCCESS,
                    kKeyNormalTailR, std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_empty_r)
{
    ExecuteTestCase(MakeCase({4, 0}, ge::DT_FLOAT, {1}), ge::GRAPH_SUCCESS, kKeyEmpty, std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_empty_a)
{
    ExecuteTestCase(MakeCase({0, 4}, ge::DT_FLOAT, {1}), ge::GRAPH_SUCCESS, kKeyEmpty, std::vector<size_t>{0});
}

TEST_F(ReduceStdWithMeanTiling, tiling_group_large_reduce)
{
    // A=2（不满核）且 R=131072（fp32 512KB，UB 装不下 → rLoopCnt>1）触发 group 模板（A×R 2D 分核）
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeCase({2, 131072}, ge::DT_FLOAT, {1}), tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, kKeyGroupTailR);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_GT(tilingInfo.workspaceSizes[0], 0U);
    EXPECT_GE(tilingInfo.blockNum, 2U);
}

TEST_F(ReduceStdWithMeanTiling, tiling_unsupported_dtype_rejected)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_INT32, {1}, true, false, false, 0.001f, 1, ge::DT_INT32), ge::GRAPH_FAILED);
}

TEST_F(ReduceStdWithMeanTiling, tiling_mean_dtype_mismatch_rejected)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {1}, true, false, false, 0.001f, 1, ge::DT_FLOAT16),
                    ge::GRAPH_FAILED);
}

TEST_F(ReduceStdWithMeanTiling, tiling_shape_mismatch_rejected)
{
    ExecuteTestCase(MakeCase({2, 6}, ge::DT_FLOAT, {1}, true, false, false, 0.001f, 1, ge::DT_FLOAT, {3, 4}),
                    ge::GRAPH_FAILED);
}

TEST_F(ReduceStdWithMeanTiling, tiling_dim_out_of_range_rejected)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {2}), ge::GRAPH_FAILED);
}

TEST_F(ReduceStdWithMeanTiling, tiling_negative_dim_out_of_range_rejected)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {-3}), ge::GRAPH_FAILED);
}

TEST_F(ReduceStdWithMeanTiling, tiling_duplicate_dim_rejected)
{
    ExecuteTestCase(MakeCase({4, 8}, ge::DT_FLOAT, {1, -1}), ge::GRAPH_FAILED);
}

TEST_F(ReduceStdWithMeanTiling, tiling_rank_9_rejected)
{
    ExecuteTestCase(MakeCase({1, 1, 1, 1, 1, 1, 1, 1, 2}, ge::DT_FLOAT, {8}), ge::GRAPH_FAILED);
}
