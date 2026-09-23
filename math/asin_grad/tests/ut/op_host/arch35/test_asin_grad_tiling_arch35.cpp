/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/asin_grad_tiling_data.h"

namespace optiling {
struct AsinGradCompileInfo {};
} // namespace optiling

namespace {
optiling::AsinGradCompileInfo g_compileInfo;

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    auto& originShape = storageShape.MutableOriginShape();
    auto& runtimeShape = storageShape.MutableStorageShape();
    originShape.SetDimNum(dimensions.size());
    runtimeShape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        originShape.SetDim(i, dimensions[i]);
        runtimeShape.SetDim(i, dimensions[i]);
    }
    return storageShape;
}

// 输入顺序与 op_def / proto.h 一致：y, dy；输出 z
static gert::TilingContextPara MakeTilingContext(const std::vector<int64_t>& shape, ge::DataType yDtype,
                                                 ge::DataType dyDtype)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {MakeStorageShape(shape), yDtype, ge::FORMAT_ND},
        {MakeStorageShape(shape), dyDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {MakeStorageShape(shape), yDtype, ge::FORMAT_ND},
    };
    return gert::TilingContextPara("AsinGrad", inputs, outputs, {}, &g_compileInfo);
}
} // namespace

class AsinGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AsinGradTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AsinGradTilingTest TearDown" << std::endl; }
};

TEST_F(AsinGradTilingTest, asin_grad_tiling_fp32_basic)
{
    auto context = MakeTilingContext({8, 128}, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(context, tilingInfo));
}

TEST_F(AsinGradTilingTest, asin_grad_tiling_fp16_basic)
{
    auto context = MakeTilingContext({4, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(context, tilingInfo));
}

TEST_F(AsinGradTilingTest, asin_grad_tiling_bf16_basic)
{
    auto context = MakeTilingContext({1024}, ge::DT_BF16, ge::DT_BF16);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(context, tilingInfo));
}

TEST_F(AsinGradTilingTest, asin_grad_tiling_empty_tensor)
{
    auto context = MakeTilingContext({0}, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(context, tilingInfo));
}

TEST_F(AsinGradTilingTest, asin_grad_tiling_rejects_unsupported_dtype)
{
    auto context = MakeTilingContext({8, 128}, ge::DT_DOUBLE, ge::DT_DOUBLE);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

// The kernel reads y/dy/z with a single StorageT derived from y, so a dy
// dtype different from y must be rejected instead of silently misreading
// the dy buffer.
TEST_F(AsinGradTilingTest, asin_grad_tiling_rejects_dy_dtype_mismatch)
{
    auto context = MakeTilingContext({8, 128}, ge::DT_FLOAT, ge::DT_FLOAT16);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST_F(AsinGradTilingTest, asin_grad_tiling_rejects_shape_mismatch)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {MakeStorageShape({8, 128}), ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeStorageShape({8, 64}), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {MakeStorageShape({8, 128}), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    gert::TilingContextPara context("AsinGrad", inputs, outputs, {}, &g_compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}
