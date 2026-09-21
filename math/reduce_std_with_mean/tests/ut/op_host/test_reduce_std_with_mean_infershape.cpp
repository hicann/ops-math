/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace {

// OpDef 属性顺序：dim(0) / unbiased(1) / keepdim(2) / invert(3) / epsilon(4) / correction(5)
std::vector<gert::InfershapeContextPara::OpAttr> MakeAttrs(const std::vector<int64_t>& dim, bool unbiased, bool keepdim,
                                                           bool invert, float epsilon, int64_t correction)
{
    return {gert::InfershapeContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(dim)),
            gert::InfershapeContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(unbiased)),
            gert::InfershapeContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepdim)),
            gert::InfershapeContextPara::OpAttr("invert", Ops::Math::AnyValue::CreateFrom<bool>(invert)),
            gert::InfershapeContextPara::OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
            gert::InfershapeContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(correction))};
}

gert::StorageShape ToStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
    return shape;
}

// 输入 x 与 mean 同形同 dtype（契约：mean 已广播到 x 形状）；输出 y 形状由 infershape 推导
gert::InfershapeContextPara MakePara(const std::vector<int64_t>& xShape, ge::DataType dtype,
                                     const std::vector<int64_t>& dim, bool unbiased = true, bool keepdim = false,
                                     bool invert = false, float epsilon = 0.001f, int64_t correction = 1)
{
    return gert::InfershapeContextPara(
        "ReduceStdWithMean",
        {gert::InfershapeContextPara::TensorDescription(ToStorageShape(xShape), dtype, ge::FORMAT_ND),
         gert::InfershapeContextPara::TensorDescription(ToStorageShape(xShape), dtype, ge::FORMAT_ND)},
        {gert::InfershapeContextPara::TensorDescription(ToStorageShape({}), dtype, ge::FORMAT_ND)},
        MakeAttrs(dim, unbiased, keepdim, invert, epsilon, correction));
}

} // namespace

class ReduceStdWithMeanInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdWithMeanInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ReduceStdWithMeanInferShapeTest TearDown" << std::endl; }
};

// 2D float32，dim=[1]，keepdim=false → 删除归约轴
TEST_F(ReduceStdWithMeanInferShapeTest, test_keepdim_false)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_FLOAT, {1});
    std::vector<std::vector<int64_t>> expectShape = {{4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// 2D float32，dim=[1]，keepdim=true → 归约轴置 1
TEST_F(ReduceStdWithMeanInferShapeTest, test_keepdim_true)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_FLOAT, {1}, true, true);
    std::vector<std::vector<int64_t>> expectShape = {{4, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// dim=[] 全轴归约 → 标量
TEST_F(ReduceStdWithMeanInferShapeTest, test_all_reduce)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_FLOAT16, {});
    std::vector<std::vector<int64_t>> expectShape = {{}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// 负数轴 -1 归一化到 rank-1
TEST_F(ReduceStdWithMeanInferShapeTest, test_negative_dim)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_BF16, {-1}, false, false, false, 0.001f, 0);
    std::vector<std::vector<int64_t>> expectShape = {{4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// 多轴 + keepdim=true：dim=[0,2] → 非归约轴保留
TEST_F(ReduceStdWithMeanInferShapeTest, test_multi_dim_keepdim)
{
    gert::InfershapeContextPara para = MakePara({2, 3, 4}, ge::DT_FLOAT, {0, 2}, true, true);
    std::vector<std::vector<int64_t>> expectShape = {{1, 3, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// 1D 仅轴全归约 → 标量
TEST_F(ReduceStdWithMeanInferShapeTest, test_rank1_reduce_only_axis)
{
    gert::InfershapeContextPara para = MakePara({8}, ge::DT_FLOAT, {0});
    std::vector<std::vector<int64_t>> expectShape = {{}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// 标量输入（rank=0）keepdim=false → 0-D
TEST_F(ReduceStdWithMeanInferShapeTest, test_scalar_input_keepdim_false)
{
    gert::InfershapeContextPara para = MakePara({}, ge::DT_FLOAT, {});
    std::vector<std::vector<int64_t>> expectShape = {{}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// 标量输入（rank=0）keepdim=true → [1]
TEST_F(ReduceStdWithMeanInferShapeTest, test_scalar_input_keepdim_true)
{
    gert::InfershapeContextPara para = MakePara({}, ge::DT_FLOAT, {}, true, true);
    std::vector<std::vector<int64_t>> expectShape = {{1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// unknown rank(-2) 透传
TEST_F(ReduceStdWithMeanInferShapeTest, test_unknown_rank_passthrough)
{
    gert::InfershapeContextPara para = MakePara({-2}, ge::DT_FLOAT, {0});
    std::vector<std::vector<int64_t>> expectShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// dim 越界 → GRAPH_FAILED
TEST_F(ReduceStdWithMeanInferShapeTest, test_dim_out_of_range_rejected)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_FLOAT, {2});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// dim 重复（1 与 -1 归一化后重复）→ GRAPH_FAILED
TEST_F(ReduceStdWithMeanInferShapeTest, test_duplicate_dim_rejected)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_FLOAT, {1, -1});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 负数 dim 越界（-3 < -rank）→ GRAPH_FAILED
TEST_F(ReduceStdWithMeanInferShapeTest, test_negative_dim_out_of_range_rejected)
{
    gert::InfershapeContextPara para = MakePara({4, 8}, ge::DT_FLOAT, {-3});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// rank=9 超上限 → GRAPH_FAILED
TEST_F(ReduceStdWithMeanInferShapeTest, test_rank_9_rejected)
{
    gert::InfershapeContextPara para = MakePara({1, 1, 1, 1, 1, 1, 1, 1, 2}, ge::DT_FLOAT, {8});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
