/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_parallel_concat_infershape.cpp
 * \brief InferShape UT for ParallelConcat (ascend950 / arch35)
 *
 * 覆盖 op_host/parallel_concat_infershape.cpp 的校验链：
 * null 防御不可构造（faker 恒有效），其余按段覆盖——
 *   - attr 值域：N>=1、shape 非空、非负、shape[0]==N、N==len(values)
 *   - 逐输入：ND 格式、rank∈[1,8]、首维==1、同形
 *   - 双源合一：attr shape[1:] == values.shape[1:]（动态维 -1 由 attr 收敛）
 *   - unknown-rank 输入（[-2]）跳过逐输入检查
 *   - 规模溢出防护：totalElements 乘积溢出拒绝
 *   - 输出写入 = attr shape
 * 注：声明输出 shape 分支（GetDimNum()>0 的 output 槽）在本仓 faker 下不可达
 * （InferShapeContextFaker::OutputShapes 为 no-op，输出槽恒为空 shape）。
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace {

constexpr int64_t kUnknownDim = -1;  // ge::UNKNOWN_DIM
constexpr int64_t kUnknownRank = -2; // ge::UNKNOWN_DIM_NUM

// 输入张量描述：N 个同形 [1, dims...] 的 ND 输入
std::vector<gert::InfershapeContextPara::TensorDescription> MakeInputs(size_t n, const std::vector<int64_t>& dims,
                                                                       ge::DataType dtype,
                                                                       ge::Format fmt = ge::FORMAT_ND)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
    for (size_t i = 0; i < n; ++i) {
        gert::StorageShape shape;
        for (int64_t d : dims) {
            shape.MutableOriginShape().AppendDim(d);
            shape.MutableStorageShape().AppendDim(d);
        }
        inputs.emplace_back(shape, dtype, fmt);
    }
    return inputs;
}

// 输出张量描述（shape 待推导，初始为空）
gert::InfershapeContextPara::TensorDescription MakeOutput(ge::DataType dtype)
{
    gert::StorageShape shape;
    return gert::InfershapeContextPara::TensorDescription(shape, dtype, ge::FORMAT_ND);
}

// 构造带 attrs（IR 顺序：shape ListInt 在前、N Int 在后）+ 动态输入实例数的 para
gert::InfershapeContextPara MakePara(const std::vector<int64_t>& attrShape, int64_t attrN,
                                     const std::vector<gert::InfershapeContextPara::TensorDescription>& inputs,
                                     ge::DataType dtype)
{
    return gert::InfershapeContextPara("ParallelConcat", inputs, {MakeOutput(dtype)},
                                       {
                                           {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(attrShape)},
                                           {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(attrN)},
                                       },
                                       {static_cast<uint32_t>(inputs.size())}, {1u});
}

} // namespace

class ParallelConcatInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// ---------- 正例：输出 shape = attr shape ----------

TEST_F(ParallelConcatInfershapeTest, InferShapeFp32Basic)
{
    auto para = MakePara({4, 8}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{4, 8}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeRank1)
{
    auto para = MakePara({5}, 5, MakeInputs(5, {1}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{5}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeRank8UpperBound)
{
    auto para = MakePara({2, 2, 3, 4, 5, 6, 7, 8}, 2, MakeInputs(2, {1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT),
                         ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 2, 3, 4, 5, 6, 7, 8}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeEmptyTensor)
{
    auto para = MakePara({3, 0}, 3, MakeInputs(3, {1, 0}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3, 0}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeFp16)
{
    auto para = MakePara({2, 16}, 2, MakeInputs(2, {1, 16}, ge::DT_FLOAT16), ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 16}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeBf16)
{
    auto para = MakePara({2, 16}, 2, MakeInputs(2, {1, 16}, ge::DT_BF16), ge::DT_BF16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 16}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeInt8)
{
    auto para = MakePara({2, 33}, 2, MakeInputs(2, {1, 33}, ge::DT_INT8), ge::DT_INT8);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 33}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeBool)
{
    auto para = MakePara({2, 4}, 2, MakeInputs(2, {1, 4}, ge::DT_BOOL), ge::DT_BOOL);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4}});
}

TEST_F(ParallelConcatInfershapeTest, InferShapeInt64)
{
    auto para = MakePara({2, 4}, 2, MakeInputs(2, {1, 4}, ge::DT_INT64), ge::DT_INT64);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4}});
}

// 动态维（-1）输入：由全定义 attr shape 收敛
TEST_F(ParallelConcatInfershapeTest, InferShapeDynamicInputDim)
{
    auto para = MakePara({2, 16}, 2, MakeInputs(2, {1, kUnknownDim}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 16}});
}

// 动态首维（-1）：首维 == UNKNOWN_DIM 允许，输出仍取 attr
TEST_F(ParallelConcatInfershapeTest, InferShapeDynamicFirstDim)
{
    auto para = MakePara({2, 8}, 2, MakeInputs(2, {kUnknownDim, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 8}});
}

// unknown-rank（[-2]）输入混排：未知秩实例跳过逐输入检查，已知实例作代表
TEST_F(ParallelConcatInfershapeTest, InferShapeUnknownRankMixed)
{
    auto inputs = MakeInputs(1, {1, 8}, ge::DT_FLOAT);
    gert::StorageShape unknownRank;
    unknownRank.MutableOriginShape().AppendDim(kUnknownRank);
    unknownRank.MutableStorageShape().AppendDim(kUnknownRank);
    inputs.emplace_back(unknownRank, ge::DT_FLOAT, ge::FORMAT_ND);
    auto para = MakePara({2, 8}, 2, inputs, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 8}});
}

// 全部 unknown-rank 输入：无代表 shape，attr shape[1:] 校验跳过，输出仍 = attr
TEST_F(ParallelConcatInfershapeTest, InferShapeUnknownRankAll)
{
    gert::StorageShape unknownRank;
    unknownRank.MutableOriginShape().AppendDim(kUnknownRank);
    unknownRank.MutableStorageShape().AppendDim(kUnknownRank);
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
    for (int i = 0; i < 2; ++i) {
        inputs.emplace_back(unknownRank, ge::DT_FLOAT, ge::FORMAT_ND);
    }
    auto para = MakePara({2, 8}, 2, inputs, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 8}});
}

// ---------- 负例：校验链各拒绝分支 ----------

// attr N < 1
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrNZero)
{
    auto para = MakePara({1, 8}, 0, MakeInputs(1, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// attr shape 为空 ListInt
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrShapeEmpty)
{
    auto para = MakePara({}, 1, MakeInputs(1, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// attr shape 含负维（attr 侧必须全定义非负）
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrNegativeDim)
{
    auto para = MakePara({4, -8}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// shape[0] != N
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrShape0Mismatch)
{
    auto para = MakePara({5, 8}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// N != len(values)
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrNNeqInputs)
{
    auto para = MakePara({4, 8}, 4, MakeInputs(3, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// rank 9 超上界
TEST_F(ParallelConcatInfershapeTest, InferShapeRankOutOfRange)
{
    auto para = MakePara({1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, MakeInputs(1, {1, 2, 3, 4, 5, 6, 7, 8, 9}, ge::DT_FLOAT),
                         ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// 首维 != 1
TEST_F(ParallelConcatInfershapeTest, InferShapeFirstDimNotOne)
{
    auto para = MakePara({2, 8}, 2, MakeInputs(2, {2, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// 输入间异形
TEST_F(ParallelConcatInfershapeTest, InferShapeInputsDiffer)
{
    auto inputs = MakeInputs(1, {1, 8}, ge::DT_FLOAT);
    gert::StorageShape other;
    other.MutableOriginShape().AppendDim(1);
    other.MutableStorageShape().AppendDim(1);
    other.MutableOriginShape().AppendDim(16);
    other.MutableStorageShape().AppendDim(16);
    inputs.emplace_back(other, ge::DT_FLOAT, ge::FORMAT_ND);
    auto para = MakePara({2, 8}, 2, inputs, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// attr shape 秩 != 输入秩
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrRankMismatch)
{
    auto para = MakePara({4, 8, 2}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// attr shape 尾维与输入尾维不一致
TEST_F(ParallelConcatInfershapeTest, InferShapeAttrTailMismatch)
{
    auto para = MakePara({4, 16}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// totalElements 乘积溢出（2^62 × 2^62）
TEST_F(ParallelConcatInfershapeTest, InferShapeTotalElementsOverflow)
{
    constexpr int64_t huge = 4611686018427387904LL; // 2^62
    auto para = MakePara({1, huge, huge}, 1, MakeInputs(1, {1, huge, huge}, ge::DT_UINT64), ge::DT_UINT64);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// ---------- 负例：attrs 缺失（REQUIRED 属性缺失的可构造形态）----------

// 缺 attr shape（ListInt, REQUIRED）：GetListInt(0) 返回空指针
TEST_F(ParallelConcatInfershapeTest, InferShapeMissingShapeAttr)
{
    auto para = gert::InfershapeContextPara("ParallelConcat", MakeInputs(1, {1, 8}, ge::DT_FLOAT),
                                            {MakeOutput(ge::DT_FLOAT)},
                                            {
                                                {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            {1u}, {1u});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// 缺 attr N（Int, REQUIRED）：GetInt(1) 返回空指针
TEST_F(ParallelConcatInfershapeTest, InferShapeMissingNAttr)
{
    auto para = gert::InfershapeContextPara(
        "ParallelConcat", MakeInputs(1, {1, 8}, ge::DT_FLOAT), {MakeOutput(ge::DT_FLOAT)},
        {
            {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
        },
        {1u}, {1u});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// ---------- 负例：输入间秩不一致（ShapesCompatible 秩不匹配分支）----------

TEST_F(ParallelConcatInfershapeTest, InferShapeInputsRankDiffer)
{
    auto inputs = MakeInputs(1, {1, 8}, ge::DT_FLOAT);
    gert::StorageShape rank3;
    for (int64_t d : {1, 8, 2}) {
        rank3.MutableOriginShape().AppendDim(d);
        rank3.MutableStorageShape().AppendDim(d);
    }
    inputs.emplace_back(rank3, ge::DT_FLOAT, ge::FORMAT_ND);
    auto para = MakePara({2, 8}, 2, inputs, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
