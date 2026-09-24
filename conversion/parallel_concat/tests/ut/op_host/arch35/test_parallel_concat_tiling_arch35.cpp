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
 * \file test_parallel_concat_tiling_arch35.cpp
 * \brief TilingFunc UT for ParallelConcat (ascend950 / arch35)
 *
 * 平台 mock 统一 24 AIV 核 / 253952 B UB（与 tests/tiling pattern-B 基线一致），
 * 正例 oracle 为独立手算值：8 字段 TilingData（布局手工转写自
 * op_kernel/arch35/parallel_concat_tiling_struct.h）+ blockDim + tilingKey。
 * 负例断言 ExecuteTiling 返回 false（GRAPH_FAILED 拒绝分支）。
 */

#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/parallel_concat_tiling_arch35.h"

namespace {

constexpr uint32_t kCoreNum = 24u;      // AIV 核数（对齐真实 Ascend950 快照）
constexpr uint64_t kUbSize = 253952ULL; // UB 字节数（对齐真实 Ascend950 快照）

// TilingData 8 字段布局 —— 手工转写自 op_kernel/arch35/
// parallel_concat_tiling_struct.h（字段名/顺序/宽度一致，oracle 独立性要求）。
struct PcTilingData {
    uint64_t n;
    uint64_t rowElems;
    uint64_t rowBytes;
    uint64_t totalBytes;
    uint8_t dtypeSize;
    uint32_t numActiveCores;
    uint32_t perCoreChunks;
    uint32_t bufferSize;
};

// 输入张量描述：N 个同形 [1, dims...] ND 输入
std::vector<gert::TilingContextPara::TensorDescription> MakeInputs(size_t n, const std::vector<int64_t>& dims,
                                                                   ge::DataType dtype, ge::Format fmt = ge::FORMAT_ND)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
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

// 输出张量描述：[N, dims...]，dtype 与输入一致
gert::TilingContextPara::TensorDescription MakeOutput(size_t n, const std::vector<int64_t>& dims, ge::DataType dtype,
                                                      ge::Format fmt = ge::FORMAT_ND)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().AppendDim(static_cast<int64_t>(n));
    shape.MutableStorageShape().AppendDim(static_cast<int64_t>(n));
    for (int64_t d : dims) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
    return gert::TilingContextPara::TensorDescription(shape, dtype, fmt);
}

// 构造 para：attrs 按 IR 顺序（shape ListInt 在前、N Int 在后），
// 动态输入实例数 {N}/{1}，平台 24 核 / 253952 B UB。
// CompileInfo：TilingParse 为 no-op，faker 仅要求非空指针（空载体结构体）。
gert::TilingContextPara MakePara(const std::vector<int64_t>& attrShape, int64_t attrN,
                                 const std::vector<gert::TilingContextPara::TensorDescription>& inputs,
                                 ge::DataType dtype, uint32_t coreNum = kCoreNum, uint64_t ubSize = kUbSize)
{
    static optiling::ParallelConcatCompileInfo compileInfo; // faker 要求非空 CompileInfo
    std::vector<int64_t> outDims = (attrShape.size() > 1UL) ?
                                       std::vector<int64_t>(attrShape.begin() + 1, attrShape.end()) :
                                       std::vector<int64_t>{};
    return gert::TilingContextPara("ParallelConcat", inputs, {MakeOutput(static_cast<size_t>(attrN), outDims, dtype)},
                                   {
                                       {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(attrShape)},
                                       {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(attrN)},
                                   },
                                   {static_cast<uint32_t>(inputs.size())}, {1u}, &compileInfo, coreNum, ubSize);
}

// 正例断言：8 字段 + blockDim（== numActiveCores）+ tilingKey + workspace[0]=0。
void ExpectTiling(const TilingInfo& info, uint64_t n, uint64_t rowElems, uint64_t rowBytes, uint64_t totalBytes,
                  uint8_t dtypeSize, uint32_t numActiveCores, uint32_t perCoreChunks, uint32_t bufferSize,
                  uint64_t tilingKey)
{
    ASSERT_EQ(info.tilingDataSize, sizeof(PcTilingData));
    PcTilingData td;
    (void)memcpy(&td, info.tilingData.get(), sizeof(PcTilingData));
    EXPECT_EQ(td.n, n);
    EXPECT_EQ(td.rowElems, rowElems);
    EXPECT_EQ(td.rowBytes, rowBytes);
    EXPECT_EQ(td.totalBytes, totalBytes);
    EXPECT_EQ(td.dtypeSize, dtypeSize);
    EXPECT_EQ(td.numActiveCores, numActiveCores);
    EXPECT_EQ(td.perCoreChunks, perCoreChunks);
    EXPECT_EQ(td.bufferSize, bufferSize);
    EXPECT_EQ(info.blockNum, numActiveCores); // SetBlockDim 恒等于 numActiveCores
    EXPECT_EQ(info.tilingKey, tilingKey);
    ASSERT_EQ(info.workspaceSizes.size(), 1u); // 无 GM workspace
    EXPECT_EQ(info.workspaceSizes[0], 0u);
}

} // namespace

class ParallelConcatTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// ---------- 正例：手算 oracle（24 核 / 253952 B，ubHalf=126976）----------

// 窄分支 SIMT + 双维度封顶回落单核：rowBytes=32B(<64B) → key 0；
// totalBytes=128B < 256B/核 → narrowCoreCap=1，单核 4 chunk。
TEST_F(ParallelConcatTilingTest, NarrowSimtNarrowCapToSingleCore)
{
    auto para = MakePara({4, 8}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 4, 8, 32, 128, 4, 1, 4, 65536, 0);
}

// rank-1 标量输入：L=1，rowBytes=4B → key 0；总量 20B → 单核 5 chunk。
TEST_F(ParallelConcatTilingTest, Rank1ScalarRows)
{
    auto para = MakePara({5}, 5, MakeInputs(5, {1}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 5, 1, 4, 20, 4, 1, 5, 65536, 0);
}

// 空 tensor（任一尾维为 0）：rowBytes=0 → key 0 + 单核零迭代短路。
TEST_F(ParallelConcatTilingTest, EmptyTensorShortCircuit)
{
    auto para = MakePara({3, 0}, 3, MakeInputs(3, {1, 0}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 3, 0, 0, 0, 4, 1, 0, 65536, 0);
}

// 宽分支 SIMD（fp16）：rowBytes=2048B >= 64B → key 1；8 chunk 摊 8 核。
// dtypeSize=2 证明 dtypeSize 链路消费真实 dtype。
TEST_F(ParallelConcatTilingTest, WideSimdFp16)
{
    auto para = MakePara({8, 1024}, 8, MakeInputs(8, {1, 1024}, ge::DT_FLOAT16), ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 8, 1024, 2048, 16384, 2, 8, 1, 65536, 1);
}

// 中行欠并行补核（64KB <= rowBytes < 512KB）：n=2 × 2 chunk < 24 核 → 触发；
// 小总量护栏 maxUseful = max(2, 327680/48K=6) = 6 → target = min(12, 20, ceil(6/2)=3)
// = 3 → bufferSize = ceil32(ceilDiv(163840,3)) = 54624 → totalChunks=6 核（每核 1 chunk）。
TEST_F(ParallelConcatTilingTest, MidRowUnderfillSplit)
{
    auto para = MakePara({2, 40960}, 2, MakeInputs(2, {1, 40960}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 2, 40960, 163840, 327680, 4, 6, 1, 54624, 1);
}

// 大行 overlapFill（rowBytes=1MB >= 512KB）：target=ceilDiv(1M,48K)=22 →
// bufferSize=ceil32(ceilDiv(1M,22))=47680 → totalChunks=44 →
// 核数封顶 min(24, ceilDiv(44,2))=22，每核恒 2 chunk（MTE2/MTE3 重叠保持）。
TEST_F(ParallelConcatTilingTest, LargeRowOverlapFill)
{
    auto para = MakePara({2, 262144}, 2, MakeInputs(2, {1, 262144}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 2, 262144, 1048576, 2097152, 4, 22, 2, 47680, 1);
}

// UB 约束绑定：rowBytes=160KB > ubHalf=126976 → bufferSize=126976（非 64KB 基座档）；
// n=24 × 2 chunk = 48 >= 24 已满核 → 不触发补核，bufferSize 保持 UB 档。
TEST_F(ParallelConcatTilingTest, UbBoundBufferSize)
{
    auto para = MakePara({24, 40000}, 24, MakeInputs(24, {1, 40000}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 24, 40000, 160000, 3840000, 4, 24, 2, 126976, 1);
}

// 平台敏感性：rowBytes=16384B < 64KB 中行门限 → 不补核（legacy）；
// 24 核：16 chunk 摊 16 核；8 核：16 chunk 摊 8 核每核 2 chunk。
TEST_F(ParallelConcatTilingTest, PlatformCoreSensitivity24Core)
{
    auto para = MakePara({16, 4096}, 16, MakeInputs(16, {1, 4096}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 16, 4096, 16384, 262144, 4, 16, 1, 65536, 1);
}

TEST_F(ParallelConcatTilingTest, PlatformCoreSensitivity8Core)
{
    auto para = MakePara({16, 4096}, 16, MakeInputs(16, {1, 4096}, ge::DT_FLOAT), ge::DT_FLOAT, 8u, kUbSize);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 16, 4096, 16384, 262144, 4, 8, 2, 65536, 1);
}

// int8 一字节 dtype：rowBytes=100B → key 1（>=64B 边界之上），2 chunk 摊 2 核。
TEST_F(ParallelConcatTilingTest, WideSimdInt8)
{
    auto para = MakePara({2, 100}, 2, MakeInputs(2, {1, 100}, ge::DT_INT8), ge::DT_INT8);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 2, 100, 100, 200, 1, 2, 1, 65536, 1);
}

// int64 八字节 dtype：rowBytes=256B → key 1，2 核各 1 chunk。
TEST_F(ParallelConcatTilingTest, WideSimdInt64)
{
    auto para = MakePara({2, 32}, 2, MakeInputs(2, {1, 32}, ge::DT_INT64), ge::DT_INT64);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 2, 32, 256, 512, 8, 2, 1, 65536, 1);
}

// 64B 路由边界：rowBytes == 64B → key 1（唯一归 key 1）；2 chunk 摊 2 核。
TEST_F(ParallelConcatTilingTest, RoutingThreshold64B)
{
    auto para = MakePara({2, 16}, 2, MakeInputs(2, {1, 16}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 2, 16, 64, 128, 4, 2, 1, 65536, 1);
}

// rank-8 输入合轴：L = 2×3×4×5×6×7×8 = 40320，fp32 rowBytes=161280B →
// 中行补核（>=64KB）+ 小总量护栏：maxUseful = max(2, 322560/48K=6) = 6 →
// target = min(12, 19, 3) = 3 → bufferSize=ceil32(53760)=53760，6 核每核 1 chunk。
TEST_F(ParallelConcatTilingTest, Rank8CollapsedRow)
{
    auto para = MakePara({2, 2, 3, 4, 5, 6, 7, 8}, 2, MakeInputs(2, {1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT),
                         ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ExpectTiling(info, 2, 40320, 161280, 322560, 4, 6, 1, 53760, 1);
}

// ---------- 负例：校验链拒绝分支（断言 ExecuteTiling 返回 false）----------

// dtype 白名单外（DT_COMPLEX64）
TEST_F(ParallelConcatTilingTest, N01UnsupportedDtype)
{
    auto para = MakePara({1, 8}, 1, MakeInputs(1, {1, 8}, ge::DT_COMPLEX64), ge::DT_COMPLEX64);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 输入间 dtype 不一致
TEST_F(ParallelConcatTilingTest, N02InputDtypeMismatch)
{
    auto inputs = MakeInputs(1, {1, 8}, ge::DT_FLOAT);
    inputs.emplace_back(MakeInputs(1, {1, 8}, ge::DT_FLOAT16)[0]);
    auto para = MakePara({2, 8}, 2, inputs, ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 输出 dtype 与输入不一致
TEST_F(ParallelConcatTilingTest, N03OutputDtypeMismatch)
{
    static optiling::ParallelConcatCompileInfo compileInfo;
    auto para = gert::TilingContextPara("ParallelConcat", MakeInputs(1, {1, 8}, ge::DT_FLOAT),
                                        {MakeOutput(1, {8}, ge::DT_FLOAT16)},
                                        {
                                            {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                        },
                                        {1u}, {1u}, &compileInfo, kCoreNum, kUbSize);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// rank 9 超上界
TEST_F(ParallelConcatTilingTest, N05RankOutOfRange)
{
    const std::vector<int64_t> rank9{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto para = MakePara(rank9, 1, MakeInputs(1, rank9, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// attr N < 1
TEST_F(ParallelConcatTilingTest, N06AttrNZero)
{
    auto para = MakePara({0, 8}, 0, MakeInputs(1, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// attr shape[0] != N
TEST_F(ParallelConcatTilingTest, N07AttrShape0Mismatch)
{
    auto para = MakePara({5, 8}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// attr N != len(values)
TEST_F(ParallelConcatTilingTest, N08AttrNNeqInputs)
{
    auto para = MakePara({4, 8}, 4, MakeInputs(3, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// attr shape 含负维
TEST_F(ParallelConcatTilingTest, N09AttrNegativeDim)
{
    auto para = MakePara({4, -8}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// attr shape 为空 ListInt
TEST_F(ParallelConcatTilingTest, N10AttrShapeEmpty)
{
    auto para = MakePara({}, 1, MakeInputs(1, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 输入首维 != 1
TEST_F(ParallelConcatTilingTest, N11FirstDimNotOne)
{
    auto para = MakePara({2, 8}, 2, MakeInputs(2, {2, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 输入间异形
TEST_F(ParallelConcatTilingTest, N12InputsDiffer)
{
    auto inputs = MakeInputs(1, {1, 8}, ge::DT_FLOAT);
    inputs.emplace_back(MakeInputs(1, {1, 16}, ge::DT_FLOAT)[0]);
    auto para = MakePara({2, 8}, 2, inputs, ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// attr shape 尾维与输入尾维不一致
TEST_F(ParallelConcatTilingTest, N13AttrShapeTailDiff)
{
    auto para = MakePara({4, 16}, 4, MakeInputs(4, {1, 8}, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 输出 shape 与 attr shape 不一致（rank 不符）
TEST_F(ParallelConcatTilingTest, N14OutputShapeMismatch)
{
    static optiling::ParallelConcatCompileInfo compileInfo;
    auto para = gert::TilingContextPara("ParallelConcat", MakeInputs(4, {1, 8}, ge::DT_FLOAT),
                                        {MakeOutput(4, {8, 2}, ge::DT_FLOAT)},
                                        {
                                            {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 8})},
                                            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                        },
                                        {4u}, {1u}, &compileInfo, kCoreNum, kUbSize);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// rowElems 溢出（2^62 × 2^62）
TEST_F(ParallelConcatTilingTest, N15RowElemsOverflow)
{
    constexpr int64_t huge = 4611686018427387904LL; // 2^62
    const std::vector<int64_t> hugeDims{1, huge, huge};
    auto para = MakePara(hugeDims, 1, MakeInputs(1, hugeDims, ge::DT_FLOAT), ge::DT_FLOAT);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// rowBytes 溢出（2^62 × dtypeSize 8）
TEST_F(ParallelConcatTilingTest, N16RowBytesOverflow)
{
    constexpr int64_t huge = 4611686018427387904LL; // 2^62
    const std::vector<int64_t> hugeDims{1, huge};
    auto para = MakePara(hugeDims, 1, MakeInputs(1, hugeDims, ge::DT_UINT64), ge::DT_UINT64);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 退化平台快照：ubSize=96 → ubHalf=floor32(48)=32 < 64B 路由阈值
TEST_F(ParallelConcatTilingTest, N17DegenerateUb)
{
    auto para = MakePara({1, 16}, 1, MakeInputs(1, {1, 16}, ge::DT_FLOAT), ge::DT_FLOAT, kCoreNum, 96ULL);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 退化平台快照：coreNum=0（SIMD 路径 numActiveCores=min(totalChunks,0)=0）
TEST_F(ParallelConcatTilingTest, N18CoreNumZero)
{
    auto para = MakePara({2, 64}, 2, MakeInputs(2, {1, 64}, ge::DT_FLOAT), ge::DT_FLOAT, 0u, kUbSize);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}
