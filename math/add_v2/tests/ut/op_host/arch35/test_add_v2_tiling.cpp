/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_v2_tiling.cpp
 * \brief add_v2 tiling UT for ascend950 (arch35)
 */

#include "math/add_v2/op_host/arch35/add_v2_tiling_arch35.h"
#include <iostream>
#include <limits>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

// 空 Tensor 分支的 tiling key = 65550 = 0x1000E：
//   低 16 位 0x000E = 14，是 schMode 999 在 BRC_TEMP_SCH_MODE_KEY_DECL 取值表
//                     (1,2,101,102,103,104,109,201,202,301,302,303,304,305,999) 里的序号；
//   第 16 位 = userDef = 1。
// 常规通路是 userDef = 0，所以高位为 0，原有用例的 key 不受影响（仍为 8）。
static constexpr uint64_t ADD_V2_UT_EMPTY_TILING_KEY = 65550;
static constexpr uint64_t ADD_V2_UT_CORE_NUM = 64;
static constexpr uint64_t ADD_V2_UT_UB_SIZE = 245760;
static constexpr uint64_t ADD_V2_UT_WORKSPACE_SIZE = 16777216;

class AddV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddV2Tiling TearDown" << std::endl; }
};

static optiling::AddV2CompileInfoArch35 g_addV2CompileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                              ADD_V2_UT_WORKSPACE_SIZE};

static gert::TilingContextPara MakeAddV2TilingPara(const gert::StorageShape& x1Shape, const gert::StorageShape& x2Shape,
                                                   const gert::StorageShape& yShape, ge::DataType x1Dtype,
                                                   ge::DataType x2Dtype, ge::DataType yDtype)
{
    return gert::TilingContextPara("AddV2",
                                   {
                                       {x1Shape, x1Dtype, ge::FORMAT_ND},
                                       {x2Shape, x2Dtype, ge::FORMAT_ND},
                                   },
                                   {
                                       {yShape, yDtype, ge::FORMAT_ND},
                                   },
                                   &g_addV2CompileInfo);
}

static gert::TilingContextPara MakeAddV2TilingPara(const gert::StorageShape& x1Shape, const gert::StorageShape& x2Shape,
                                                   const gert::StorageShape& yShape, ge::DataType dtype = ge::DT_FLOAT)
{
    return MakeAddV2TilingPara(x1Shape, x2Shape, yShape, dtype, dtype, dtype);
}

TEST_F(AddV2Tiling, add_v2_tiling_fp32)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// workspace 必须来自 CompileInfo/平台查询结果，不能在 Tiling 中写死。
TEST_F(AddV2Tiling, add_v2_workspace_uses_compile_info_value)
{
    constexpr uint64_t customWorkspaceSize = 4096;
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE, customWorkspaceSize};
    gert::StorageShape shape = {{8, 8}, {8, 8}};
    gert::TilingContextPara para("AddV2",
                                 {
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 8, std::vector<size_t>{customWorkspaceSize});
}

TEST_F(AddV2Tiling, add_v2_tiling_fp16_broadcast)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_bf16)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_int64)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_complex64)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_invalid_dtype)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ── 空 Tensor ──────────────────────────────────────────────────────────────
// ATVOSS 的 BroadcastBaseTiling 在合轴后显式拒绝 0 元素，空 Tensor 必须走
// 自定义模板分支（schMode 999 + userDef 1），blockDim = 1，kernel 侧直接返回。
TEST_F(AddV2Tiling, add_v2_tiling_empty_1d)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_empty_2d)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 空张量 + 广播：x1 空、x2 是标量，输出仍为空
TEST_F(AddV2Tiling, add_v2_tiling_empty_broadcast)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 3}, {1, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 高维中间维为 0
TEST_F(AddV2Tiling, add_v2_tiling_empty_highrank)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {true, ADD_V2_UT_CORE_NUM, ADD_V2_UT_UB_SIZE,
                                                    ADD_V2_UT_WORKSPACE_SIZE};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 末轴为 0：补齐与首轴/中间轴不同的空 Tensor 位置。
TEST_F(AddV2Tiling, add_v2_tiling_empty_last_axis)
{
    gert::StorageShape x1Shape = {{2, 3, 0}, {2, 3, 0}};
    gert::StorageShape x2Shape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape yShape = {{2, 3, 0}, {2, 3, 0}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_SUCCESS, ADD_V2_UT_EMPTY_TILING_KEY,
                    std::vector<size_t>{16777216});
}

// 多个轴为 0，且分别与对端的 1 广播。
TEST_F(AddV2Tiling, add_v2_tiling_empty_multiple_axes)
{
    gert::StorageShape x1Shape = {{0, 2, 1, 0}, {0, 2, 1, 0}};
    gert::StorageShape x2Shape = {{1, 1, 4, 1}, {1, 1, 4, 1}};
    gert::StorageShape yShape = {{0, 2, 4, 0}, {0, 2, 4, 0}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_SUCCESS, ADD_V2_UT_EMPTY_TILING_KEY,
                    std::vector<size_t>{16777216});
}

// 即使 0 轴在超大维之后，数学上的 numel 仍为 0，不应被 GetShapeSize 的乘法顺序误判为溢出。
TEST_F(AddV2Tiling, add_v2_tiling_empty_late_zero_after_large_dims)
{
    const int64_t maxDim = std::numeric_limits<int64_t>::max();
    gert::StorageShape shape = {{maxDim, 2, 0}, {maxDim, 2, 0}};
    ExecuteTestCase(MakeAddV2TilingPara(shape, shape, shape), ge::GRAPH_SUCCESS, ADD_V2_UT_EMPTY_TILING_KEY,
                    std::vector<size_t>{16777216});
}

// rank_range=[1, 8]：rank0 标量必须在进入定长 broadcast 结构前拒绝。
TEST_F(AddV2Tiling, add_v2_tiling_rank0_failed)
{
    gert::StorageShape scalar = {{}, {}};
    ExecuteTestCase(MakeAddV2TilingPara(scalar, scalar, scalar), ge::GRAPH_FAILED);
}

// rank=8 是支持上界，不能被上界保护误拒绝。
TEST_F(AddV2Tiling, add_v2_tiling_rank8_success)
{
    gert::StorageShape shape = {{2, 1, 1, 1, 1, 1, 1, 2}, {2, 1, 1, 1, 1, 1, 1, 2}};
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(MakeAddV2TilingPara(shape, shape, shape), info));
}

// 2^31 单轴无法在 ST 中实际分配数据；这里仅构造 shape 元数据，验证合法大维与 [1] 广播可完成 tiling。
TEST_F(AddV2Tiling, add_v2_tiling_axis_2_to_31_broadcast_success)
{
    constexpr int64_t largeDim = static_cast<int64_t>(1) << 31;
    gert::StorageShape largeShape = {{largeDim}, {largeDim}};
    gert::StorageShape oneShape = {{1}, {1}};
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(MakeAddV2TilingPara(largeShape, oneShape, largeShape, ge::DT_UINT8), info));
}

// rank=9 即使可被 ATVOSS 合轴为更低 rank，也违反原始接口契约，必须先拒绝。
TEST_F(AddV2Tiling, add_v2_tiling_rank9_failed_before_axis_merge)
{
    gert::StorageShape shape = {
        {2, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 1, 1, 1, 1, 1, 1, 1, 1},
    };
    ExecuteTestCase(MakeAddV2TilingPara(shape, shape, shape), ge::GRAPH_FAILED);
}

// 保护逻辑检查的是原始 rank，而不只是可能已被格式转换压缩的 storage rank。
TEST_F(AddV2Tiling, add_v2_tiling_origin_rank9_storage_rank1_failed)
{
    gert::StorageShape x1Shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2}};
    gert::StorageShape x2Shape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_input_dtype_mismatch_failed)
{
    gert::StorageShape shape = {{8, 8}, {8, 8}};
    ExecuteTestCase(MakeAddV2TilingPara(shape, shape, shape, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16),
                    ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_output_dtype_mismatch_failed)
{
    gert::StorageShape shape = {{8, 8}, {8, 8}};
    ExecuteTestCase(MakeAddV2TilingPara(shape, shape, shape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT),
                    ge::GRAPH_FAILED);
}

// 不支持 dtype 的空 Tensor 也必须失败，证明 dtype 校验没有被空输出早返绕过。
TEST_F(AddV2Tiling, add_v2_tiling_empty_unsupported_dtype_failed)
{
    gert::StorageShape emptyShape = {{0, 3}, {0, 3}};
    ExecuteTestCase(MakeAddV2TilingPara(emptyShape, emptyShape, emptyShape, ge::DT_DOUBLE), ge::GRAPH_FAILED);
}

// 覆盖剩余整数 dispatch 分支；每个 dtype 都必须通过同 dtype 白名单校验并完成 tiling。
TEST_F(AddV2Tiling, add_v2_tiling_supported_integer_dispatches)
{
    gert::StorageShape shape = {{16}, {16}};
    const std::vector<ge::DataType> dtypes = {ge::DT_UINT8, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32};
    for (const ge::DataType dtype : dtypes) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        TilingInfo info;
        EXPECT_TRUE(ExecuteTiling(MakeAddV2TilingPara(shape, shape, shape, dtype), info));
    }
}

// x1/x2 不能广播；伪造一个空 y 不能再触发早返成功。
TEST_F(AddV2Tiling, add_v2_tiling_invalid_broadcast_with_empty_output_failed)
{
    gert::StorageShape x1Shape = {{0, 2}, {0, 2}};
    gert::StorageShape x2Shape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{0, 2}, {0, 2}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_FAILED);
}

// 输入广播结果为 [2, 3]，伪造的空输出 [0, 3] 必须被逐维校验拒绝。
TEST_F(AddV2Tiling, add_v2_tiling_forged_empty_output_failed)
{
    gert::StorageShape x1Shape = {{2, 3}, {2, 3}};
    gert::StorageShape x2Shape = {{1, 3}, {1, 3}};
    gert::StorageShape yShape = {{0, 3}, {0, 3}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_nonempty_output_not_broadcast_result_failed)
{
    gert::StorageShape x1Shape = {{2, 3}, {2, 3}};
    gert::StorageShape x2Shape = {{1, 3}, {1, 3}};
    gert::StorageShape yShape = {{1, 3}, {1, 3}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_negative_input_dim_failed)
{
    gert::StorageShape x1Shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape x2Shape = {{1, 3}, {1, 3}};
    gert::StorageShape yShape = {{-1, 3}, {-1, 3}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_FAILED);
}

// y 含 0 时旧逻辑会直接成功；同时带负维时必须先由 concrete-shape 校验拒绝。
TEST_F(AddV2Tiling, add_v2_tiling_empty_output_with_negative_dim_failed)
{
    gert::StorageShape x1Shape = {{0, 3}, {0, 3}};
    gert::StorageShape x2Shape = {{1, 3}, {1, 3}};
    gert::StorageShape yShape = {{0, -1}, {0, -1}};
    ExecuteTestCase(MakeAddV2TilingPara(x1Shape, x2Shape, yShape), ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_shape_element_count_overflow_failed)
{
    const int64_t maxDim = std::numeric_limits<int64_t>::max();
    gert::StorageShape shape = {{maxDim, 2}, {maxDim, 2}};
    ExecuteTestCase(MakeAddV2TilingPara(shape, shape, shape), ge::GRAPH_FAILED);
}

// 空 Tensor 自定义模板同样必须检查 tiling-data 容量，而不能向 0 字节缓冲区写入。
TEST_F(AddV2Tiling, add_v2_tiling_empty_insufficient_tiling_data_failed)
{
    gert::StorageShape shape = {{0}, {0}};
    gert::TilingContextPara para("AddV2",
                                 {
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &g_addV2CompileInfo, 64, 245760, 0);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_zero_platform_core_failed)
{
    gert::StorageShape shape = {{8}, {8}};
    gert::TilingContextPara para("AddV2",
                                 {
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &g_addV2CompileInfo, 0, 245760);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AddV2Tiling, add_v2_tiling_zero_platform_ub_failed)
{
    gert::StorageShape shape = {{8}, {8}};
    gert::TilingContextPara para("AddV2",
                                 {
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &g_addV2CompileInfo, 64, 0);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 注册入口必须在任何日志或成员访问前处理根 context 为空。
TEST_F(AddV2Tiling, add_v2_tiling_root_context_null_failed)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto functionStruct = spaceRegistry->GetOpImpl("AddV2");
    ASSERT_NE(functionStruct, nullptr);
    ASSERT_NE(functionStruct->tiling, nullptr);
    EXPECT_EQ(functionStruct->tiling(nullptr), ge::GRAPH_FAILED);
}
