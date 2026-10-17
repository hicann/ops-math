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
 * \file test_transpose_tiling.cpp
 * \brief broadcast_to tiling ut test
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_arch35.h"

using namespace ge;

class TransposeB4Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransposeB4Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransposeB4Tiling TearDown" << std::endl; }
};

// 场景：B4 4D 输入 [36,204,26,30]，perm={1,0,3,2}，交换最后两轴，输出 [204,36,30,26]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 30}, {36, 204, 26, 30}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 30, 26}, {204, 36, 30, 26}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 292057776140 0 0 0 0 0 0 477360 2340 0 0 0 0 0 0 2340 "
                              "84240 12884901891 128849018906"
                              " 159120 780 30 1 780 28080 1 26 28080 780 26 1 1 8589934595 4294967296 12884901890 "
                              "4294967296 12884901891 4294967296 17179869186 270582939661 108989090103306 4 1 0 3 2 0 "
                              "0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 极小 shape 输入 [2,2,4,8]，perm={1,0,3,2}，交换最后两轴，输出 [2,2,8,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_tiny)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{2, 2, 4, 8}, {2, 2, 4, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 2, 8, 4}, {2, 2, 8, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 4294967297 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 2 8589934594 "
                              "34359738372 64"
                              " 32 8 1 32 64 1 4 64 32 4 1 1 8589934595 4294967296 12884901890 -1 -1 -1 17179869185 "
                              "4294967297 108989090103297 4 1 0 3 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 输入 [2,628,4,8]，perm={1,0,3,2}，批量维为 2 且中间轴较大的尾轴转置，输出 [628,2,8,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_batch_large)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{2, 628, 4, 8}, {2, 628, 4, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{628, 2, 8, 4}, {628, 2, 8, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 450971566081 0 0 0 0 0 0 0 192 0 0 0 0 0 0 0 384 "
                              "25769803778 34359738372"
                              " 20096 32 8 1 32 64 1 4 64 32 4 1 1 8589934595 4294967296 12884901890 -4294967295 "
                              "-4294967292 0 17179869185 227633266690 108989090103297 4 1 0 3 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 2D 输入 [2,628]，perm={1,0}，标准二维转置，输出 [628,2]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_2d_perm_10)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{2, 628}, {2, 628}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{628, 2}, {628, 2}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 4294967297 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 2 4294967297 "
                              "2697239461890"
                              " 0 0 628 1 0 0 1 2 0 0 2 1 4294967296 8589934595 -1 4294967296 -1 -1 -1 8589934593 "
                              "4294967297 108989090103297 2 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 6D 输入 [10,20,6,2,8,10]，perm={1,0,3,2,5,4}，三组相邻轴两两互换，输出 [20,10,2,6,10,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_6d_perm_103254)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[6] = {1, 0, 3, 2, 5, 4};
    gert::TilingContextPara::TensorDescription x({{10, 20, 6, 2, 8, 10}, {10, 20, 6, 2, 8, 10}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10, 2, 6, 10, 8}, {20, 10, 2, 6, 10, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 85899345930 0 0 0 0 0 0 19200 960 0 0 0 0 0 0 960 9600 "
                              "8589934598 42949672968 160 80 10"
                              " 1 80 480 1 8 480 80 8 1 1 8589934595 12884901890 21474836484 -1 -1 -1 17179869186 "
                              "214748364804 108989090103300 6 1 0 3 2 5 4 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 输入 [36,204,26,50]，perm={1,0,2,3}，尾轴不转置且后两轴合并后尾轴较大，输出 [204,36,26,50]。
// 期望：走尾轴不转置的连续行搬移模板，tiling key=10004（N_LAST_TRANSPOSE）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1023_n_last)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 50}, {36, 204, 26, 50}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 26, 50}, {204, 36, 26, 50}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10004;
    string
        expectTilingData = "3 1 0 68 0 0 0 64 1 44 253952 1 0 0 0 0 0 0 0 1 36 204 650 0 0 0 0 0 204 36 650 0 0 0 0 0 "
                           "1 0 2 0 0 0 0 0 0 0 0 0"
                           " 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 3 2 4 1 1 36 204 650 1 1 204 36 650 0 0 0 0 0 0 0 0 0 "
                           "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 大 shape 输入 [36,204,2600,3020]，perm={1,0,3,2}，交换最后两轴，输出 [204,36,3020,2600]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_huge)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{36, 204, 2600, 3020}, {36, 204, 2600, 3020}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 3020, 2600}, {204, 36, 3020, 2600}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 876173328420 137438953492 0 0 0 0 1601808000 7852000 398640 96 0 "
                              "0 0 0 7852000 282672000 132 249600"
                              " 4294967297 412316860548 0 0 3020 1 0 0 1 2600 0 0 2600 1 4294967296 8589934595 -1 "
                              "12884901890 12884901890 188978561116 12884901890 8589934596 274877980384 "
                              "108989090176736 4 1 0 3 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 大 shape 输入 [36,204,2600,3020]，perm={3,2,1,0}，四轴完全逆序，输出 [3020,2600,204,36]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_3210_reverse)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {3, 2, 1, 0};
    gert::TilingContextPara::TensorDescription x({{36, 204, 2600, 3020}, {36, 204, 2600, 3020}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{3020, 2600, 204, 36}, {3020, 2600, 204, 36}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 219043332097 68719479336 0 0 0 0 0 31408000 3020 192 0 0 0 0 0 "
                              "144 7344 3666124800 154618822657 824633720836 0 1601808000 7852000 1 0 1 36 19094400 0 "
                              "19094400 36 1"
                              " 12884901888 4294967298 4294967295 12884901889 12884901889 601295421444 8589934592 "
                              "12884901891 274877940094 108989090136446 4 3 2 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 输入 [36,204,26,66]，perm={1,0,2,3}，尾轴不转置且后两轴合并后尾轴较大，输出 [204,36,26,66]。
// 期望：走尾轴不转置的连续行搬移模板，tiling key=10004（N_LAST_TRANSPOSE）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1023_n_last_tail66)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 66}, {36, 204, 26, 66}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 26, 66}, {204, 36, 26, 66}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10004;
    string
        expectTilingData = "3 1 0 68 0 0 0 64 1 44 253952 1 0 0 0 0 0 0 0 1 36 204 858 0 0 0 0 0 204 36 858 0 0 0 0 0 "
                           "1 0 2 0 0 0 0 0 0 0 0 0"
                           " 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 3 2 4 1 1 36 204 858 1 1 204 36 858 0 0 0 0 0 0 0 0 0 "
                           "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 3D 输入 [8,38,4]，perm={0,2,1}，仅交换后两轴（尾轴转置），输出 [8,4,38]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{8, 38, 4}, {8, 38, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{8, 4, 38}, {8, 4, 38}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 4294967297 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 8 34359738369 "
                              "17179869222 0"
                              " 152 4 1 0 152 1 38 0 152 38 1 4294967296 8589934595 4294967295 8589934593 -1 -1 -1 "
                              "12884901889 4294967297 108989090103297 3 0 2 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 3D 输入 [672,7,83]，perm={1,2,0}，尾轴参与转置，输出 [7,83,672]。
// 期望：B4 双切分无法为该 shape 找到合法切分轴，tiling 返回失败（GRAPH_FAILED），不接受该 shape。
TEST_F(TransposeB4Tiling, b4_3d_perm_120_dual_cut_fail)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {1, 2, 0};
    gert::TilingContextPara::TensorDescription x({{672, 7, 83}, {672, 7, 83}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{7, 83, 672}, {7, 83, 672}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "3 1 0 68 0 0 0 64 1 44 253952 1 0 0 0 0 0 0 0 36 204 858 0 0 0 0 0 204 36 858 0 0 0 0 0 "
                              "1 0 2 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 3 2 4 1 1 36 204 858 1 1 204 36 858 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：FLOAT4_E1M2 3D 输入 [32,8,55480]，perm={2,0,1}，4bit 类型的尾轴转置，输出 [55480,32,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_e1m2_perm_201)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{32, 8, 55480}, {32, 8, 55480}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{55480, 32, 8}, {55480, 32, 8}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 1241245548545 0 0 0 0 0 0 0 192 0 0 0 0 0 0 0 49152 "
                              "4294967297 824633721088"
                              " 0 0 55480 1 0 0 1 256 0 0 256 1 4294967296 8589934595 -1 4294967296 -4294967295 "
                              "-4294967112 0 8589934593 249108103173 108989090103300 2 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：FLOAT4_E2M1 3D 输入 [3,15968,6]，perm={0,2,1}，4bit 类型的尾轴转置，输出 [3,6,15968]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_e2m1_perm_021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{3, 15968, 6}, {3, 15968, 6}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{3, 6, 15968}, {3, 6, 15968}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 176093659139 0 0 0 0 0 0 95808 2376 0 0 0 0 0 0 95808 "
                              "396 4294967297 25769804172"
                              " 0 0 6 1 0 0 1 15968 0 0 15968 1 4294967296 8589934595 -1 8589934593 -4294967295 "
                              "-4294967168 4294967297 8589934594 266287972354 108989090103297 3 0 2 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 大 shape 输入 [3690,204,2600,3020]，perm={1,3,2,0}，输出 [204,3020,2600,3690]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1320_large)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 3, 2, 0};
    gert::TilingContextPara::TensorDescription x({{3690, 204, 2600, 3020}, {3690, 204, 2600, 3020}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 3020, 2600, 3690}, {204, 3020, 2600, 3690}},
                                                   ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 876173328412 137438956072 0 0 0 0 211438656000 7852000 3020 96 0 "
                              "0 0 0 132 28973880000 3690 921024000"
                              " 4294967297 412316860548 0 0 1601808000 1 0 0 1 9594000 0 0 9594000 1 4294967296 "
                              "8589934595 -1 12884901888 12884901888 188978561150 12884901888 8589934596 274885332544 "
                              "108989097528896 4 1 3 2 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 高瘦 shape 输入 [67785,18,4,2]，perm={1,3,0,2}，输出 [18,2,67785,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1302_tall)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 3, 0, 2};
    gert::TilingContextPara::TensorDescription x({{67785, 18, 4, 2}, {67785, 18, 4, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{18, 2, 67785, 4}, {18, 2, 67785, 4}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 77309416977 0 0 0 0 0 0 1728 8 0 0 0 0 0 0 48 542280 "
                              "51539607553 8589934596 0 144"
                              " 2 1 0 4 1 271140 0 271140 4 1 12884901888 8589934593 4294967295 12884901890 "
                              "-4294967296 -4294967287 0 12884901890 274877908533 108989090104871 4 1 3 0 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 3D 输入 [630,8112,40]，perm={1,0,2}，尾轴不转置，输出 [8112,630,40]。
// 期望：按 int8 视图处理后尾轴（40/2=20）小于 32，走 NDDMA 双轴切分模板，
//       tiling key=10003（CUT_TWICE）；tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_102_cut_twice)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{630, 8112, 40}, {630, 8112, 40}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{8112, 630, 40}, {8112, 630, 40}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10003;
    string expectTilingData =
        "3 1 1 25 315 12 0 64 10 10 253952 1 647 648 649 0 0 0 0 1 630 8112 20 0 0 0 0 0 8112 630 20 0 0 0 0 0 1 0 2 0"
        " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 3 2 4 1 1 630 8112 20 1 1 8112 630 20 1 1 315 25 20 1 1 25"
        " 315 20 1 1 315 12 20 1 1 12 315 20 1 1 0 25 20 1 1 25 0 20 1 1 0 12 20 1 1 12 0 20 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 3D 小 shape 输入 [456,848,8]，perm={1,0,2}，尾轴不转置，输出 [848,456,8]。
// 期望：数据量低于小 shape 阈值，走 SIMT 兜底模板，tiling key=10001（SMALL_SHAPE）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_102_small_shape)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{456, 848, 8}, {456, 848, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{848, 456, 8}, {848, 456, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10001;
    string expectTilingData =
        "3 0 0 0 0 0 0 64 24192 22656 253952 1 0 0 0 0 0 0 0 1 456 848 4 0 0 0 0 0 848 456 4 0 0 0 0 0 1 0 2 0 0 0"
        " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 3 2 4 1 1 456 848 4 1 1 848 456 4 1 1 456 0 4 1 1 0 456 4"
        " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 456 0 4 1 1 0 456 4 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 3D 输入 [456,847,8]，perm={0,2,1}，首轴为偶、中间轴为奇的尾轴转置，输出 [456,8,847]。
// 期望：B4 双切分无法为该 shape 找到合法切分轴，tiling 返回失败（GRAPH_FAILED），不接受该 shape。
TEST_F(TransposeB4Tiling, b4_3d_perm_021_dual_cut_fail)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{456, 847, 8}, {456, 847, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{456, 8, 847}, {456, 8, 847}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData =
        "4294967297 4294967297 4294967297 133143986193 0 0 0 0 0 0 189952 224 0 0 0 0 0 0 224 102144 120259084289"
        " 34359738396 0 6784 8 1 0 8 3648 1 0 3648 8 1 8589934592 12884901889 4294967295 8589934593 4294967296 "
        "34359738376 4294967296 12884901890 253403070473 108989090103301 3 1 0 2 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 3D 输入 [457,848,8]，perm={0,2,1}，与失败用例相反的首轴为奇、中间轴为偶，输出 [457,8,848]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_021_odd_first_dim)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{457, 848, 8}, {457, 848, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{457, 8, 848}, {457, 8, 848}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData =
        "4294967297 4294967297 4294967297 12884902345 0 0 0 0 0 0 6784 3168 0 0 0 0 0 0 6784 396 4294967297"
        " 34359738764 0 0 8 1 0 0 1 848 0 0 848 1 4294967296 8589934595 -1 8589934593 -4294967295 -4294967240"
        " 4294967297 8589934594 270582939670 108989090103303 3 0 2 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 高瘦 shape 输入 [67785,17,4,2]，perm={1,3,0,2}，输出 [17,2,67785,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1302_tall_batch17)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 3, 0, 2};
    gert::TilingContextPara::TensorDescription x({{67785, 17, 4, 2}, {67785, 17, 4, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{17, 2, 67785, 4}, {17, 2, 67785, 4}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 73014449681 0 0 0 0 0 0 1632 8 0 0 0 0 0 0 48 542280 "
                              "51539607553 8589934596 0"
                              " 136 2 1 0 4 1 271140 0 271140 4 1 12884901888 8589934593 4294967295 12884901890 "
                              "-4294967296 -4294967287 0 12884901890 274877908445 108989090104766 4 1 3 0 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 输入 [6775,17,48,20]，perm={1,0,3,2}，小批量交换最后两轴，输出 [17,6775,20,48]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_small_batch)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{6775, 17, 48, 20}, {6775, 17, 48, 20}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{17, 6775, 20, 48}, {17, 6775, 20, 48}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 25769806035 0 0 0 0 0 0 48960 2880 0 0 0 0 0 0 2880 "
                              "19512000 12884901891 85899345968"
                              " 16320 960 20 1 960 6504000 1 48 6504000 960 48 1 1 8589934595 4294967296 12884901890 "
                              "4294967296 8589934593 4294967296 17179869186 274877907156 108989090103494 4 1 0 3 2 0 0 "
                              "0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 5D 输入 [8,97,8,494,2]，perm={3,2,1,4,0}，输出 [494,8,97,2,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_32140)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {3, 2, 1, 4, 0};
    gert::TilingContextPara::TensorDescription x({{8, 97, 8, 494, 2}, {8, 97, 8, 494, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{494, 8, 97, 2, 8}, {494, 8, 97, 2, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 416611827713 180388626440 0 0 0 0 0 7904 988 24 0 0 0 0 0 16 1552 "
                              "148992 34359738369 8589934604 0 766688"
                              " 2 1 0 1 12416 8 0 12416 8 1 8589934592 4294967299 4294967295 17179869187 -4294967293 "
                              "-4294967294 8589934594 12884901891 274877907454 108989090103758 5 3 2 1 4 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 输入 [54,9,14,46]，perm={2,1,3,0}，输出 [14,9,46,54]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_2130)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {2, 1, 3, 0};
    gert::TilingContextPara::TensorDescription x({{54, 9, 14, 46}, {54, 9, 14, 46}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{14, 9, 46, 54}, {14, 9, 46, 54}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 38654705671 0 0 0 0 0 0 46368 644 0 0 0 0 0 0 8 2484 "
                              "4294967304 197568495630"
                              " 5796 644 46 1 1 2484 22356 54 22356 2484 54 1 4294967298 3 4294967296 12884901890 "
                              "4294967296 4294967302 4294967296"
                              " 17179869186 270582939649 108989090103297 4 2 1 3 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 4D 输入 [404,128,3,2]，perm={3,0,2,1}，输出 [2,404,3,128]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_3021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {3, 0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{404, 128, 3, 2}, {404, 128, 3, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 404, 3, 128}, {2, 404, 3, 128}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData =
        "4294967297 4294967297 4294967297 867583393793 0 0 0 0 0 0 0 1536 0 0 0 0 0 0 0 768 549755813890 8589934595"
        " 768 6 2 1 384 1 128 155136 155136 384 128 1 3 4294967298 4294967296 12884901890 -4294967296 -4294967294 0"
        " 17179869185 219043332100 108989090103298 4 3 0 2 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 5D 输入 [8,77560,2,4,6]，perm={3,2,1,4,0}，输出 [4,2,77560,6,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_32140_wide)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {3, 2, 1, 4, 0};
    gert::TilingContextPara::TensorDescription x({{8, 77560, 2, 4, 6}, {8, 77560, 2, 4, 6}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{4, 2, 77560, 6, 8}, {4, 2, 77560, 6, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData =
        "4294967297 4294967297 4294967297 8589960446 0 0 0 0 0 0 144 24 0 0 0 0 0 0 144 3722880 12884901896 25769803780"
        " 3722880 48 6 1 1 48 7445760 8 7445760 48 8 1 4294967298 3 4294967296 17179869187 -4294967295 -4294967295 0"
        " 17179869186 274877907752 108989090104100 5 3 2 1 4 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 5D 输入 [2,2875,147,8,4]，perm={2,4,1,0,3}，输出 [147,4,2875,2,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_24103)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {2, 4, 1, 0, 3};
    gert::TilingContextPara::TensorDescription x({{2, 2875, 147, 8, 4}, {2, 2875, 147, 8, 4}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{147, 4, 2875, 2, 8}, {147, 4, 2875, 2, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 4294967297 631360192992 0 0 0 0 0 0 28224 32 0 0 0 0 0 0 96 "
                              "184000 25769803778 17179869192"
                              " 13524000 4704 4 1 8 16 1 46000 46000 16 8 1 4294967299 8589934592 4294967296 "
                              "17179869187 -4294967295 -4294967295"
                              " 0 17179869186 274877908047 108989090104367 5 2 4 1 0 3 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 场景：B4 5D 超大 shape 输入 [8008,77560,2,4,6]，perm={3,2,1,4,0}，输出 [4,2,77560,6,8008]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4_DOUBLE_SPLIT）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_32140_huge)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {3, 2, 1, 4, 0};
    gert::TilingContextPara::TensorDescription x({{8008, 77560, 2, 4, 6}, {8008, 77560, 2, 4, 6}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{4, 2, 77560, 6, 8008}, {4, 2, 77560, 6, 8008}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    string expectTilingData = "4294967297 4294967297 180388626433 8590012152 0 0 0 0 0 714792960 48 24 0 0 0 0 0 192 "
                              "48048 3726602880 824633720833"
                              " 25769803780 0 3722880 6 1 0 1 7453205760 8008 0 7453205760 8008 1 8589934592 "
                              "4294967299 4294967295 17179869187"
                              " -4294967296 -4294967160 0 12884901891 274878008742 108989090205062 5 3 2 1 4 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
