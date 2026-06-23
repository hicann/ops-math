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
#include "../../../../op_host/arch35/transpose_tiling_with_gather_arch35.h"
#include "../../../../op_host/arch35/transpose_tiling_with_021vconv_arch35.h"

using namespace ge;
// using namespace ut_util;

class TransposeTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransposeTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TransposeTiling TearDown" << std::endl;
    }
};

TEST_F(TransposeTiling, transpose_tiling_01)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{31, 36, 203, 26}, {31, 36, 203, 26}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10006;
    string expectTilingData =
        "10006 93802085788320 215891311432040488 16842752 27 27 190008 190008 755914244272 0 0 0 190008 0 0 1080 1 1 1 "
        "1 1 1 1 5456 1 1 1 1 1 1 1 176 1 1 1 1 1 1 1 133143986352 4294967297 4294967297 755914244127 4294967297 "
        "4294967297 16557351567361 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TransposeTiling, transpose_021vconv_float32_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 8, 16}, {2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 8);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 16);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 8);
        std::vector<size_t> expectWorkspaces = {16777216};
        ASSERT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaces.size());
        for (size_t i = 0; i < expectWorkspaces.size(); i++) {
            EXPECT_EQ(tilingInfo.workspaceSizes[i], expectWorkspaces[i]);
        }
    }
}

TEST_F(TransposeTiling, transpose_021vconv_float32_rconv_aligned_w)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 32, 24}, {1, 32, 24}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 24, 32}, {1, 24, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 32);
        EXPECT_EQ(tilingData.get_WLen(), 24);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 24);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_float32_cconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 8, 32}, {1, 8, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 8}, {1, 32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 8);
        EXPECT_EQ(tilingData.get_WLen(), 32);
        EXPECT_EQ(tilingData.get_UseRConv(), false);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 16);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 32);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int32_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 16, 16}, {2, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 16, 16}, {2, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 16);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 16);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 16);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 32, 32}, {2, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 32, 32}, {2, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 32);
        EXPECT_EQ(tilingData.get_WLen(), 32);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 32);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_uint8_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 64, 32}, {1, 64, 32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 64}, {1, 32, 64}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 64);
        EXPECT_EQ(tilingData.get_WLen(), 32);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 64);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 32);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_cconv_fallback)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 16, 32}, {1, 16, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 16}, {1, 32, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success) {
        EXPECT_NE(tilingInfo.tilingKey, 10008);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_cconv_hlen16_wlen128)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{128, 16, 128}, {128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{128, 128, 16}, {128, 128, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 128);
        EXPECT_EQ(tilingData.get_UseRConv(), false);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 128);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_cconv_hlen16_wlen64)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1600, 16, 64}, {1600, 16, 64}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1600, 64, 16}, {1600, 64, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 64);
        EXPECT_EQ(tilingData.get_UseRConv(), false);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 64);
    }
}
