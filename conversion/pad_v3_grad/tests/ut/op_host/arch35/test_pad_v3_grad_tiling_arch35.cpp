/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV3GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3GradTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PadV3GradTilingTest TearDown" << std::endl;
    }
};

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_reflect_1)
{
    PadV3GradCompileInfo compileInfo = {0,0,0,0,0,false,"reflect",0,0,0,false,"ascend950"};

    gert::StorageShape xShape = {{60}, {60}};
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    std::vector<int32_t> pad_value = {6, 4};
    gert::StorageShape yShape = {{50}, {50}};

    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value.data()}
        },
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
            gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "1 60 0 0 0 0 0 0 0 50 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 6 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_symmetric_2)
{   
    PadV3GradCompileInfo compileInfo = {0,0,0,0,0,false,"symmetric",0,0,0,false,"ascend950"};

    std::vector<int32_t> paddingsValue = {20, 25, 10, 18};
    gert::StorageShape xShape = {{1284, 1053}, {1284, 1053}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{1239, 1025}, {1239, 1025}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {
          { xShape, ge::DT_FLOAT16, ge::FORMAT_ND },
          { paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data() }
        },
        {
          { yShape, ge::DT_FLOAT16, ge::FORMAT_ND }
        },
        {
            gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
            gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))
        },
        &compileInfo);
    uint64_t expectedTilingKey = 3;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_circular_3)
{   
    PadV3GradCompileInfo compileInfo = {0,0,0,0,0,false,"circular",0,0,0,false,"ascend950"};

    std::vector<int64_t> paddingsValue = {15, 22, 11, 1, 22, 3};
    
    gert::StorageShape xShape = {{146, 7090, 29}, {146, 7090, 29}};
    gert::StorageShape paddingsShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{130,7046,15}, {130,7046,15}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {
          { xShape, ge::DT_BF16, ge::FORMAT_ND },
          { paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data() }
        },
        {
          { yShape, ge::DT_BF16, ge::FORMAT_ND }
        },
        {
            gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
            gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))
        },
        &compileInfo);
    uint64_t expectedTilingKey = 4;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_edge_4)
{   
    PadV3GradCompileInfo compileInfo = {0,0,0,0,0,false,"edge",0,0,0,false,"ascend950"};

    std::vector<int32_t> paddingsValue = {5, 3, 4, 4, 4, 4, 5, 4};
    gert::StorageShape xShape = {{78, 81, 82, 78}, {78, 81, 82, 78}};
    gert::StorageShape paddingsShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{70, 73, 74, 69}, {70, 73, 74, 69}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {
          { xShape, ge::DT_FLOAT16, ge::FORMAT_ND },
          { paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data() }
        },
        {
          { yShape, ge::DT_FLOAT16, ge::FORMAT_ND }
        },
        {
            gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
            gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))
        },
        &compileInfo);
    uint64_t expectedTilingKey = 1;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadV3GradTilingTest, pad_v3_grad_tiling_test_const_5)   
{   
    PadV3GradCompileInfo compileInfo = {0,0,0,0,0,false,"constant",0,0,0,false,"ascend950"};

    std::vector<int32_t> paddingsValue = {24, 14, 6, 17, 6, 7, 15, 17, 5, 6};
    gert::StorageShape yShape = {{4, 45, 42, 21, 10}, {4, 45, 42, 21, 10}};
    gert::StorageShape paddingsShape = {{5, 2}, {5, 2}};
    gert::StorageShape xShape = {{42, 68, 55, 53, 21}, {42, 68, 55, 53, 21}};
    gert::TilingContextPara tilingContextPara(
        "PadV3Grad",
        {
          { xShape, ge::DT_FLOAT, ge::FORMAT_ND },
          { paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data() }
        },
        {
          { yShape, ge::DT_FLOAT, ge::FORMAT_ND }
        },
        {
            gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
            gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))
        },
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}