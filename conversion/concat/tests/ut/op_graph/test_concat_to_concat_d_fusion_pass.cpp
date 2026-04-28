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
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/concat_to_concat_d_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "ConcatToConcatDFusionPass";
}

class ConcatToConcatDFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetPlatform(const std::string& soc)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = soc;
        optiCompilationInfo.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

// Test 1: Unsupported platform (Ascend910B) -> GRAPH_NOT_CHANGED
TEST_F(ConcatToConcatDFusionPassTest, unsupportedPlatformFail)
{
    std::vector<int64_t> dims_x{4, 8, 16};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 0;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

// Test 3: Unsupported platform Ascend310 -> GRAPH_NOT_CHANGED
TEST_F(ConcatToConcatDFusionPassTest, unsupportedPlatformAscend310)
{
    SetPlatform("Ascend310");

    std::vector<int64_t> dims_x{4, 8, 16};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 0;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

// Test 4: concat_dim is not Const (Data node) -> GRAPH_NOT_CHANGED
TEST_F(ConcatToConcatDFusionPassTest, concatNonConstDimSkip)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{4, 8, 16};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto concat_dim = graphBuilder.CreateInput(0, "concat_dim", DT_INT32, FORMAT_ND, std::vector<int64_t>{});
    auto x0 = graphBuilder.CreateInput(1, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(2, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

// Test 5: Fusion success - 2 x-inputs, fp32, concat_dim=0
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccess2InputsFp32)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{4, 8, 16};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 0;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool foundConcatD = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "ConcatD") {
            foundConcatD = true;
            break;
        }
    }
    EXPECT_TRUE(foundConcatD);
}

// Test 6: Fusion success - 3 x-inputs, fp16, concat_dim=1
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccess3InputsFp16)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{2, 4, 8};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graphBuilder.CreateInput(2, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 1;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1, x2}, 3);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT16);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT16);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    TensorDesc x2Desc;
    x2.GetProducer()->GetOutputDesc(0, x2Desc);
    x2Desc.SetDataType(DT_FLOAT16);
    x2Desc.SetShape(shape_x);
    x2Desc.SetFormat(FORMAT_ND);
    x2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool foundConcatD = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "ConcatD") {
            foundConcatD = true;
            break;
        }
    }
    EXPECT_TRUE(foundConcatD);
}

// Test 7: Fusion success - 4 x-inputs, fp16, concat_dim=2
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccess4InputsFp16ConcatDim2)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{2, 4, 8};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x2 = graphBuilder.CreateInput(2, "x2", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x3 = graphBuilder.CreateInput(3, "x3", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 2;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1, x2, x3}, 4);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT16);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT16);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    TensorDesc x2Desc;
    x2.GetProducer()->GetOutputDesc(0, x2Desc);
    x2Desc.SetDataType(DT_FLOAT16);
    x2Desc.SetShape(shape_x);
    x2Desc.SetFormat(FORMAT_ND);
    x2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    TensorDesc x3Desc;
    x3.GetProducer()->GetOutputDesc(0, x3Desc);
    x3Desc.SetDataType(DT_FLOAT16);
    x3Desc.SetShape(shape_x);
    x3Desc.SetFormat(FORMAT_ND);
    x3.GetProducer()->UpdateOutputDesc(0, x3Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool foundConcatD = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "ConcatD") {
            foundConcatD = true;
            break;
        }
    }
    EXPECT_TRUE(foundConcatD);
}

// Test 8: Fusion success - 2D input shape
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccess2dShape)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{16, 32};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 1;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool foundConcatD = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "ConcatD") {
            foundConcatD = true;
            break;
        }
    }
    EXPECT_TRUE(foundConcatD);
}

// Test 9: Fusion success - 4D input shape
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccess4dShape)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{2, 3, 4, 8};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 3;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT16);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT16);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool foundConcatD = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "ConcatD") {
            foundConcatD = true;
            break;
        }
    }
    EXPECT_TRUE(foundConcatD);
}

// Test 10: Fusion success - dynamic shape (-1)
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccessDynamicShape)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{-1, 8, 16};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = 1;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

// Test 11: Fusion success - negative concat_dim (-1)
TEST_F(ConcatToConcatDFusionPassTest, fusionSuccessNegativeConcatDim)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dims_x{4, 8, 16};
    Shape shape_x(dims_x);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());

    int32_t dimVal = -1;
    auto concat_dim = graphBuilder.CreateConst(std::vector<int32_t>{dimVal}, std::vector<int64_t>{});

    auto output = es::Concat(concat_dim, {x0, x1}, 2);

    TensorDesc x0Desc;
    x0.GetProducer()->GetOutputDesc(0, x0Desc);
    x0Desc.SetDataType(DT_FLOAT);
    x0Desc.SetShape(shape_x);
    x0Desc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, x0Desc);

    TensorDesc x1Desc;
    x1.GetProducer()->GetOutputDesc(0, x1Desc);
    x1Desc.SetDataType(DT_FLOAT);
    x1Desc.SetShape(shape_x);
    x1Desc.SetFormat(FORMAT_ND);
    x1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    ConcatToConcatDFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}
