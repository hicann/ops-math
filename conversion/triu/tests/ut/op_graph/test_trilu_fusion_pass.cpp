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
#include "../../../op_graph/fusion_pass/trilu_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "TriluFusionPass";
}

class TriluFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
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

TEST_F(TriluFusionPassTest, patternTest)
{
    TriluFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(TriluFusionPassTest, fusionSuccessUpper0NoK)
{
    std::vector<int64_t> dimsX{10, 10};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto output = es::Trilu(x, nullptr, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    TriluFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool foundTril = false;
    bool foundTriu = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Tril") {
            foundTril = true;
        }
        if (type == "Triu") {
            foundTriu = true;
        }
    }
    EXPECT_TRUE(foundTril || foundTriu);
}

TEST_F(TriluFusionPassTest, fusionSuccessUpper1NoK)
{
    std::vector<int64_t> dimsX{10, 10};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto output = es::Trilu(x, nullptr, 1);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    TriluFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS);
}

TEST_F(TriluFusionPassTest, fusionSuccessFp32)
{
    std::vector<int64_t> dimsX{10, 10};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());

    auto output = es::Trilu(x, nullptr, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    TriluFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS);

    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Tril" || type == "Triu") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(TriluFusionPassTest, fusionSuccess3dShape)
{
    std::vector<int64_t> dimsX{2, 10, 10};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());

    auto output = es::Trilu(x, nullptr, 1);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    TriluFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS);
}

TEST_F(TriluFusionPassTest, unsupportedPlatformFail)
{
    // Set unsupported platform (not in kTriluSupportSocList and not bf16 capable)
    SetPlatform("Ascend310");

    std::vector<int64_t> dimsX{10, 10};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto output = es::Trilu(x, nullptr, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    TriluFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TriluFusionPassTest, fusionSuccessAscend950)
{
    SetPlatform("Ascend950");

    std::vector<int64_t> dimsX{10, 10};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto output = es::Trilu(x, nullptr, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});

    CustomPassContext passContext;
    TriluFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS);

    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Tril" || type == "Triu") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}
