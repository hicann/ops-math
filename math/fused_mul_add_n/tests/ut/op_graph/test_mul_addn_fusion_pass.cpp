/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/mul_addn_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {

TensorDesc MakeDesc(DataType dtype, const vector<int64_t>& dims)
{
    TensorDesc tensorDesc;
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetShape(Shape(dims));
    tensorDesc.SetFormat(FORMAT_ND);
    return tensorDesc;
}

// Build a source-graph Mul node (INPUT x1, INPUT x2, OUTPUT y) wired to two holder inputs.
GNode BuildMulNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input0, const es::EsTensorHolder& input1,
                   const string& nodeName)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode mulNode = es::CompliantNodeBuilder(graph)
                        .OpType("Mul")
                        .Name(nodeName.c_str())
                        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    EXPECT_EQ(es::AddEdgeAndUpdatePeerDesc(*graph, *input0.GetProducer(), input0.GetProducerOutIndex(), mulNode, 0),
              GRAPH_SUCCESS);
    EXPECT_EQ(es::AddEdgeAndUpdatePeerDesc(*graph, *input1.GetProducer(), input1.GetProducerOutIndex(), mulNode, 1),
              GRAPH_SUCCESS);
    return mulNode;
}

// Build a source-graph AddN node (DYNAMIC_INPUT x, OUTPUT y, REQUIRED_ATTR N) wired to holders.
GNode BuildAddNNode(es::EsGraphBuilder& builder, const vector<es::EsTensorHolder>& inputs, const string& nodeName)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    const int64_t inputNum = static_cast<int64_t>(inputs.size());
    GNode addnNode = es::CompliantNodeBuilder(graph)
                         .OpType("AddN")
                         .Name(nodeName.c_str())
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs(
                             {{"N", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(inputNum)}})
                         .InstanceDynamicInputNum("x", static_cast<int32_t>(inputNum))
                         .Build();
    for (int64_t i = 0; i < inputNum; ++i) {
        EXPECT_EQ(es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[static_cast<size_t>(i)].GetProducer(),
                                               inputs[static_cast<size_t>(i)].GetProducerOutIndex(), addnNode,
                                               static_cast<int32_t>(i)),
                  GRAPH_SUCCESS);
    }
    return addnNode;
}

// Build a source-graph Square node as an extra consumer of a tensor (INPUT x, OUTPUT y).
GNode BuildSquareNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input, const string& nodeName)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode squareNode = es::CompliantNodeBuilder(graph)
                           .OpType("Square")
                           .Name(nodeName.c_str())
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
    EXPECT_EQ(es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), squareNode, 0),
              GRAPH_SUCCESS);
    return squareNode;
}

// Build a multi-input source graph: N Muls (x1[i] * x2[i]) summed by one AddN.
shared_ptr<Graph> BuildMultiInputGraph(DataType dtype, const vector<int64_t>& dimsX1, const vector<int64_t>& dimsX2,
                                       const vector<int64_t>& dimsY, int64_t addnInputNum,
                                       es::EsGraphBuilder& graphBuilder)
{
    vector<es::EsTensorHolder> mulOutputs;
    for (int64_t i = 0; i < addnInputNum; ++i) {
        auto x1 = graphBuilder.CreateInput(2 * i, "x1", dtype, FORMAT_ND, dimsX1);
        auto x2 = graphBuilder.CreateInput(2 * i + 1, "x2", dtype, FORMAT_ND, dimsX2);
        string mulNodeName = "mul_" + to_string(i);
        GNode mulNode = BuildMulNode(graphBuilder, x1, x2, mulNodeName);
        TensorDesc x1Desc = MakeDesc(dtype, dimsX1);
        TensorDesc x2Desc = MakeDesc(dtype, dimsX2);
        TensorDesc mulOutDesc = MakeDesc(dtype, dimsY);
        mulNode.UpdateInputDesc(0, x1Desc);
        mulNode.UpdateInputDesc(1, x2Desc);
        mulNode.UpdateOutputDesc(0, mulOutDesc);
        auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
        if (mulOutput == nullptr) {
            return nullptr;
        }
        mulOutputs.emplace_back(mulOutput);
    }
    GNode addnNode = BuildAddNNode(graphBuilder, mulOutputs, "addn");
    TensorDesc yDesc = MakeDesc(dtype, dimsY);
    for (int64_t i = 0; i < addnInputNum; ++i) {
        addnNode.UpdateInputDesc(static_cast<int32_t>(i), yDesc);
    }
    addnNode.UpdateOutputDesc(0, yDesc);
    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    if (addnOutput == nullptr) {
        return nullptr;
    }
    return graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
}
const char* const kArchDav2201 = "2201";
const char* const kArchDav3510 = "3510";

// UT stub for rtGetSocSpec: the UT executable does not link the runtime lib, while the pass
// gates the multi-input path on NpuArch via this API. The stub feeds the arch from the
// ASCEND_NPU_ARCH env variable (same convention as the concat_d UT and runtime_stub.cpp).
// Weak definition: yields to any real one.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

__attribute__((weak)) rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    const char* arch = std::getenv("ASCEND_NPU_ARCH");
    if (arch == nullptr || std::strcmp(label, "version") != 0 || std::strcmp(key, "NpuArch") != 0) {
        return static_cast<rtError_t>(1);
    }
    (void)std::snprintf(val, maxLen, "%s", arch);
    return RT_ERROR_NONE;
}

static void SetNpuArch(const char* arch) { (void)setenv("ASCEND_NPU_ARCH", arch, 1); }
} // namespace

class MulAddNPassTest : public testing::Test {
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
        SetNpuArch(kArchDav2201);
    }

    void SetUp() override
    {
        SetPlatform91093();
        SetNpuArch(kArchDav2201);
    }

    void SetPlatform91093()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetPlatform910B()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetPlatform950()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(MulAddNPassTest, patternTest)
{
    MulAddNPass pass;
    vector<PatternUniqPtr> patterns = pass.Patterns();
    // one two-input pattern plus patterns for AddN input num in [3, 16]
    EXPECT_GE(patterns.size(), 1);
}

// Two-input path: AddN(Mul(x, s), b) --> FusedMulAddN(x, b, s), y = x * s[0] + b.
TEST_F(MulAddNPassTest, twoInputFusionSuccessOn91093)
{
    SetPlatform91093();
    vector<int64_t> dimsX{8, 4};
    vector<int64_t> dimsScalar{1};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto s = graphBuilder.CreateInput(1, "s", DT_FLOAT, FORMAT_ND, dimsScalar);
    auto b = graphBuilder.CreateInput(2, "b", DT_FLOAT, FORMAT_ND, dimsX);

    GNode mulNode = BuildMulNode(graphBuilder, x, s, "mul");
    TensorDesc xDesc = MakeDesc(DT_FLOAT, dimsX);
    TensorDesc sDesc = MakeDesc(DT_FLOAT, dimsScalar);
    TensorDesc mulOutDesc = MakeDesc(DT_FLOAT, dimsX);
    mulNode.UpdateInputDesc(0, xDesc);
    mulNode.UpdateInputDesc(1, sDesc);
    mulNode.UpdateOutputDesc(0, mulOutDesc);

    auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    ASSERT_NE(mulOutput, nullptr);
    GNode addnNode = BuildAddNNode(graphBuilder, {es::EsTensorHolder(mulOutput), b}, "addn");
    TensorDesc bDesc = MakeDesc(DT_FLOAT, dimsX);
    addnNode.UpdateInputDesc(0, mulOutDesc);
    addnNode.UpdateInputDesc(1, bDesc);
    addnNode.UpdateOutputDesc(0, mulOutDesc);

    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    ASSERT_NE(addnOutput, nullptr);
    shared_ptr<Graph> graph = graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "FusedMulAddN") {
            isFound = true;
            EXPECT_EQ(node.GetInputsSize(), 3);
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

// The two-input path has no soc restriction in the legacy pass, keep it on Ascend950.
TEST_F(MulAddNPassTest, twoInputFusionSuccessOn950)
{
    SetPlatform950();
    vector<int64_t> dimsX{16, 32};
    vector<int64_t> dimsScalar{1};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto s = graphBuilder.CreateInput(1, "s", DT_FLOAT, FORMAT_ND, dimsScalar);
    auto b = graphBuilder.CreateInput(2, "b", DT_FLOAT, FORMAT_ND, dimsX);

    GNode mulNode = BuildMulNode(graphBuilder, x, s, "mul");
    TensorDesc xDesc = MakeDesc(DT_FLOAT, dimsX);
    TensorDesc sDesc = MakeDesc(DT_FLOAT, dimsScalar);
    TensorDesc mulOutDesc = MakeDesc(DT_FLOAT, dimsX);
    mulNode.UpdateInputDesc(0, xDesc);
    mulNode.UpdateInputDesc(1, sDesc);
    mulNode.UpdateOutputDesc(0, mulOutDesc);

    auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    ASSERT_NE(mulOutput, nullptr);
    GNode addnNode = BuildAddNNode(graphBuilder, {es::EsTensorHolder(mulOutput), b}, "addn");
    TensorDesc bDesc = MakeDesc(DT_FLOAT, dimsX);
    addnNode.UpdateInputDesc(0, mulOutDesc);
    addnNode.UpdateInputDesc(1, bDesc);
    addnNode.UpdateOutputDesc(0, mulOutDesc);

    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    ASSERT_NE(addnOutput, nullptr);
    shared_ptr<Graph> graph = graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "FusedMulAddN") {
            isFound = true;
            EXPECT_EQ(node.GetInputsSize(), 3);
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

// Neither input of Mul is scalar or shape-(1,), the two-input path must not fuse.
TEST_F(MulAddNPassTest, twoInputScalarSideNotFoundFail)
{
    SetPlatform91093();
    vector<int64_t> dimsX{8, 4};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_ND, dimsX);
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_ND, dimsX);
    auto b = graphBuilder.CreateInput(2, "b", DT_FLOAT, FORMAT_ND, dimsX);

    GNode mulNode = BuildMulNode(graphBuilder, x1, x2, "mul");
    TensorDesc x1Desc = MakeDesc(DT_FLOAT, dimsX);
    TensorDesc x2Desc = MakeDesc(DT_FLOAT, dimsX);
    TensorDesc mulOutDesc = MakeDesc(DT_FLOAT, dimsX);
    mulNode.UpdateInputDesc(0, x1Desc);
    mulNode.UpdateInputDesc(1, x2Desc);
    mulNode.UpdateOutputDesc(0, mulOutDesc);

    auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    ASSERT_NE(mulOutput, nullptr);
    GNode addnNode = BuildAddNNode(graphBuilder, {es::EsTensorHolder(mulOutput), b}, "addn");
    TensorDesc bDesc = MakeDesc(DT_FLOAT, dimsX);
    addnNode.UpdateInputDesc(0, mulOutDesc);
    addnNode.UpdateInputDesc(1, bDesc);
    addnNode.UpdateOutputDesc(0, mulOutDesc);

    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    ASSERT_NE(addnOutput, nullptr);
    shared_ptr<Graph> graph = graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "FusedMulAddN");
    }
}

// Multi-input path: AddN with 3 Mul inputs (all shape-conformant) --> MulAddn with N = 3.
TEST_F(MulAddNPassTest, multiInputFusionSuccessOn91093)
{
    SetPlatform91093();
    SetNpuArch(kArchDav2201);
    vector<int64_t> dimsX1{1500, 512, 1};
    vector<int64_t> dimsX2{1500, 1, 128};
    vector<int64_t> dimsY{1500, 512, 128};

    auto graphBuilder = es::EsGraphBuilder("test");
    shared_ptr<Graph> graph = BuildMultiInputGraph(DT_FLOAT, dimsX1, dimsX2, dimsY, 3, graphBuilder);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "MulAddn") {
            isFound = true;
            EXPECT_EQ(node.GetInputsSize(), 6);
            int64_t attrN = 0;
            EXPECT_EQ(node.GetAttr("N", attrN), GRAPH_SUCCESS);
            EXPECT_EQ(attrN, 3);
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

// Multi-input path also works on Ascend910B.
TEST_F(MulAddNPassTest, multiInputFusionSuccessOn910B)
{
    SetPlatform910B();
    SetNpuArch(kArchDav2201);
    vector<int64_t> dimsX1{8, 16, 1};
    vector<int64_t> dimsX2{8, 1, 32};
    vector<int64_t> dimsY{8, 16, 32};

    auto graphBuilder = es::EsGraphBuilder("test");
    shared_ptr<Graph> graph = BuildMultiInputGraph(DT_FLOAT16, dimsX1, dimsX2, dimsY, 3, graphBuilder);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "MulAddn") {
            isFound = true;
            EXPECT_EQ(node.GetInputsSize(), 6);
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

// The multi-input path is restricted to NpuArch DAV_2201 (Ascend910B / Ascend910_93).
TEST_F(MulAddNPassTest, multiInputPlatformFailOn950)
{
    SetPlatform950();
    SetNpuArch(kArchDav3510);
    vector<int64_t> dimsX1{1500, 512, 1};
    vector<int64_t> dimsX2{1500, 1, 128};
    vector<int64_t> dimsY{1500, 512, 128};

    auto graphBuilder = es::EsGraphBuilder("test");
    shared_ptr<Graph> graph = BuildMultiInputGraph(DT_FLOAT, dimsX1, dimsX2, dimsY, 3, graphBuilder);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "MulAddn");
    }
}

// Unsupported dtype on the multi-input path.
TEST_F(MulAddNPassTest, multiInputDtypeFail)
{
    SetPlatform91093();
    vector<int64_t> dimsX1{1500, 512, 1};
    vector<int64_t> dimsX2{1500, 1, 128};
    vector<int64_t> dimsY{1500, 512, 128};

    auto graphBuilder = es::EsGraphBuilder("test");
    shared_ptr<Graph> graph = BuildMultiInputGraph(DT_INT32, dimsX1, dimsX2, dimsY, 3, graphBuilder);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "MulAddn");
    }
}

// The 3rd dim of x2 exceeds 2040, violating the MulAddn kernel constraint.
TEST_F(MulAddNPassTest, multiInputShapeFail)
{
    SetPlatform91093();
    vector<int64_t> dimsX1{1500, 512, 1};
    vector<int64_t> dimsX2{1500, 1, 3000};
    vector<int64_t> dimsY{1500, 512, 3000};

    auto graphBuilder = es::EsGraphBuilder("test");
    shared_ptr<Graph> graph = BuildMultiInputGraph(DT_FLOAT, dimsX1, dimsX2, dimsY, 3, graphBuilder);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "MulAddn");
    }
}

// The 3rd dim of x1 is not 1, violating the MulAddn kernel constraint.
TEST_F(MulAddNPassTest, multiInputX1DimFail)
{
    SetPlatform91093();
    vector<int64_t> dimsX1{1500, 512, 2};
    vector<int64_t> dimsX2{1500, 1, 128};
    vector<int64_t> dimsY{1500, 512, 256};

    auto graphBuilder = es::EsGraphBuilder("test");
    shared_ptr<Graph> graph = BuildMultiInputGraph(DT_FLOAT, dimsX1, dimsX2, dimsY, 3, graphBuilder);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "MulAddn");
    }
}

// AddN with a non-Mul input matches no pattern of this pass (legacy: NOT_CHANGED).
TEST_F(MulAddNPassTest, mixedAddNInputNotMatched)
{
    SetPlatform91093();
    vector<int64_t> dimsX1{1500, 512, 1};
    vector<int64_t> dimsX2{1500, 1, 128};
    vector<int64_t> dimsY{1500, 512, 128};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_ND, dimsX1);
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_ND, dimsX2);
    GNode mulNode = BuildMulNode(graphBuilder, x1, x2, "mul");
    TensorDesc x1Desc = MakeDesc(DT_FLOAT, dimsX1);
    TensorDesc x2Desc = MakeDesc(DT_FLOAT, dimsX2);
    TensorDesc mulOutDesc = MakeDesc(DT_FLOAT, dimsY);
    mulNode.UpdateInputDesc(0, x1Desc);
    mulNode.UpdateInputDesc(1, x2Desc);
    mulNode.UpdateOutputDesc(0, mulOutDesc);
    auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    ASSERT_NE(mulOutput, nullptr);

    auto other = graphBuilder.CreateInput(2, "other", DT_FLOAT, FORMAT_ND, dimsY);
    GNode addnNode = BuildAddNNode(graphBuilder, {es::EsTensorHolder(mulOutput), other, other}, "addn");
    TensorDesc yDesc = MakeDesc(DT_FLOAT, dimsY);
    addnNode.UpdateInputDesc(0, mulOutDesc);
    addnNode.UpdateInputDesc(1, yDesc);
    addnNode.UpdateInputDesc(2, yDesc);
    addnNode.UpdateOutputDesc(0, yDesc);

    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    ASSERT_NE(addnOutput, nullptr);
    shared_ptr<Graph> graph = graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "MulAddn");
        EXPECT_NE(type, "FusedMulAddN");
    }
}

// The Mul output is also consumed by a Square node, legacy pass requires Mul out-degree == 1.
TEST_F(MulAddNPassTest, twoInputMulMultiConsumerFail)
{
    SetPlatform91093();
    vector<int64_t> dimsX{8, 4};
    vector<int64_t> dimsScalar{1};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto s = graphBuilder.CreateInput(1, "s", DT_FLOAT, FORMAT_ND, dimsScalar);
    auto b = graphBuilder.CreateInput(2, "b", DT_FLOAT, FORMAT_ND, dimsX);

    GNode mulNode = BuildMulNode(graphBuilder, x, s, "mul");
    TensorDesc xDesc = MakeDesc(DT_FLOAT, dimsX);
    TensorDesc sDesc = MakeDesc(DT_FLOAT, dimsScalar);
    TensorDesc mulOutDesc = MakeDesc(DT_FLOAT, dimsX);
    mulNode.UpdateInputDesc(0, xDesc);
    mulNode.UpdateInputDesc(1, sDesc);
    mulNode.UpdateOutputDesc(0, mulOutDesc);
    auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    ASSERT_NE(mulOutput, nullptr);

    GNode squareNode = BuildSquareNode(graphBuilder, es::EsTensorHolder(mulOutput), "square");
    squareNode.UpdateInputDesc(0, mulOutDesc);
    squareNode.UpdateOutputDesc(0, mulOutDesc);
    auto squareOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(squareNode, 0);
    ASSERT_NE(squareOutput, nullptr);

    GNode addnNode = BuildAddNNode(graphBuilder, {es::EsTensorHolder(mulOutput), b}, "addn");
    TensorDesc bDesc = MakeDesc(DT_FLOAT, dimsX);
    addnNode.UpdateInputDesc(0, mulOutDesc);
    addnNode.UpdateInputDesc(1, bDesc);
    addnNode.UpdateOutputDesc(0, mulOutDesc);
    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    ASSERT_NE(addnOutput, nullptr);

    shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {es::EsTensorHolder(addnOutput), es::EsTensorHolder(squareOutput)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "FusedMulAddN");
    }
}

// On the multi-input path, one Mul output is also consumed by a Square node (out-degree == 2).
TEST_F(MulAddNPassTest, multiInputMulMultiConsumerFail)
{
    SetPlatform91093();
    vector<int64_t> dimsX1{1500, 512, 1};
    vector<int64_t> dimsX2{1500, 1, 128};
    vector<int64_t> dimsY{1500, 512, 128};

    auto graphBuilder = es::EsGraphBuilder("test");
    vector<es::EsTensorHolder> mulOutputs;
    for (int64_t i = 0; i < 3; ++i) {
        auto x1 = graphBuilder.CreateInput(2 * i, "x1", DT_FLOAT, FORMAT_ND, dimsX1);
        auto x2 = graphBuilder.CreateInput(2 * i + 1, "x2", DT_FLOAT, FORMAT_ND, dimsX2);
        string mulNodeName = "mul_" + to_string(i);
        GNode mulNode = BuildMulNode(graphBuilder, x1, x2, mulNodeName);
        TensorDesc x1Desc = MakeDesc(DT_FLOAT, dimsX1);
        TensorDesc x2Desc = MakeDesc(DT_FLOAT, dimsX2);
        TensorDesc mulOutDesc = MakeDesc(DT_FLOAT, dimsY);
        mulNode.UpdateInputDesc(0, x1Desc);
        mulNode.UpdateInputDesc(1, x2Desc);
        mulNode.UpdateOutputDesc(0, mulOutDesc);
        auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
        ASSERT_NE(mulOutput, nullptr);
        mulOutputs.emplace_back(mulOutput);
    }

    GNode squareNode = BuildSquareNode(graphBuilder, mulOutputs[0], "square");
    TensorDesc yDesc = MakeDesc(DT_FLOAT, dimsY);
    squareNode.UpdateInputDesc(0, yDesc);
    squareNode.UpdateOutputDesc(0, yDesc);
    auto squareOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(squareNode, 0);
    ASSERT_NE(squareOutput, nullptr);

    GNode addnNode = BuildAddNNode(graphBuilder, mulOutputs, "addn");
    for (int64_t i = 0; i < 3; ++i) {
        addnNode.UpdateInputDesc(static_cast<int32_t>(i), yDesc);
    }
    addnNode.UpdateOutputDesc(0, yDesc);
    auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
    ASSERT_NE(addnOutput, nullptr);

    shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {es::EsTensorHolder(addnOutput), es::EsTensorHolder(squareOutput)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    MulAddNPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        EXPECT_NE(type, "MulAddn");
        EXPECT_NE(type, "FusedMulAddN");
    }
}
