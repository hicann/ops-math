/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>

#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "es_math_ops.h"
#include "graph/tensor.h"
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/concat_to_concatd_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace ge::fusion;
using namespace ops;

namespace {
const std::string kTypeConcatD = "ConcatD";
const std::string kTypeConcat = "Concat";
const std::string kTypeConcatV2 = "ConcatV2";
const std::string kTypeConst = "Const";
} // namespace

class ConcatToConcatDFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatform("Ascend950", "Ascend950"); }

    void SetUp() override { SetPlatform("Ascend950", "Ascend950"); }

    static void SetPlatform(const std::string& shortSoc, const std::string& socVersion)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = shortSoc;
        optiCompilationInfo.soc_version = socVersion;
        PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static int32_t CountNodesOfType(const std::shared_ptr<Graph>& graphPtr, const std::string& target)
    {
        int32_t count = 0;
        for (auto node : graphPtr->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == target.c_str()) {
                ++count;
            }
        }
        return count;
    }

    static bool HasNodeType(const std::shared_ptr<Graph>& graphPtr, const std::string& target)
    {
        return CountNodesOfType(graphPtr, target) > 0;
    }

    static void SetTensorOutputDesc(const es::EsTensorHolder& tensor, DataType dtype, const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        tensor.GetProducer()->GetOutputDesc(0, desc);
        desc.SetDataType(dtype);
        desc.SetShape(Shape(dims));
        desc.SetFormat(FORMAT_ND);
        tensor.GetProducer()->UpdateOutputDesc(0, desc);
    }

    // Build Concat: concat_dim is a scalar int64 Const at input[0], followed by data inputs.
    static std::shared_ptr<Graph> BuildConcatGraph(int64_t concatDim, const std::vector<std::vector<int64_t>>& dimsVec,
                                                   DataType dtype)
    {
        auto graphBuilder = es::EsGraphBuilder("concat_test_graph");
        auto concatDimTensor = graphBuilder.CreateScalar(concatDim);
        SetTensorOutputDesc(concatDimTensor, DT_INT64, {});

        std::vector<es::EsTensorHolder> xVec;
        for (size_t i = 0; i < dimsVec.size(); ++i) {
            auto x = graphBuilder.CreateInput(static_cast<int32_t>(i), "x", dtype, FORMAT_ND, dimsVec[i]);
            SetTensorOutputDesc(x, dtype, dimsVec[i]);
            xVec.emplace_back(x);
        }
        auto y = es::Concat(concatDimTensor, xVec, static_cast<int64_t>(dimsVec.size()));
        GNode concatNode = *y.GetProducer();
        // es API connects edges without propagating descs; real graphs get input descs
        // filled by graph normalization. Fill them explicitly so the pass can read
        // shapes (needed e.g. for zero-dim pruning).
        TensorDesc axisDesc;
        concatDimTensor.GetProducer()->GetOutputDesc(0, axisDesc);
        (void)concatNode.UpdateInputDesc(0, axisDesc);
        for (size_t i = 0; i < dimsVec.size(); ++i) {
            TensorDesc xDesc;
            xVec[i].GetProducer()->GetOutputDesc(0, xDesc);
            (void)concatNode.UpdateInputDesc(static_cast<int32_t>(i + 1), xDesc);
        }
        return graphBuilder.BuildAndReset({y});
    }

    // Build ConcatV2: data inputs first, concat_dim is a scalar int32 Const at the last input.
    static std::shared_ptr<Graph> BuildConcatV2Graph(int32_t concatDim,
                                                     const std::vector<std::vector<int64_t>>& dimsVec, DataType dtype)
    {
        auto graphBuilder = es::EsGraphBuilder("concat_v2_test_graph");
        std::vector<es::EsTensorHolder> xVec;
        for (size_t i = 0; i < dimsVec.size(); ++i) {
            auto x = graphBuilder.CreateInput(static_cast<int32_t>(i), "x", dtype, FORMAT_ND, dimsVec[i]);
            SetTensorOutputDesc(x, dtype, dimsVec[i]);
            xVec.emplace_back(x);
        }
        auto concatDimTensor = graphBuilder.CreateScalar(concatDim);
        SetTensorOutputDesc(concatDimTensor, DT_INT32, {});
        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto concatV2 = es::CompliantNodeBuilder(graph)
                            .OpType("ConcatV2")
                            .Name("concat_v2")
                            .IrDefInputs({
                                {"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                                {"concat_dim", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                            })
                            .InstanceDynamicInputNum("x", static_cast<int32_t>(dimsVec.size()))
                            .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .IrDefAttrs({
                                {"N", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                 es::CreateFrom(static_cast<int64_t>(dimsVec.size()))},
                            })
                            .Build();
        for (size_t i = 0; i < dimsVec.size(); ++i) {
            es::AddEdgeAndUpdatePeerDesc(*graph, *xVec[i].GetProducer(), xVec[i].GetProducerOutIndex(), concatV2,
                                         static_cast<int32_t>(i));
        }
        es::AddEdgeAndUpdatePeerDesc(*graph, *concatDimTensor.GetProducer(), concatDimTensor.GetProducerOutIndex(),
                                     concatV2, static_cast<int32_t>(dimsVec.size()));
        TensorDesc axisDesc;
        concatDimTensor.GetProducer()->GetOutputDesc(0, axisDesc);
        (void)concatV2.UpdateInputDesc(static_cast<int32_t>(dimsVec.size()), axisDesc);
        for (size_t i = 0; i < dimsVec.size(); ++i) {
            TensorDesc xDesc;
            xVec[i].GetProducer()->GetOutputDesc(0, xDesc);
            (void)concatV2.UpdateInputDesc(static_cast<int32_t>(i), xDesc);
        }
        auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatV2, 0);
        return graphBuilder.BuildAndReset({y});
    }
};

TEST_F(ConcatToConcatDFusionPassTest, instantiationTest)
{
    ConcatToConcatDFusionPass pass;
    SUCCEED();
}

TEST_F(ConcatToConcatDFusionPassTest, concatFusionSuccess)
{
    std::vector<std::vector<int64_t>> dimsVec{{2, 3}, {2, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    EXPECT_FALSE(HasNodeType(graphPtr, kTypeConcat));
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            int64_t concatDim = 0;
            EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
            EXPECT_EQ(concatDim, 1);
            int64_t nAttr = 0;
            EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
            EXPECT_EQ(nAttr, 2);
            EXPECT_EQ(node.GetInputsSize(), 2);
            // The two data inputs must come from two different producers (the first
            // migration attempt lost one input and duplicated the other).
            auto producer0 = node.GetInDataNodesAndPortIndexs(0).first;
            auto producer1 = node.GetInDataNodesAndPortIndexs(1).first;
            ASSERT_NE(producer0, nullptr);
            ASSERT_NE(producer1, nullptr);
            EXPECT_NE(producer0.get(), producer1.get());
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, concatV2FusionSuccess)
{
    std::vector<std::vector<int64_t>> dimsVec{{2, 3}, {2, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatV2Graph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    EXPECT_FALSE(HasNodeType(graphPtr, kTypeConcatV2));
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            int64_t concatDim = 0;
            EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
            EXPECT_EQ(concatDim, 1);
            int64_t nAttr = 0;
            EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
            EXPECT_EQ(nAttr, 2);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, concatV2DFusionSuccess)
{
    const int64_t inputNum = 2;
    std::vector<int64_t> dimsX{2, 3};

    auto graphBuilder = es::EsGraphBuilder("concat_v2d_test_graph");
    std::vector<es::EsTensorHolder> xVec;
    for (int64_t i = 0; i < inputNum; ++i) {
        auto x = graphBuilder.CreateInput(static_cast<int32_t>(i), "x", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVec.emplace_back(x);
    }
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto concatV2D = es::CompliantNodeBuilder(graph)
                         .OpType("ConcatV2D")
                         .Name("concat_v2d")
                         .IrDefInputs({
                             {"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                         })
                         .InstanceDynamicInputNum("x", static_cast<int32_t>(inputNum))
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({
                             {"concat_dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                              es::CreateFrom(static_cast<int64_t>(1))},
                             {"N", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                              es::CreateFrom(static_cast<int64_t>(inputNum))},
                         })
                         .Build();
    for (int64_t i = 0; i < inputNum; ++i) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *xVec[i].GetProducer(), xVec[i].GetProducerOutIndex(), concatV2D,
                                     static_cast<int32_t>(i));
    }
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatV2D, 0);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    EXPECT_FALSE(HasNodeType(graphPtr, "ConcatV2D"));
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            int64_t concatDim = 0;
            EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
            EXPECT_EQ(concatDim, 1);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, concatNonConstDimNotFused)
{
    // concat_dim comes from a graph input (Data node), pattern must not match, graph unchanged.
    std::vector<int64_t> dimsX{2, 3};
    auto graphBuilder = es::EsGraphBuilder("concat_non_const_graph");
    auto concatDimInput = graphBuilder.CreateInput(0, "concat_dim", DT_INT64, FORMAT_ND, {});
    SetTensorOutputDesc(concatDimInput, DT_INT64, {});
    std::vector<es::EsTensorHolder> xVec;
    for (size_t i = 0; i < 2; ++i) {
        auto x = graphBuilder.CreateInput(static_cast<int32_t>(i + 1), "x", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVec.emplace_back(x);
    }
    auto y = es::Concat(concatDimInput, xVec, 2);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graphPtr, kTypeConcatD));
    EXPECT_TRUE(HasNodeType(graphPtr, kTypeConcat));
}

TEST_F(ConcatToConcatDFusionPassTest, sharedAxisConstFusionSuccess)
{
    // One axis Const feeding two Concat nodes: both must be fused correctly and the graph
    // must stay intact (this is the scenario that graph-broke in the first migration
    // attempt: inputs were grafted off-by-one and one data input got lost).
    std::vector<int64_t> dimsX{2, 3};
    auto graphBuilder = es::EsGraphBuilder("concat_shared_const_graph");
    auto concatDimTensor = graphBuilder.CreateScalar(static_cast<int64_t>(1));
    SetTensorOutputDesc(concatDimTensor, DT_INT64, {});

    std::vector<es::EsTensorHolder> outputs;
    std::vector<es::EsTensorHolder> xVecA;
    for (size_t i = 0; i < 2; ++i) {
        auto x = graphBuilder.CreateInput(static_cast<int32_t>(i), "xa", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVecA.emplace_back(x);
    }
    outputs.emplace_back(es::Concat(concatDimTensor, xVecA, 2));

    std::vector<es::EsTensorHolder> xVecB;
    for (size_t i = 0; i < 2; ++i) {
        auto x = graphBuilder.CreateInput(static_cast<int32_t>(i + 2), "xb", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVecB.emplace_back(x);
    }
    outputs.emplace_back(es::Concat(concatDimTensor, xVecB, 2));

    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    // Both Concat nodes are fused, no Concat left, and both ConcatD nodes keep exactly
    // two data inputs (no input lost / duplicated).
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 2);
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcat), 0);
    int32_t concatDChecked = 0;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            EXPECT_EQ(node.GetInputsSize(), 2);
            int64_t nAttr = 0;
            EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
            EXPECT_EQ(nAttr, 2);
            int64_t concatDim = 0;
            EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
            EXPECT_EQ(concatDim, 1);
            ++concatDChecked;
        }
    }
    EXPECT_EQ(concatDChecked, 2);
}

TEST_F(ConcatToConcatDFusionPassTest, negativeAxisFusionSuccess)
{
    std::vector<std::vector<int64_t>> dimsVec{{2, 3}, {2, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(-1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            int64_t concatDim = 0;
            EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
            EXPECT_EQ(concatDim, -1);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, thirtyInputsFusionSuccess)
{
    constexpr int32_t inputNum = 30;
    std::vector<std::vector<int64_t>> dimsVec;
    for (int32_t i = 0; i < inputNum; ++i) {
        dimsVec.push_back({2, static_cast<int64_t>(i + 1)});
    }
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT16);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            int64_t nAttr = 0;
            EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
            EXPECT_EQ(nAttr, inputNum);
            EXPECT_EQ(node.GetInputsSize(), inputNum);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, pruneZeroDimInputsFused)
{
    // RemoveInvalidEdge equivalent: data inputs whose shape contains a zero dim are
    // pruned (Concat/ConcatV2 only). Here x2 (2,0,3) is pruned, ConcatD keeps N=1.
    std::vector<std::vector<int64_t>> dimsVec{{2, 3}, {2, 0, 3}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            EXPECT_EQ(node.GetInputsSize(), 1);
            // es::ConcatD omits the optional N attr when it equals the IR default (1);
            // a missing N is therefore equivalent to N=1.
            int64_t nAttr = 0;
            if (node.GetAttr("N", nAttr) == GRAPH_SUCCESS) {
                EXPECT_EQ(nAttr, 1);
            }
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, allZeroDimInputsKeepAll)
{
    // When every data input contains a zero dim, the original RemoveInvalidEdge keeps
    // all inputs (num_n == num_n_del case).
    std::vector<std::vector<int64_t>> dimsVec{{2, 0, 3}, {2, 0, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            EXPECT_EQ(node.GetInputsSize(), 2);
            int64_t nAttr = 0;
            EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
            EXPECT_EQ(nAttr, 2);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, concatV2DZeroDimNoPrune)
{
    // ConcatV2D never prunes zero-dim inputs (original pass has no RemoveInvalidEdge on
    // the ConcatV2D branch).
    const int64_t inputNum = 2;
    std::vector<int64_t> dimsX{2, 0, 3};

    auto graphBuilder = es::EsGraphBuilder("concat_v2d_zerodim_graph");
    std::vector<es::EsTensorHolder> xVec;
    for (int64_t i = 0; i < inputNum; ++i) {
        auto x = graphBuilder.CreateInput(static_cast<int32_t>(i), "x", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVec.emplace_back(x);
    }
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto concatV2D = es::CompliantNodeBuilder(graph)
                         .OpType("ConcatV2D")
                         .Name("concat_v2d")
                         .IrDefInputs({
                             {"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                         })
                         .InstanceDynamicInputNum("x", static_cast<int32_t>(inputNum))
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({
                             {"concat_dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                              es::CreateFrom(static_cast<int64_t>(1))},
                             {"N", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                              es::CreateFrom(static_cast<int64_t>(inputNum))},
                         })
                         .Build();
    for (int64_t i = 0; i < inputNum; ++i) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *xVec[i].GetProducer(), xVec[i].GetProducerOutIndex(), concatV2D,
                                     static_cast<int32_t>(i));
    }
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatV2D, 0);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            EXPECT_EQ(node.GetInputsSize(), 2);
            int64_t nAttr = 0;
            EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
            EXPECT_EQ(nAttr, 2);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, over512InputsSplit)
{
    // SplitConcatNode equivalent: 520 data inputs are split into a ConcatD tree:
    // two children (512 + 8 inputs) plus one parent aggregating them.
    constexpr int32_t inputNum = 520;
    std::vector<std::vector<int64_t>> dimsVec;
    for (int32_t i = 0; i < inputNum; ++i) {
        dimsVec.push_back({2, static_cast<int64_t>(i % 100 + 1)});
    }
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    // 520 inputs -> ceil(520/512) = 2 children + 1 parent = 3 ConcatD nodes.
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 3);

    int32_t childChecked = 0;
    int32_t parentChecked = 0;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type != kTypeConcatD.c_str()) {
            continue;
        }
        int64_t nAttr = 0;
        EXPECT_EQ(node.GetAttr("N", nAttr), GRAPH_SUCCESS);
        int64_t concatDim = 0;
        EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
        EXPECT_EQ(concatDim, 1);
        if (node.GetInputsSize() == 2) {
            // parent: aggregates the two children outputs
            EXPECT_EQ(nAttr, 2);
            ++parentChecked;
        } else {
            // children: 512 and 8 boundary data inputs respectively
            EXPECT_TRUE(node.GetInputsSize() == 512 || node.GetInputsSize() == 8);
            ++childChecked;
        }
    }
    EXPECT_EQ(parentChecked, 1);
    EXPECT_EQ(childChecked, 2);
}

TEST_F(ConcatToConcatDFusionPassTest, noConcatNodeNotChanged)
{
    // Path: graph without any Concat-family node -> GRAPH_NOT_CHANGED.
    auto graphBuilder = es::EsGraphBuilder("no_concat_graph");
    std::vector<int64_t> dimsX{2, 3};
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_ND, dimsX);
    SetTensorOutputDesc(x1, DT_FLOAT, dimsX);
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_ND, dimsX);
    SetTensorOutputDesc(x2, DT_FLOAT, dimsX);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({x1, x2});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graphPtr, kTypeConcatD));
}

TEST_F(ConcatToConcatDFusionPassTest, concatAxisFloatConstNotFused)
{
    // Path: concat_dim const dtype is float (only DT_INT32/DT_INT64 accepted) -> skip.
    auto graphBuilder = es::EsGraphBuilder("concat_float_axis_graph");
    auto concatDim = graphBuilder.CreateScalar(static_cast<float>(1.0f));
    SetTensorOutputDesc(concatDim, DT_FLOAT, {});
    std::vector<int64_t> dimsX{2, 3};
    std::vector<es::EsTensorHolder> xVec;
    for (int32_t i = 0; i < 2; ++i) {
        auto x = graphBuilder.CreateInput(i, "x", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVec.emplace_back(x);
    }
    auto y = es::Concat(concatDim, xVec, 2);
    GNode concatNode = *y.GetProducer();
    TensorDesc axisDesc;
    concatDim.GetProducer()->GetOutputDesc(0, axisDesc);
    (void)concatNode.UpdateInputDesc(0, axisDesc);
    for (int32_t i = 0; i < 2; ++i) {
        TensorDesc xDesc;
        xVec[i].GetProducer()->GetOutputDesc(0, xDesc);
        (void)concatNode.UpdateInputDesc(i + 1, xDesc);
    }
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graphPtr, kTypeConcat));
}

TEST_F(ConcatToConcatDFusionPassTest, concatV2DNoAttrNotFused)
{
    // Path: ConcatV2D without concat_dim attr -> skip node.
    const int64_t inputNum = 2;
    std::vector<int64_t> dimsX{2, 3};
    auto graphBuilder = es::EsGraphBuilder("concat_v2d_noattr_graph");
    std::vector<es::EsTensorHolder> xVec;
    for (int64_t i = 0; i < inputNum; ++i) {
        auto x = graphBuilder.CreateInput(static_cast<int32_t>(i), "x", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVec.emplace_back(x);
    }
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto concatV2D = es::CompliantNodeBuilder(graph)
                         .OpType("ConcatV2D")
                         .Name("concat_v2d")
                         .IrDefInputs({
                             {"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                         })
                         .InstanceDynamicInputNum("x", static_cast<int32_t>(inputNum))
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({
                             {"N", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                              es::CreateFrom(static_cast<int64_t>(inputNum))},
                         })
                         .Build();
    for (int64_t i = 0; i < inputNum; ++i) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *xVec[i].GetProducer(), xVec[i].GetProducerOutIndex(), concatV2D,
                                     static_cast<int32_t>(i));
    }
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatV2D, 0);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graphPtr, "ConcatV2D"));
}

TEST_F(ConcatToConcatDFusionPassTest, concatOnlyAxisNoDataNotFused)
{
    // Path: Concat with only the axis input and no data input -> skip node.
    auto graphBuilder = es::EsGraphBuilder("concat_axis_only_graph");
    auto concatDim = graphBuilder.CreateScalar(static_cast<int64_t>(1));
    SetTensorOutputDesc(concatDim, DT_INT64, {});
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto concatNode = es::CompliantNodeBuilder(graph)
                          .OpType("Concat")
                          .Name("concat")
                          .IrDefInputs({
                              {"concat_dim", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          })
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *concatDim.GetProducer(), concatDim.GetProducerOutIndex(), concatNode, 0);
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatNode, 0);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graphPtr, kTypeConcat));
}

TEST_F(ConcatToConcatDFusionPassTest, mixedTypesOneGraphFusionSuccess)
{
    // Path: Concat + ConcatV2 + ConcatV2D in one graph, all fused independently.
    auto graphBuilder = es::EsGraphBuilder("mixed_types_graph");
    std::vector<int64_t> dimsX{2, 3};
    std::vector<es::EsTensorHolder> outputs;

    {
        auto concatDim = graphBuilder.CreateScalar(static_cast<int64_t>(1));
        SetTensorOutputDesc(concatDim, DT_INT64, {});
        std::vector<es::EsTensorHolder> xVec;
        for (int32_t i = 0; i < 2; ++i) {
            auto x = graphBuilder.CreateInput(i, "xa", DT_FLOAT, FORMAT_ND, dimsX);
            SetTensorOutputDesc(x, DT_FLOAT, dimsX);
            xVec.emplace_back(x);
        }
        outputs.emplace_back(es::Concat(concatDim, xVec, 2));
    }
    {
        std::vector<es::EsTensorHolder> xVec;
        for (int32_t i = 0; i < 2; ++i) {
            auto x = graphBuilder.CreateInput(2 + i, "xb", DT_FLOAT, FORMAT_ND, dimsX);
            SetTensorOutputDesc(x, DT_FLOAT, dimsX);
            xVec.emplace_back(x);
        }
        auto concatDim = graphBuilder.CreateScalar(static_cast<int32_t>(1));
        SetTensorOutputDesc(concatDim, DT_INT32, {});
        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto concatV2 = es::CompliantNodeBuilder(graph)
                            .OpType("ConcatV2")
                            .Name("concat_v2")
                            .IrDefInputs({
                                {"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                                {"concat_dim", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                            })
                            .InstanceDynamicInputNum("x", 2)
                            .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .IrDefAttrs({
                                {"N", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                 es::CreateFrom(static_cast<int64_t>(2))},
                            })
                            .Build();
        for (int32_t i = 0; i < 2; ++i) {
            es::AddEdgeAndUpdatePeerDesc(*graph, *xVec[i].GetProducer(), xVec[i].GetProducerOutIndex(), concatV2, i);
        }
        es::AddEdgeAndUpdatePeerDesc(*graph, *concatDim.GetProducer(), concatDim.GetProducerOutIndex(), concatV2, 2);
        outputs.emplace_back(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatV2, 0));
    }
    {
        std::vector<es::EsTensorHolder> xVec;
        for (int32_t i = 0; i < 2; ++i) {
            auto x = graphBuilder.CreateInput(4 + i, "xc", DT_FLOAT, FORMAT_ND, dimsX);
            SetTensorOutputDesc(x, DT_FLOAT, dimsX);
            xVec.emplace_back(x);
        }
        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto concatV2D = es::CompliantNodeBuilder(graph)
                             .OpType("ConcatV2D")
                             .Name("concat_v2d")
                             .IrDefInputs({
                                 {"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                             })
                             .InstanceDynamicInputNum("x", 2)
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs({
                                 {"concat_dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                  es::CreateFrom(static_cast<int64_t>(1))},
                                 {"N", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                  es::CreateFrom(static_cast<int64_t>(2))},
                             })
                             .Build();
        for (int32_t i = 0; i < 2; ++i) {
            es::AddEdgeAndUpdatePeerDesc(*graph, *xVec[i].GetProducer(), xVec[i].GetProducerOutIndex(), concatV2D, i);
        }
        outputs.emplace_back(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatV2D, 0));
    }
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 3);
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcat), 0);
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatV2), 0);
    EXPECT_EQ(CountNodesOfType(graphPtr, "ConcatV2D"), 0);
}

TEST_F(ConcatToConcatDFusionPassTest, exactly512InputsNoSplit)
{
    // Path: exactly kMaxConcatDInputs(512) data inputs -> single ConcatD, no split.
    constexpr int32_t inputNum = 512;
    std::vector<std::vector<int64_t>> dimsVec;
    for (int32_t i = 0; i < inputNum; ++i) {
        dimsVec.push_back({2, static_cast<int64_t>(i % 100 + 1)});
    }
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            EXPECT_EQ(node.GetInputsSize(), inputNum);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, five13InputsSplit)
{
    // Path: kMaxConcatDInputs+1 (513) data inputs -> split into 512+1 children + 1 parent.
    constexpr int32_t inputNum = 513;
    std::vector<std::vector<int64_t>> dimsVec;
    for (int32_t i = 0; i < inputNum; ++i) {
        dimsVec.push_back({2, static_cast<int64_t>(i % 100 + 1)});
    }
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 3);
    int32_t parentChecked = 0;
    int32_t childChecked = 0;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type != kTypeConcatD.c_str()) {
            continue;
        }
        if (node.GetInputsSize() == 2) {
            ++parentChecked;
        } else {
            EXPECT_TRUE(node.GetInputsSize() == 512 || node.GetInputsSize() == 1);
            ++childChecked;
        }
    }
    EXPECT_EQ(parentChecked, 1);
    EXPECT_EQ(childChecked, 2);
}

TEST_F(ConcatToConcatDFusionPassTest, ascend910_95PlatformNotFused)
{
    // Path: non-Ascend950 platform -> graph not changed.
    SetPlatform("Ascend910_95", "Ascend910_95");
    std::vector<std::vector<int64_t>> dimsVec{{2, 3}, {2, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graphPtr, kTypeConcatD));
    EXPECT_TRUE(HasNodeType(graphPtr, kTypeConcat));
}

TEST_F(ConcatToConcatDFusionPassTest, concatAxisInt32FusionSuccess)
{
    // Path: Concat with an int32 const axis (dtype branch DT_INT32).
    auto graphBuilder = es::EsGraphBuilder("concat_i32_axis_graph");
    auto concatDim = graphBuilder.CreateScalar(static_cast<int32_t>(1));
    SetTensorOutputDesc(concatDim, DT_INT32, {});
    std::vector<int64_t> dimsX{2, 3};
    std::vector<es::EsTensorHolder> xVec;
    for (int32_t i = 0; i < 2; ++i) {
        auto x = graphBuilder.CreateInput(i, "x", DT_FLOAT, FORMAT_ND, dimsX);
        SetTensorOutputDesc(x, DT_FLOAT, dimsX);
        xVec.emplace_back(x);
    }
    auto y = es::Concat(concatDim, xVec, 2);
    GNode concatNode = *y.GetProducer();
    TensorDesc axisDesc;
    concatDim.GetProducer()->GetOutputDesc(0, axisDesc);
    (void)concatNode.UpdateInputDesc(0, axisDesc);
    for (int32_t i = 0; i < 2; ++i) {
        TensorDesc xDesc;
        xVec[i].GetProducer()->GetOutputDesc(0, xDesc);
        (void)concatNode.UpdateInputDesc(i + 1, xDesc);
    }
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({y});

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeConcatD.c_str()) {
            int64_t concatDim = 0;
            EXPECT_EQ(node.GetAttr("concat_dim", concatDim), GRAPH_SUCCESS);
            EXPECT_EQ(concatDim, 1);
        }
    }
}

TEST_F(ConcatToConcatDFusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910B", "Ascend910B1");
    std::vector<std::vector<int64_t>> dimsVec{{2, 3}, {2, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(1, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graphPtr, kTypeConcatD));
    EXPECT_TRUE(HasNodeType(graphPtr, kTypeConcat));
}

TEST_F(ConcatToConcatDFusionPassTest, ascend950FusionSuccess)
{
    SetPlatform("Ascend950", "Ascend950");
    std::vector<std::vector<int64_t>> dimsVec{{2, 3, 4}, {2, 3, 5}};
    std::shared_ptr<Graph> graphPtr = BuildConcatGraph(2, dimsVec, DT_FLOAT);

    ConcatToConcatDFusionPass pass;
    CustomPassContext passContext;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConcatD), 1);
}
