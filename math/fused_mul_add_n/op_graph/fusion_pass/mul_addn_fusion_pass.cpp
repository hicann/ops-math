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
 * \file mul_addn_fusion_pass.cpp
 * \brief mul addn fusion pass (mul + addn --> FusedMulAddN / MulAddn)
 * Migrated from canndev ops/built-in/fusion_pass/graph_fusion/ai_core/mul_addn_fusion_pass.cc
 *
 * Fusion paths (dispatched by the actual connected input num of the matched AddN node):
 *  - two-input AddN:  AddN(Mul(x1, x3), x2) --> FusedMulAddN(x1, x2, x3), y = x1 * x3[0] + x2,
 *    where x3 is the scalar-or-(1,) shaped side of Mul. No soc restriction, same as legacy pass.
 *  - multi-input AddN (all inputs are Mul, N in (2, 16]): AddN(Mul(x1_0, x2_0), ..., Mul(x1_n, x2_n))
 *    --> MulAddn(x1[0..n], x2[0..n], attr N = n), soc restricted to Ascend910B / Ascend910_93.
 *
 * All source and target nodes (Mul / AddN / FusedMulAddN / MulAddn) are built with
 * CompliantNodeBuilder: the CANN built-in ops have no in-process ES API under the single-op
 * build scope, and MulAddn deliberately avoids its generated es API so that the pass stays
 * self-contained when the repo is built with a single-op scope (its proto lives in another
 * operator directory).
 */

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/soc_spec.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "mul_addn_fusion_pass.h"

using namespace ge;

// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
// Strategy: compile-time macro guard + runtime version check + overall silence.
#define GE_COMPILER_VERSION_900 90000000
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

// Weak declare aclsysGetVersionNum to avoid hard link dependency on libascendcl; declare
// rtGetSocSpec for NpuArch-based platform gating, resolved from the runtime library.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

namespace {
CustomPassStage GetMulAddNPassStage()
{
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}

// All interfaces used by this pass (CustomPassStage::kCompatibleInherited, EsGraphBuilder,
// CompliantNodeBuilder dynamic input) are available since GE 9.0.0, so the
// runtime guard is 9.0.0 instead of the 9.1.0 guard used by some other migrated passes.
bool IsSupportedRuntimeVersion()
{
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    return version >= GE_COMPILER_VERSION_900;
}
} // namespace

namespace ge::fusion {

namespace {
const std::string kPassName = "MulAddNPass";
const std::string kStreamLabelAttrName = "_stream_label";
const std::string kAddNOpType = "AddN";
const std::string kMulOpType = "Mul";
const std::string kFusedMulAddNOpType = "FusedMulAddN";
const std::string kMulAddnOpType = "MulAddn";

// capture index 0 is the matched AddN output tensor, capture index (1 + i) is the i-th matched Mul
// output tensor, enumerated in the AddN input order.
constexpr size_t kCaptureIdxAddNNode = 0;
constexpr size_t kCaptureIdxFirstMulNode = 1;

// legacy pass: AddN with exactly 2 connected inputs goes to the FusedMulAddN path,
// AddN with more than 2 inputs (all Mul) goes to the MulAddn path.
constexpr int64_t kTwoInputAddNNum = 2;
// patterns for the multi-input path cover AddN input num in [3, kMaxMultiInputAddNNum].
// The legacy pass was unbounded; a practical upper bound keeps the pattern set finite.
constexpr int64_t kMaxMultiInputAddNNum = 16;
// legacy shape constraint of the MulAddn kernel: the 3rd dim of x2 must be <= 2040.
constexpr int64_t kX2ThirdDimLimit = 2040;
constexpr size_t kMulInputNum = 2;
// AddN's second input index on the two-input path (the non-Mul side).
constexpr int32_t kAddNOtherInputIdx = 1;

const std::vector<DataType> kSupportedDtypes = {DT_FLOAT, DT_FLOAT16, DT_BF16};

std::vector<int64_t> GetShapeDims(const Shape& shape)
{
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

bool CheckDtype(DataType dtype)
{
    return std::find(kSupportedDtypes.begin(), kSupportedDtypes.end(), dtype) != kSupportedDtypes.end();
}

// legacy CheckLossScaleInput requires the chosen side of Mul to be scalar or shape-(1,).
bool IsScalarOrOneElement(const Shape& shape)
{
    const std::vector<int64_t> dims = shape.GetDims();
    return dims.empty() || (dims.size() == 1 && dims[0] == 1);
}

// Pick the scalar side of the Mul node: prefer input 0, then input 1. The legacy pass preferred
// the Constant-input side and fell back to input 0; the new framework cannot traverse the producer
// node type from the matched node, so the side is picked by shape only. When both sides qualify the
// two mappings are mathematically equivalent, and the only divergence from the legacy pass is that
// graphs whose input-1 alone qualifies are now fused (semantically still y = x1 * x3[0] + x2).
int32_t GetScalarSideIdx(const GNode& mulNode)
{
    for (int32_t i = 0; i < static_cast<int32_t>(kMulInputNum); ++i) {
        TensorDesc inputDesc;
        if (mulNode.GetInputDesc(i, inputDesc) != GRAPH_SUCCESS) {
            return -1;
        }
        if (IsScalarOrOneElement(inputDesc.GetShape())) {
            return i;
        }
    }
    return -1;
}

bool GetStreamLabel(const GNode& node, std::string& streamLabel)
{
    AscendString label;
    if (node.GetAttr(kStreamLabelAttrName.c_str(), label) != GRAPH_SUCCESS) {
        streamLabel = "";
        return true;
    }
    streamLabel = std::string(label.GetString());
    return true;
}

std::string GetNodeName(const GNode& node)
{
    AscendString name;
    if (node.GetName(name) != GRAPH_SUCCESS) {
        return "";
    }
    return std::string(name.GetString());
}

Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

// Locate the boundary input slot whose tensor is consumed by (nodeName, inputIdx) inside the matched
// subgraph. One boundary tensor may be consumed by multiple node inputs (dedup), and one node input
// maps to exactly one boundary slot (or none when the input comes from a producer inside the subgraph).
int64_t FindBoundarySlot(const std::vector<SubgraphInput>& subgraphInputs, const std::string& nodeName,
                         int32_t inputIdx)
{
    for (size_t slot = 0; slot < subgraphInputs.size(); ++slot) {
        for (const auto& nodeIo : subgraphInputs[slot].GetAllInputs()) {
            if (nodeIo.index == inputIdx && GetNodeName(nodeIo.node) == nodeName) {
                return static_cast<int64_t>(slot);
            }
        }
    }
    return -1;
}

// Create replacement graph inputs, one per subgraph boundary input, in boundary order. The i-th
// CreateInput call corresponds to the i-th boundary input of the matched subgraph.
std::vector<es::EsTensorHolder> CreateBoundaryInputs(es::EsGraphBuilder& builder,
                                                     const std::vector<SubgraphInput>& subgraphInputs)
{
    std::vector<es::EsTensorHolder> inputs;
    for (size_t slot = 0; slot < subgraphInputs.size(); ++slot) {
        auto matchNode = subgraphInputs[slot].GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        std::string inputName = "boundary_input_" + std::to_string(slot);
        inputs.emplace_back(builder.CreateInput(static_cast<int64_t>(slot), inputName.c_str(), tensorDesc.GetDataType(),
                                                tensorDesc.GetFormat(), GetShapeDims(tensorDesc.GetShape())));
    }
    return inputs;
}

// Build a Mul node (IR: INPUT x1, INPUT x2, OUTPUT y) wired to two tensor holders.
GNode BuildMulNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input0, const es::EsTensorHolder& input1,
                   const std::string& nodeName)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode mulNode = es::CompliantNodeBuilder(graph)
                        .OpType(kMulOpType.c_str())
                        .Name(nodeName.c_str())
                        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *input0.GetProducer(), input0.GetProducerOutIndex(), mulNode, 0) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for Mul input 0.");
        return mulNode;
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *input1.GetProducer(), input1.GetProducerOutIndex(), mulNode, 1) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for Mul input 1.");
    }
    return mulNode;
}

// Build an AddN node wired to n tensor holders. In pattern graphs the node must declare n regular
// inputs because the pattern matcher rejects dynamic input declarations; in replacement graphs the
// regular IR definition (DYNAMIC_INPUT x + REQUIRED_ATTR N) is used.
GNode BuildAddNNode(es::EsGraphBuilder& builder, const std::vector<es::EsTensorHolder>& inputs,
                    const std::string& nodeName, bool forPattern)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    const int64_t inputNum = static_cast<int64_t>(inputs.size());
    GNode addnNode;
    if (forPattern) {
        // The matcher compares node type, input num and IR attrs (not input names), so the pattern
        // node declares n regular inputs with distinct names to keep n input anchors.
        std::vector<es::CompliantNodeBuilder::IrInputDef> inputDefs;
        for (int64_t i = 0; i < inputNum; ++i) {
            inputDefs.push_back({"x" + std::to_string(i), es::CompliantNodeBuilder::kEsIrInputRequired, ""});
        }
        addnNode = es::CompliantNodeBuilder(graph)
                       .OpType(kAddNOpType.c_str())
                       .Name(nodeName.c_str())
                       .IrDefInputs(inputDefs)
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .IrDefAttrs({{"N", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(inputNum)}})
                       .Build();
    } else {
        addnNode = es::CompliantNodeBuilder(graph)
                       .OpType(kAddNOpType.c_str())
                       .Name(nodeName.c_str())
                       .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""}})
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .IrDefAttrs({{"N", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(inputNum)}})
                       .InstanceDynamicInputNum("x", static_cast<int32_t>(inputNum))
                       .Build();
    }
    for (int64_t i = 0; i < inputNum; ++i) {
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[static_cast<size_t>(i)].GetProducer(),
                                         inputs[static_cast<size_t>(i)].GetProducerOutIndex(), addnNode,
                                         static_cast<int32_t>(i)) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to add edge for AddN input [%ld].", i);
            return addnNode;
        }
    }
    return addnNode;
}

// Build a FusedMulAddN node (IR: INPUT x1, INPUT x2, INPUT x3, OUTPUT y) wired to three holders.
GNode BuildFusedMulAddNNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input0,
                            const es::EsTensorHolder& input1, const es::EsTensorHolder& input2,
                            const std::string& nodeName)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode fusedNode = es::CompliantNodeBuilder(graph)
                          .OpType(kFusedMulAddNOpType.c_str())
                          .Name(nodeName.c_str())
                          .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x3", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    const std::vector<es::EsTensorHolder> holders = {input0, input1, input2};
    for (size_t i = 0; i < holders.size(); ++i) {
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *holders[i].GetProducer(), holders[i].GetProducerOutIndex(), fusedNode,
                                         static_cast<int32_t>(i)) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to add edge for FusedMulAddN input [%zu].", i);
            return fusedNode;
        }
    }
    return fusedNode;
}

// Build a MulAddn node (IR: DYNAMIC_INPUT x1, DYNAMIC_INPUT x2, REQUIRED_ATTR N, OUTPUT y)
// wired to 2n tensor holders, first the x1 group then the x2 group in AddN input order.
GNode BuildMulAddnNode(es::EsGraphBuilder& builder, const std::vector<es::EsTensorHolder>& x1Holders,
                       const std::vector<es::EsTensorHolder>& x2Holders, const std::string& nodeName)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    const int64_t inputNum = static_cast<int64_t>(x1Holders.size());
    GNode mulAddnNode = es::CompliantNodeBuilder(graph)
                            .OpType(kMulAddnOpType.c_str())
                            .Name(nodeName.c_str())
                            .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                                          {"x2", es::CompliantNodeBuilder::kEsIrInputDynamic, ""}})
                            .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .IrDefAttrs(
                                {{"N", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(inputNum)}})
                            .InstanceDynamicInputNum("x1", static_cast<int32_t>(inputNum))
                            .InstanceDynamicInputNum("x2", static_cast<int32_t>(inputNum))
                            .Build();
    for (int64_t i = 0; i < inputNum; ++i) {
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *x1Holders[static_cast<size_t>(i)].GetProducer(),
                                         x1Holders[static_cast<size_t>(i)].GetProducerOutIndex(), mulAddnNode,
                                         static_cast<int32_t>(i)) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to add edge for MulAddn x1 input [%ld].", i);
            return mulAddnNode;
        }
    }
    for (int64_t i = 0; i < inputNum; ++i) {
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *x2Holders[static_cast<size_t>(i)].GetProducer(),
                                         x2Holders[static_cast<size_t>(i)].GetProducerOutIndex(), mulAddnNode,
                                         static_cast<int32_t>(inputNum + i)) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to add edge for MulAddn x2 input [%ld].", i);
            return mulAddnNode;
        }
    }
    return mulAddnNode;
}

// Checks shared by every matched Mul node on the multi-input path: input num, stream label,
// dtype, format and the MulAddn kernel shape constraints.
bool CheckMulNodeForMulAddn(const GNode& mulNode, const std::string& addnStreamLabel)
{
    if (mulNode.GetInputsSize() != kMulInputNum) {
        OP_LOGI(kPassName.c_str(), "Input num of Mul node is [%zu], which is not 2.", mulNode.GetInputsSize());
        return false;
    }
    // legacy check on the multi-input path: each Mul output must be consumed by the AddN only.
    if (mulNode.GetOutDataNodesAndPortIndexs(0).size() != 1) {
        OP_LOGI(kPassName.c_str(), "Output of Mul node is consumed by [%zu] nodes, which is not 1.",
                mulNode.GetOutDataNodesAndPortIndexs(0).size());
        return false;
    }
    std::string mulStreamLabel;
    GetStreamLabel(mulNode, mulStreamLabel);
    if (addnStreamLabel != mulStreamLabel) {
        OP_LOGI(kPassName.c_str(), "AddN node and Mul node do not have the same stream label.");
        return false;
    }
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    if (mulNode.GetInputDesc(0, x1Desc) != GRAPH_SUCCESS || mulNode.GetInputDesc(1, x2Desc) != GRAPH_SUCCESS) {
        OP_LOGI(kPassName.c_str(), "Failed to get input desc of Mul node.");
        return false;
    }
    if (!CheckDtype(x1Desc.GetDataType()) || !CheckDtype(x2Desc.GetDataType())) {
        OP_LOGI(kPassName.c_str(), "Dtype of Mul inputs is not float, fp16 or bf16.");
        return false;
    }
    if (x1Desc.GetFormat() != FORMAT_ND || x2Desc.GetFormat() != FORMAT_ND) {
        OP_LOGI(kPassName.c_str(), "Format of Mul inputs is not ND.");
        return false;
    }
    const std::vector<int64_t> x1Dims = x1Desc.GetShape().GetDims();
    const std::vector<int64_t> x2Dims = x2Desc.GetShape().GetDims();
    if (x1Dims.size() != 3 || x2Dims.size() != 3) {
        OP_LOGI(kPassName.c_str(), "Mul inputs must both be 3D tensors.");
        return false;
    }
    if (x1Dims[2] != 1 || x2Dims[1] != 1) {
        OP_LOGI(kPassName.c_str(), "The 3rd dim of x1 or the 2nd dim of x2 is not 1.");
        return false;
    }
    if (x2Dims[2] > kX2ThirdDimLimit) {
        OP_LOGI(kPassName.c_str(), "The 3rd dim of x2 is greater than 2040.");
        return false;
    }
    return true;
}
} // namespace

std::vector<PatternUniqPtr> MulAddNPass::Patterns()
{
    OP_LOGI(kPassName.c_str(), "Enter Patterns for MulAddNPass");
    std::vector<PatternUniqPtr> patternGraphs;
    if (!IsSupportedRuntimeVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version is below 9.0.0, skip pass.");
        return patternGraphs;
    }

    // Pattern for the two-input path: AddN(Mul(x1, x2), x3) with exactly 2 inputs, Mul pinned at
    // AddN input 0 (the legacy pass only fused when Mul was the 0th input of AddN).
    {
        auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
        auto mulInput0 = graphBuilder.CreateInput(0);
        auto mulInput1 = graphBuilder.CreateInput(1);
        auto addnOtherInput = graphBuilder.CreateInput(2);
        GNode mulNode = BuildMulNode(graphBuilder, mulInput0, mulInput1, "pattern_mul");
        auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
        if (mulOutput == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to get Mul output holder for two-input pattern.");
            return patternGraphs;
        }
        GNode addnNode = BuildAddNNode(graphBuilder, {es::EsTensorHolder(mulOutput), addnOtherInput}, "pattern_addn",
                                       true);
        auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
        if (addnOutput == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to get AddN output holder for two-input pattern.");
            return patternGraphs;
        }
        auto graph = graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
        if (graph == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to build two-input pattern graph.");
            return patternGraphs;
        }
        auto pattern = std::make_unique<Pattern>(std::move(*graph));
        pattern->CaptureTensor({addnOutput->GetProducer(), 0});
        pattern->CaptureTensor({mulOutput->GetProducer(), 0});
        patternGraphs.emplace_back(std::move(pattern));
    }

    // Patterns for the multi-input path: AddN whose inputs are all Mul, input num in
    // [3, kMaxMultiInputAddNNum]. Pattern inputs are created per Mul in AddN input order.
    for (int64_t inputNum = kTwoInputAddNNum + 1; inputNum <= kMaxMultiInputAddNNum; ++inputNum) {
        auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
        std::vector<es::EsTensorHolder> mulOutputs;
        bool buildFailed = false;
        for (int64_t i = 0; i < inputNum; ++i) {
            auto mulInput0 = graphBuilder.CreateInput(2 * i);
            auto mulInput1 = graphBuilder.CreateInput(2 * i + 1);
            std::string mulNodeName = "pattern_mul_" + std::to_string(i);
            GNode mulNode = BuildMulNode(graphBuilder, mulInput0, mulInput1, mulNodeName);
            auto mulOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
            if (mulOutput == nullptr) {
                buildFailed = true;
                break;
            }
            mulOutputs.emplace_back(mulOutput);
        }
        if (buildFailed) {
            OP_LOGE(kPassName.c_str(), "Failed to build Mul node for multi-input pattern, input num [%ld].", inputNum);
            continue;
        }
        GNode addnNode = BuildAddNNode(graphBuilder, mulOutputs, "pattern_addn", true);
        auto addnOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addnNode, 0);
        if (addnOutput == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to get AddN output holder, input num [%ld].", inputNum);
            continue;
        }
        auto graph = graphBuilder.BuildAndReset({es::EsTensorHolder(addnOutput)});
        if (graph == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to build multi-input pattern graph, input num [%ld].", inputNum);
            continue;
        }
        auto pattern = std::make_unique<Pattern>(std::move(*graph));
        pattern->CaptureTensor({addnOutput->GetProducer(), 0});
        for (int64_t i = 0; i < inputNum; ++i) {
            pattern->CaptureTensor({*mulOutputs[static_cast<size_t>(i)].GetProducer(), 0});
        }
        patternGraphs.emplace_back(std::move(pattern));
    }

    return patternGraphs;
}

bool MulAddNPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGI(kPassName.c_str(), "Enter MeetRequirements for MulAddNPass");

    if (!IsSupportedRuntimeVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version is below 9.0.0, skip pass.");
        return false;
    }

    NodeIo addnNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxAddNNode, addnNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured AddN tensor.");
        return false;
    }
    GNode addnNode = addnNodeIo.node;
    AscendString addnType;
    addnNode.GetType(addnType);
    if (std::string(addnType.GetString()) != kAddNOpType) {
        OP_LOGI(kPassName.c_str(), "Matched node type [%s] is not AddN.", addnType.GetString());
        return false;
    }

    const int64_t addnInputNum = static_cast<int64_t>(addnNode.GetInputsSize());
    if (addnInputNum < kTwoInputAddNNum) {
        OP_LOGI(kPassName.c_str(), "Input num of AddN node is [%ld], too few to fuse.", addnInputNum);
        return false;
    }

    std::string addnStreamLabel;
    GetStreamLabel(addnNode, addnStreamLabel);

    if (addnInputNum == kTwoInputAddNNum) {
        // Two-input path: AddN(Mul(x1, x3), x2) --> FusedMulAddN(x1, x2, x3).
        // The legacy pass applied no soc/dtype/format restriction on this path.
        NodeIo mulNodeIo;
        if (matchResult->GetCapturedTensor(kCaptureIdxFirstMulNode, mulNodeIo) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to get captured Mul tensor.");
            return false;
        }
        GNode mulNode = mulNodeIo.node;
        AscendString mulType;
        mulNode.GetType(mulType);
        if (std::string(mulType.GetString()) != kMulOpType) {
            OP_LOGI(kPassName.c_str(), "Matched node type [%s] is not Mul.", mulType.GetString());
            return false;
        }
        if (mulNode.GetInputsSize() != kMulInputNum) {
            OP_LOGI(kPassName.c_str(), "Input num of Mul node is [%zu], which is not 2.", mulNode.GetInputsSize());
            return false;
        }
        // legacy CheckNodeInfo: the Mul output must be consumed by this AddN only.
        if (mulNode.GetOutDataNodesAndPortIndexs(0).size() != 1) {
            OP_LOGI(kPassName.c_str(), "Output of Mul node is consumed by [%zu] nodes, which is not 1.",
                    mulNode.GetOutDataNodesAndPortIndexs(0).size());
            return false;
        }
        std::string mulStreamLabel;
        GetStreamLabel(mulNode, mulStreamLabel);
        if (addnStreamLabel != mulStreamLabel) {
            OP_LOGI(kPassName.c_str(), "AddN node and Mul node do not have the same stream label.");
            return false;
        }
        // The scalar side of Mul must be scalar or shape-(1,), same as legacy CheckLossScaleInput.
        if (GetScalarSideIdx(mulNode) < 0) {
            OP_LOGI(kPassName.c_str(), "No input of Mul node is scalar or shape-(1,).");
            return false;
        }
        return true;
    }

    // Multi-input path: AddN inputs are all Mul (guaranteed by pattern topology) --> MulAddn.
    // Patterns cover N in [3, kMaxMultiInputAddNNum] and the matcher rejects input-count
    // mismatches first, so an out-of-range N cannot reach here; warn only (no rejection) and
    // let the per-Mul capture validation below backstop any hypothetical overflow.
    if (addnInputNum > kMaxMultiInputAddNNum) {
        OP_LOGW(kPassName.c_str(), "AddN input num [%ld] is beyond pattern coverage [3, %ld].", addnInputNum,
                kMaxMultiInputAddNNum);
    }

    // Legacy CheckSoc gates the multi-input path on Ascend910B / Ascend910_93, which both map
    // to NpuArch DAV_2201; express the gate by arch so same-arch chips are covered uniformly.
    char npuArchVal[32] = {0};
    if (rtGetSocSpec("version", "NpuArch", npuArchVal, sizeof(npuArchVal)) != RT_ERROR_NONE) {
        OP_LOGW(kPassName.c_str(), "Get NpuArch from soc spec failed, skip the multi-input fusion.");
        return false;
    }
    const auto npuArch = static_cast<NpuArch>(std::strtol(npuArchVal, nullptr, 10));
    static const std::set<NpuArch> supportedArch = {NpuArch::DAV_2201};
    if (supportedArch.count(npuArch) == 0) {
        OP_LOGI(kPassName.c_str(), "NpuArch [%u] is not DAV_2201, skip the multi-input fusion.",
                static_cast<uint32_t>(npuArch));
        return false;
    }

    // Legacy CheckInfo on the AddN node (input dtype/format; the legacy condition mixed input
    // indices, the intended check on input 0 is applied here).
    TensorDesc addnInputDesc;
    if (addnNode.GetInputDesc(0, addnInputDesc) != GRAPH_SUCCESS) {
        OP_LOGI(kPassName.c_str(), "Failed to get input desc of AddN node.");
        return false;
    }
    if (!CheckDtype(addnInputDesc.GetDataType())) {
        OP_LOGI(kPassName.c_str(), "Dtype of AddN input is not float, fp16 or bf16.");
        return false;
    }
    if (addnInputDesc.GetFormat() != FORMAT_ND) {
        OP_LOGI(kPassName.c_str(), "Format of AddN input is not ND.");
        return false;
    }

    for (int64_t i = 0; i < addnInputNum; ++i) {
        NodeIo mulNodeIo;
        if (matchResult->GetCapturedTensor(kCaptureIdxFirstMulNode + static_cast<size_t>(i), mulNodeIo) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to get captured Mul tensor, index [%ld].", i);
            return false;
        }
        GNode mulNode = mulNodeIo.node;
        AscendString mulType;
        mulNode.GetType(mulType);
        if (std::string(mulType.GetString()) != kMulOpType) {
            OP_LOGI(kPassName.c_str(), "The [%ld]th input of AddN is not a Mul node.", i);
            return false;
        }
        if (!CheckMulNodeForMulAddn(mulNode, addnStreamLabel)) {
            return false;
        }
    }
    return true;
}

GraphUniqPtr MulAddNPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGI(kPassName.c_str(), "Enter Replacement for MulAddNPass");

    NodeIo addnNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxAddNNode, addnNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured AddN tensor in Replacement.");
        return nullptr;
    }
    GNode addnNode = addnNodeIo.node;
    const std::string addnNodeName = GetNodeName(addnNode);
    const int64_t addnInputNum = static_cast<int64_t>(addnNode.GetInputsSize());

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    if (addnInputNum == kTwoInputAddNNum) {
        // Two-input path: build FusedMulAddN(x1, x2, x3) with y = x1 * x3[0] + x2.
        NodeIo mulNodeIo;
        if (matchResult->GetCapturedTensor(kCaptureIdxFirstMulNode, mulNodeIo) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to get captured Mul tensor in Replacement.");
            return nullptr;
        }
        GNode mulNode = mulNodeIo.node;
        const std::string mulNodeName = GetNodeName(mulNode);
        const int32_t scalarSideIdx = GetScalarSideIdx(mulNode);
        if (scalarSideIdx < 0) {
            OP_LOGE(kPassName.c_str(), "No scalar side found on Mul node in Replacement.");
            return nullptr;
        }
        const int32_t vectorSideIdx = 1 - scalarSideIdx;

        const int64_t slotX1 = FindBoundarySlot(subgraphInputs, mulNodeName, vectorSideIdx);
        const int64_t slotX2 = FindBoundarySlot(subgraphInputs, addnNodeName, kAddNOtherInputIdx);
        const int64_t slotX3 = FindBoundarySlot(subgraphInputs, mulNodeName, scalarSideIdx);
        if (slotX1 < 0 || slotX2 < 0 || slotX3 < 0) {
            OP_LOGE(kPassName.c_str(),
                    "Failed to locate boundary slots for FusedMulAddN inputs, "
                    "slotX1 [%ld], slotX2 [%ld], slotX3 [%ld].",
                    slotX1, slotX2, slotX3);
            return nullptr;
        }

        auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
        std::vector<es::EsTensorHolder> boundaryInputs = CreateBoundaryInputs(replaceGraphBuilder, subgraphInputs);
        std::string fusedNodeName = addnNodeName.empty() ? kFusedMulAddNOpType :
                                                           addnNodeName + "/" + kFusedMulAddNOpType;
        GNode fusedNode = BuildFusedMulAddNNode(replaceGraphBuilder, boundaryInputs[static_cast<size_t>(slotX1)],
                                                boundaryInputs[static_cast<size_t>(slotX2)],
                                                boundaryInputs[static_cast<size_t>(slotX3)], fusedNodeName);

        TensorDesc x1Desc;
        TensorDesc x2Desc;
        TensorDesc x3Desc;
        TensorDesc outputDesc;
        if (mulNode.GetInputDesc(vectorSideIdx, x1Desc) != GRAPH_SUCCESS ||
            addnNode.GetInputDesc(kAddNOtherInputIdx, x2Desc) != GRAPH_SUCCESS ||
            mulNode.GetInputDesc(scalarSideIdx, x3Desc) != GRAPH_SUCCESS ||
            addnNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to get desc of source nodes for FusedMulAddN.");
            return nullptr;
        }
        fusedNode.UpdateInputDesc(0, x1Desc);
        fusedNode.UpdateInputDesc(1, x2Desc);
        fusedNode.UpdateInputDesc(2, x3Desc);
        fusedNode.UpdateOutputDesc(0, outputDesc);

        auto outputHolder = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(fusedNode, 0);
        if (outputHolder == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to get FusedMulAddN output holder.");
            return nullptr;
        }
        GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({es::EsTensorHolder(outputHolder)});
        if (replaceGraph == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to build FusedMulAddN replacement graph.");
            return nullptr;
        }
        if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Infershape for FusedMulAddN replacement failed.");
            return nullptr;
        }
        OP_LOGI(kPassName.c_str(), "MulAddNPass two-input fusion success.");
        return replaceGraph;
    }

    // Multi-input path: build MulAddn(x1[0..n-1], x2[0..n-1], N) where x1_i / x2_i are the
    // input-0 / input-1 of the i-th matched Mul, in AddN input order (legacy wiring order).
    if (addnInputNum <= kTwoInputAddNNum || addnInputNum > kMaxMultiInputAddNNum) {
        OP_LOGE(kPassName.c_str(), "Invalid AddN input num [%ld] in Replacement.", addnInputNum);
        return nullptr;
    }

    std::vector<GNode> mulNodes;
    for (int64_t i = 0; i < addnInputNum; ++i) {
        NodeIo mulNodeIo;
        if (matchResult->GetCapturedTensor(kCaptureIdxFirstMulNode + static_cast<size_t>(i), mulNodeIo) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to get captured Mul tensor, index [%ld], in Replacement.", i);
            return nullptr;
        }
        mulNodes.emplace_back(mulNodeIo.node);
    }

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    std::vector<es::EsTensorHolder> boundaryInputs = CreateBoundaryInputs(replaceGraphBuilder, subgraphInputs);

    std::vector<es::EsTensorHolder> x1Holders;
    std::vector<es::EsTensorHolder> x2Holders;
    std::vector<TensorDesc> x1Descs;
    std::vector<TensorDesc> x2Descs;
    for (int64_t i = 0; i < addnInputNum; ++i) {
        const std::string mulNodeName = GetNodeName(mulNodes[static_cast<size_t>(i)]);
        const int64_t slotX1 = FindBoundarySlot(subgraphInputs, mulNodeName, 0);
        const int64_t slotX2 = FindBoundarySlot(subgraphInputs, mulNodeName, 1);
        if (slotX1 < 0 || slotX2 < 0) {
            OP_LOGE(kPassName.c_str(),
                    "Failed to locate boundary slots for the [%ld]th Mul, slotX1 [%ld], "
                    "slotX2 [%ld].",
                    i, slotX1, slotX2);
            return nullptr;
        }
        x1Holders.emplace_back(boundaryInputs[static_cast<size_t>(slotX1)]);
        x2Holders.emplace_back(boundaryInputs[static_cast<size_t>(slotX2)]);
        TensorDesc x1Desc;
        TensorDesc x2Desc;
        if (mulNodes[static_cast<size_t>(i)].GetInputDesc(0, x1Desc) != GRAPH_SUCCESS ||
            mulNodes[static_cast<size_t>(i)].GetInputDesc(1, x2Desc) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to get input desc of the [%ld]th Mul.", i);
            return nullptr;
        }
        x1Descs.emplace_back(x1Desc);
        x2Descs.emplace_back(x2Desc);
    }

    const std::string mulAddnNodeName = addnNodeName.empty() ? kMulAddnOpType : addnNodeName + "/" + kMulAddnOpType;
    GNode mulAddnNode = BuildMulAddnNode(replaceGraphBuilder, x1Holders, x2Holders, mulAddnNodeName);
    for (int64_t i = 0; i < addnInputNum; ++i) {
        mulAddnNode.UpdateInputDesc(static_cast<int32_t>(i), x1Descs[static_cast<size_t>(i)]);
        mulAddnNode.UpdateInputDesc(static_cast<int32_t>(addnInputNum + i), x2Descs[static_cast<size_t>(i)]);
    }
    TensorDesc outputDesc;
    if (addnNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get output desc of AddN node.");
        return nullptr;
    }
    mulAddnNode.UpdateOutputDesc(0, outputDesc);

    auto mulAddnOutputHolder = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulAddnNode, 0);
    if (mulAddnOutputHolder == nullptr) {
        OP_LOGE(kPassName.c_str(), "Failed to get MulAddn output holder.");
        return nullptr;
    }
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({es::EsTensorHolder(mulAddnOutputHolder)});
    if (replaceGraph == nullptr) {
        OP_LOGE(kPassName.c_str(), "Failed to build MulAddn replacement graph.");
        return nullptr;
    }
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Infershape for MulAddn replacement failed.");
        return nullptr;
    }
    OP_LOGI(kPassName.c_str(), "MulAddNPass multi-input fusion success, N [%ld].", addnInputNum);
    return replaceGraph;
}

REG_FUSION_PASS(MulAddNPass).Stage(GetMulAddNPassStage());

} // namespace ge::fusion

#endif // GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
