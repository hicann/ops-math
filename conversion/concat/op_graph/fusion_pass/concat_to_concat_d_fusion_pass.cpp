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
 * \file concat_to_concat_d_fusion_pass.cpp
 * \brief Concat fusion pass (Concat --> ConcatD)
 *
 * Pattern:
 *   concat_dim(Const), x0, x1...        x0, x1...
 *              |                              |
 *           Concat              ==>        ConcatD (concat_dim, N as attrs)
 *              |                              |
 *              y                              y
 *
 * The key transformation:
 * - Concat has concat_dim as input[0] (must be Const) and x as dynamic inputs
 * - ConcatD uses concat_dim and N as attributes, x as dynamic inputs
 *
 * Uses FusionBasePass (graph traversal) instead of PatternFusionPass because
 * PatternFusionPass::ValidatePattern rejects patterns containing DYNAMIC_INPUT nodes.
 */

#include <vector>
#include <string>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/fusion/graph_rewriter.h"
#include "log/log.h"
#include "concat_to_concat_d_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace ge::fusion;

namespace ops {

static const std::string kPassName = "ConcatToConcatDFusionPass";

Status ConcatToConcatDFusionPass::Run(GraphPtr &graph, CustomPassContext &pass_context)
{
    (void)pass_context;
    OP_LOGD(kPassName.c_str(), "Enter Run for ConcatToConcatDFusionPass");

    // 1. Platform check - only Ascend950 is supported
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platformInfo failed.");
        return GRAPH_NOT_CHANGED;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc != "Ascend950") {
        OP_LOGD(kPassName.c_str(), "Platform %s is not supported, only Ascend950.", soc.c_str());
        return GRAPH_NOT_CHANGED;
    }

    // 2. Traverse graph to find all Concat nodes that meet requirements
    std::vector<GNode> concatNodes;
    for (auto &node : graph->GetDirectNode()) {
        AscendString nodeType;
        if (node.GetType(nodeType) != GRAPH_SUCCESS) {
            continue;
        }
        if (std::string(nodeType.GetString()) != "Concat") {
            continue;
        }

        // concat_dim input (index 0) must be Const
        auto dimSrcInfo = node.GetInDataNodesAndPortIndexs(0);
        if (dimSrcInfo.first == nullptr) {
            OP_LOGD(kPassName.c_str(), "concat_dim input source node is null, skip.");
            continue;
        }
        AscendString dimType;
        dimSrcInfo.first->GetType(dimType);
        std::string dimTypeStr(dimType.GetString());
        if (dimTypeStr != "Const" && dimTypeStr != "Constant") {
            OP_LOGD(kPassName.c_str(), "concat_dim is not Const (type=%s), skip.", dimTypeStr.c_str());
            continue;
        }

        concatNodes.emplace_back(node);
    }

    if (concatNodes.empty()) {
        OP_LOGD(kPassName.c_str(), "No eligible Concat nodes found.");
        return GRAPH_NOT_CHANGED;
    }

    // 3. Replace each eligible Concat node
    for (auto &concatNode : concatNodes) {
        if (ReplaceConcatWithConcatD(concatNode) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "ReplaceConcatWithConcatD failed.");
            return FAILED;
        }
    }

    return SUCCESS;
}

Status ConcatToConcatDFusionPass::GetConcatDimValue(const GNode &concatNode, int64_t &concatDimValue, int64_t &N)
{
    auto dimSrcInfo = concatNode.GetInDataNodesAndPortIndexs(0);
    GNode constNode = *dimSrcInfo.first;
    Tensor constTensor;
    if (constNode.GetAttr("value", constTensor) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get value attr from concat_dim Const node.");
        return FAILED;
    }
    const int32_t *dimData = static_cast<const int32_t *>(static_cast<const void *>(constTensor.GetData()));
    if (dimData == nullptr) {
        OP_LOGE(kPassName.c_str(), "concat_dim data is null.");
        return FAILED;
    }
    concatDimValue = static_cast<int64_t>(*dimData);
    N = static_cast<int64_t>(concatNode.GetInputsSize()) - 1;
    OP_LOGD(kPassName.c_str(), "concat_dim=%ld, N=%ld", concatDimValue, N);
    return SUCCESS;
}

GraphUniqPtr ConcatToConcatDFusionPass::BuildReplaceGraph(const GNode &concatNode, int64_t concatDimValue, int64_t N)
{
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    // Input 0: concat_dim placeholder (boundary alignment, not connected to ConcatD)
    TensorDesc dimDesc;
    concatNode.GetInputDesc(0, dimDesc);
    replaceGraphBuilder.CreateInput(
        0, "concat_dim", dimDesc.GetDataType(), dimDesc.GetFormat(), dimDesc.GetShape().GetDims());

    // Inputs 1..N: x0..xN-1
    std::vector<es::EsTensorHolder> xInputHolders;
    for (int64_t i = 0; i < N; ++i) {
        TensorDesc xDesc;
        concatNode.GetInputDesc(static_cast<uint32_t>(i + 1), xDesc);
        std::vector<int64_t> dims;
        for (size_t d = 0; d < xDesc.GetShape().GetDimNum(); ++d) {
            dims.push_back(xDesc.GetShape().GetDim(d));
        }
        auto rX = replaceGraphBuilder.CreateInput(
            static_cast<int32_t>(i + 1), ("x" + std::to_string(i)).c_str(),
            xDesc.GetDataType(), xDesc.GetFormat(), dims);
        xInputHolders.push_back(rX);
    }

    auto concatDOut = es::ConcatD(xInputHolders, concatDimValue, N);
    return replaceGraphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{concatDOut});
}

Status ConcatToConcatDFusionPass::ReplaceConcatWithConcatD(const GNode &concatNode)
{
    int64_t concatDimValue = 0;
    int64_t N = 0;
    if (GetConcatDimValue(concatNode, concatDimValue, N) != SUCCESS) {
        return FAILED;
    }

    auto replaceGraph = BuildReplaceGraph(concatNode, concatDimValue, N);

    // InferShape
    std::vector<Shape> inputShapes;
    for (uint32_t i = 0; i < static_cast<uint32_t>(N + 1); ++i) {
        TensorDesc desc;
        concatNode.GetInputDesc(i, desc);
        inputShapes.emplace_back(desc.GetShape());
    }
    if (GeUtils::InferShape(*replaceGraph, inputShapes) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "InferShape for replacement failed.");
        return FAILED;
    }

    // Construct subgraph boundary: N+1 inputs, 1 output
    auto boundary = std::make_unique<SubgraphBoundary>();
    for (int64_t i = 0; i < N + 1; ++i) {
        SubgraphInput subgraphInput;
        subgraphInput.AddInput({concatNode, i});
        if (boundary->AddInput(static_cast<size_t>(i), std::move(subgraphInput)) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "boundary->AddInput failed at index %ld.", i);
            return FAILED;
        }
    }
    SubgraphOutput subgraphOutput({concatNode, 0});
    if (boundary->AddOutput(0, std::move(subgraphOutput)) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "boundary->AddOutput failed.");
        return FAILED;
    }

    if (SubgraphRewriter::Replace(*boundary, *replaceGraph) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "SubgraphRewriter::Replace failed.");
        return FAILED;
    }

    OP_LOGD(kPassName.c_str(), "Concat -> ConcatD replacement done, N=%ld.", N);
    return SUCCESS;
}

REG_FUSION_PASS(ConcatToConcatDFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
