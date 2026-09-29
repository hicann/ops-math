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
 * \file concat_to_concatd_fusion_pass.cpp
 * \brief Concat/ConcatV2/ConcatV2D -> ConcatD
 */

#include <algorithm>
#include <string>
#include <vector>

#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/graph_rewriter.h"
#include "log/log.h"
#include "graph/tensor.h"

#include "concat_to_concatd_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace ge::fusion;

namespace ops {

namespace {
const std::string kPassName = "ConcatToConcatDFusionPass";
const std::string kTypeConcat = "Concat";
const std::string kTypeConcatV2 = "ConcatV2";
const std::string kTypeConcatV2D = "ConcatV2D";
constexpr int32_t kConcatDimIdxConcat = 0;
constexpr int64_t kMaxConcatDInputs = 512; // max data inputs per ConcatD node; more are split into a tree

bool IsTargetPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    const bool isPlatform950 = (soc == "Ascend950");
    if (!isPlatform950) {
        OP_LOGI(kPassName.c_str(), "Platform %s is not Ascend950, skip whole graph.", soc.c_str());
    }
    return isPlatform950;
}

bool ReadConcatDim(const GNode& matchedNode, int32_t concatDimIdx, int64_t& concatDim)
{
    Tensor constTensor;
    if (matchedNode.GetInputConstData(concatDimIdx, constTensor) != SUCCESS) {
        OP_LOGI(kPassName.c_str(), "Concat has input which is not a constant, skip node.");
        return false;
    }
    const DataType dimDtype = constTensor.GetTensorDesc().GetDataType();
    const uint8_t* rawData = constTensor.GetData();
    if (rawData == nullptr) {
        OP_LOGD(kPassName.c_str(), "Concat_dim const data is null, skip node.");
        return false;
    }
    if (dimDtype == DT_INT32) {
        concatDim = static_cast<int64_t>(*reinterpret_cast<const int32_t*>(rawData));
    } else if (dimDtype == DT_INT64) {
        concatDim = *reinterpret_cast<const int64_t*>(rawData);
    } else {
        OP_LOGD(kPassName.c_str(), "Unsupported concat_dim dtype %d, skip node.", static_cast<int32_t>(dimDtype));
        return false;
    }
    return true;
}

bool HasZeroDim(const TensorDesc& tensorDesc)
{
    for (const int64_t dim : tensorDesc.GetShape().GetDims()) {
        if (dim == 0) {
            return true;
        }
    }
    return false;
}

// Build a multi-level ConcatD tree when inputs exceed kMaxConcatDInputs: each level
// groups at most kMaxConcatDInputs tensors into one ConcatD node, parents aggregate
// the children outputs, recursively. Leaf ConcatD inherits the input descs of the
// matched node; intermediate levels get descs from the unified inference.
es::EsTensorHolder BuildConcatDTree(es::EsGraphBuilder& builder, std::vector<es::EsTensorHolder> holders,
                                    const std::vector<int64_t>& srcInputIdxs, const GNode& matchedNode,
                                    int64_t concatDim)
{
    if (static_cast<int64_t>(holders.size()) <= kMaxConcatDInputs) {
        auto y = es::ConcatD(holders, concatDim, static_cast<int64_t>(holders.size()));
        GNode concatDNode = *y.GetProducer();
        for (size_t i = 0; i < holders.size(); ++i) {
            TensorDesc tensorDesc;
            if (matchedNode.GetInputDesc(static_cast<int32_t>(srcInputIdxs[i]), tensorDesc) == SUCCESS) {
                (void)concatDNode.UpdateInputDesc(static_cast<int32_t>(i), tensorDesc);
            }
        }
        return y;
    }

    std::vector<es::EsTensorHolder> parentHolders;
    std::vector<int64_t> parentIdxs;
    const int64_t total = static_cast<int64_t>(holders.size());
    for (int64_t begin = 0; begin < total; begin += kMaxConcatDInputs) {
        const int64_t end = std::min(begin + kMaxConcatDInputs, total);
        std::vector<es::EsTensorHolder> group(holders.begin() + begin, holders.begin() + end);
        std::vector<int64_t> groupIdxs(srcInputIdxs.begin() + begin, srcInputIdxs.begin() + end);
        parentHolders.emplace_back(BuildConcatDTree(builder, group, groupIdxs, matchedNode, concatDim));
        parentIdxs.push_back(-1);
    }
    return BuildConcatDTree(builder, parentHolders, parentIdxs, matchedNode, concatDim);
}
} // namespace

Status ConcatToConcatDFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    // The framework sets the pass name before invoking Run in real compilation; set it
    // here for direct Run calls (e.g. UT with a default-constructed context) so the
    // fusion-result reporting works everywhere.
    if (passContext.GetPassName().GetLength() == 0) {
        passContext.SetPassName(kPassName.c_str());
    }
    OP_LOGI(kPassName.c_str(), "ConcatToConcatDFusionPass fusion start.");
    if (!IsTargetPlatform()) {
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> matchedNodes;
    for (auto node : graph->GetAllNodes()) {
        AscendString typeStr;
        node.GetType(typeStr);
        const std::string opType = typeStr.GetString();
        if (opType == kTypeConcat || opType == kTypeConcatV2 || opType == kTypeConcatV2D) {
            matchedNodes.emplace_back(node);
        }
    }
    if (matchedNodes.empty()) {
        OP_LOGD(kPassName.c_str(), "No concat node found, graph not changed.");
        return GRAPH_NOT_CHANGED;
    }

    int32_t fusedCount = 0;
    for (auto matchedNode : matchedNodes) {
        AscendString nodeNameStr;
        (void)matchedNode.GetName(nodeNameStr);
        AscendString typeStr;
        matchedNode.GetType(typeStr);
        const std::string opType = typeStr.GetString();
        const int64_t inputNum = static_cast<int64_t>(matchedNode.GetInputsSize());

        // Locate the concat_dim input index (-1 means ConcatV2D, concat_dim is an attr).
        int32_t concatDimIdx = -1;
        if (opType == kTypeConcat) {
            concatDimIdx = kConcatDimIdxConcat;
        } else if (opType == kTypeConcatV2) {
            concatDimIdx = static_cast<int32_t>(inputNum) - 1;
        }

        // Guard: axis must be a constant (Concat/ConcatV2) or an attr (ConcatV2D).
        int64_t concatDim = 0;
        if (concatDimIdx >= 0) {
            if (inputNum < 2) {
                OP_LOGD(kPassName.c_str(), "Node %s has too few inputs (%ld), skip node.", nodeNameStr.GetString(),
                        inputNum);
                continue;
            }
            if (!ReadConcatDim(matchedNode, concatDimIdx, concatDim)) {
                continue;
            }
        } else {
            if (matchedNode.GetAttr("concat_dim", concatDim) != SUCCESS) {
                OP_LOGD(kPassName.c_str(), "ConcatV2D node %s has no concat_dim attr, skip node.",
                        nodeNameStr.GetString());
                continue;
            }
        }

        // Prune empty-tensor data inputs (shape contains a zero dim), Concat/ConcatV2
        // only, ConcatV2D never prunes; if all data inputs are empty, keep all.
        std::vector<int64_t> dataIdxs;
        std::vector<int64_t> keepIdxs;
        for (int64_t i = 0; i < inputNum; ++i) {
            if (concatDimIdx >= 0 && i == concatDimIdx) {
                continue;
            }
            dataIdxs.push_back(i);
            TensorDesc tensorDesc;
            if (concatDimIdx >= 0 && matchedNode.GetInputDesc(static_cast<int32_t>(i), tensorDesc) == SUCCESS &&
                HasZeroDim(tensorDesc)) {
                continue; // pruned
            }
            keepIdxs.push_back(i);
        }
        if (keepIdxs.empty() && !dataIdxs.empty()) {
            keepIdxs = dataIdxs; // all data inputs are empty, keep all (original behavior)
        }
        const int64_t dataInputNum = static_cast<int64_t>(keepIdxs.size());
        if (dataInputNum < 1) {
            OP_LOGD(kPassName.c_str(), "Node %s has no data input, skip node.", nodeNameStr.GetString());
            continue;
        }

        // Build the boundary: every matched-node input slot (axis, pruned and kept)
        // becomes a boundary input with the same index, so the framework grafts
        // boundary inputs one-to-one without any offset and a shared axis Const stays
        // legal (its edge remains declared in the boundary).
        auto boundary = std::make_unique<SubgraphBoundary>();
        bool nodeOk = true;
        for (int64_t i = 0; i < inputNum; ++i) {
            SubgraphInput subgraphInput;
            if (subgraphInput.AddInput({matchedNode, i}) != SUCCESS ||
                boundary->AddInput(i, std::move(subgraphInput)) != SUCCESS) {
                nodeOk = false;
                break;
            }
        }
        SubgraphOutput subgraphOutput;
        if (nodeOk && subgraphOutput.SetOutput({matchedNode, 0}) != SUCCESS) {
            nodeOk = false;
        }
        if (nodeOk && boundary->AddOutput(0, std::move(subgraphOutput)) != SUCCESS) {
            nodeOk = false;
        }
        if (!nodeOk) {
            OP_LOGE(kPassName.c_str(), "Node %s failed to build boundary.", nodeNameStr.GetString());
            continue;
        }

        // Build the replacement graph: axis and pruned slots become dangling placeholder
        // Data nodes (not consumed); kept slots feed the ConcatD node (or tree).
        auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
        std::vector<es::EsTensorHolder> dataHolders;
        dataHolders.reserve(dataInputNum);
        for (int64_t i = 0; i < inputNum; ++i) {
            TensorDesc tensorDesc;
            if (matchedNode.GetInputDesc(static_cast<int32_t>(i), tensorDesc) != SUCCESS) {
                OP_LOGE(kPassName.c_str(), "Node %s failed to get input desc %ld.", nodeNameStr.GetString(), i);
                nodeOk = false;
                break;
            }
            const bool isAxisSlot = (concatDimIdx >= 0 && i == concatDimIdx);
            const bool isPrunedSlot = (concatDimIdx >= 0 && i != concatDimIdx &&
                                       std::find(keepIdxs.begin(), keepIdxs.end(), i) == keepIdxs.end());
            if (isAxisSlot || isPrunedSlot) {
                // Dangling placeholder: occupies the boundary slot only.
                (void)replaceGraphBuilder.CreateInput(i, ("placeholder" + std::to_string(i)).c_str(),
                                                      tensorDesc.GetDataType(), tensorDesc.GetFormat(),
                                                      tensorDesc.GetShape().GetDims());
                continue;
            }
            // Names must be unique, otherwise the builder merges the Data nodes and
            // multiple ConcatD inputs end up connected to one single placeholder.
            dataHolders.emplace_back(replaceGraphBuilder.CreateInput(i, ("x" + std::to_string(i)).c_str(),
                                                                     tensorDesc.GetDataType(), tensorDesc.GetFormat(),
                                                                     tensorDesc.GetShape().GetDims()));
        }
        if (!nodeOk) {
            continue;
        }

        auto y = BuildConcatDTree(replaceGraphBuilder, dataHolders, keepIdxs, matchedNode, concatDim);
        std::shared_ptr<Graph> replaceGraphPtr = replaceGraphBuilder.BuildAndReset({y});
        if (replaceGraphPtr == nullptr) {
            OP_LOGE(kPassName.c_str(), "Node %s failed to build replacement graph.", nodeNameStr.GetString());
            continue;
        }

        // Inherit the source output desc BEFORE the support check: ConcatD keeps the
        // output dtype/shape/format of the source node. Without this the freshly built
        // ConcatD carries a default output desc (measured: es default dtype FLOAT),
        // which makes CheckNodeSupportOnAicore reject int32 cases with a spurious
        // "format and dtype not equivalent" verdict.
        GNode topNode = *y.GetProducer();
        TensorDesc srcOutputDesc;
        if (matchedNode.GetOutputDesc(0, srcOutputDesc) == SUCCESS) {
            (void)topNode.UpdateOutputDesc(0, srcOutputDesc);
        } else {
            TensorDesc firstDataDesc;
            if (matchedNode.GetInputDesc(static_cast<int32_t>(keepIdxs[0]), firstDataDesc) == SUCCESS) {
                TensorDesc dstOutputDesc;
                if (topNode.GetOutputDesc(0, dstOutputDesc) == SUCCESS) {
                    dstOutputDesc.SetDataType(firstDataDesc.GetDataType());
                    dstOutputDesc.SetFormat(firstDataDesc.GetFormat());
                    (void)topNode.UpdateOutputDesc(0, dstOutputDesc);
                }
            }
        }

        // AI Core support check: the verdict is only trusted when it explicitly reports
        // "not supported"; a failed check is logged and the node proceeds, because at
        // kBeforeInferShape shapes are not derived yet (and UT has no op-kernel
        // registry) - GE's own op selection during compilation still guards the rest.
        bool isSupported = false;
        AscendString unsupportedReason;
        const Status checkStatus = GeUtils::CheckNodeSupportOnAicore(topNode, isSupported, unsupportedReason);
        if (checkStatus == SUCCESS && !isSupported) {
            OP_LOGI(kPassName.c_str(), "ConcatD not supported (%s), skip node %s.", unsupportedReason.GetString(),
                    nodeNameStr.GetString());
            continue;
        }
        if (checkStatus != SUCCESS) {
            OP_LOGW(kPassName.c_str(),
                    "CheckNodeSupportOnAicore not conclusive (status %d), proceed without the check.",
                    static_cast<int32_t>(checkStatus));
        }

        // The 4-arg overload performs the fusable check AND the official fusion-result
        // reporting (match_times/effect_times) that ATC aggregates into
        // fusion_result.json; the 2-arg overload skips the reporting.
        const Status replaceStatus = SubgraphRewriter::Replace(*boundary, *replaceGraphPtr, passContext);
        if (replaceStatus != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "SubgraphRewriter::Replace failed for node %s, status=%d.",
                    nodeNameStr.GetString(), static_cast<int32_t>(replaceStatus));
            return FAILED;
        }

        ++fusedCount;
        OP_LOGI(kPassName.c_str(), "%s node %s --> ConcatD fusion SUCCESS (concat_dim=%ld, data inputs=%ld of %zu).",
                opType.c_str(), nodeNameStr.GetString(), concatDim, dataInputNum, dataIdxs.size());
    }

    OP_LOGI(kPassName.c_str(), "ConcatToConcatDFusionPass fusion end, %d node(s) fused.", fusedCount);
    return (fusedCount > 0) ? SUCCESS : GRAPH_NOT_CHANGED;
}

REG_FUSION_PASS(ConcatToConcatDFusionPass).Stage(CustomPassStage::kBeforeInferShape);

} // namespace ops
