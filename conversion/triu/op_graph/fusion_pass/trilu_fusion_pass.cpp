/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief Trilu fusion pass (Trilu --> Tril/Triu)
 * @details
 *                                           upper?
 *        x, k(input), upper(attr)     0  /          \ 1
 *              |                        /            \
 *            Trilu      ==>   x, diagonal(attr)   x, diagonal(attr)
 *              |                    |                    |
 *              y                   Tril                 Triu
 *                                   |                    |
 *                                   y                    y
 *
 * The key transformation:
 * - Trilu has optional input k (diagonal offset) and attribute upper
 * - Tril/Triu have attribute diagonal instead of input k
 * - upper=0 -> Tril, upper=1 -> Triu
 * - k input value is extracted and converted to diagonal attribute
 */

#include <vector>
#include <string>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/compliant_node_builder.h"
#include "log/log.h"
#include "trilu_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace ge::fusion;

namespace ops {

const std::string kFusionPassName = "TriluFusionPass";
const int64_t kCaptureIdxTriluNode = 0L;

const std::set<std::string> kTriluSupportSocList = {
    "Ascend310B", "Ascend310P", "Ascend910", "Ascend910B", "Ascend910_93", "Ascend950"};

static bool IsSupportSoc()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (unlikely(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS)) {
            OP_LOGE(kFusionPassName.c_str(), "Get platform_info failed.");
            return false; 
        }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (kTriluSupportSocList.count(soc) == 0) {
        OP_LOGE(kFusionPassName.c_str(), "SoC %s is not supported by this fusion pass.", soc.c_str());
        return false;
    }
    return true;
}

static bool GetDiagonalValue(const GNode& triluNode, int32_t& diagonalValue)
{
    if (triluNode.GetInputsSize() < 2) {
        diagonalValue = 0;
        OP_LOGD(kFusionPassName.c_str(), "No k input, set diagonal to default value 0.");
        return true;
    }

    auto srcInfo = triluNode.GetInDataNodesAndPortIndexs(1);
    auto srcNodePtr = srcInfo.first;
    if (srcNodePtr == nullptr) {
        diagonalValue = 0;
        OP_LOGD(kFusionPassName.c_str(), "k input source node is null, set diagonal to default value 0.");
        return true;
    }

    GNode srcNode = *srcNodePtr;
    AscendString nodeType;
    srcNode.GetType(nodeType);
    std::string typeStr = nodeType.GetString();

    if (typeStr != "Const" && typeStr != "Constant") {
        OP_LOGE(kFusionPassName.c_str(), "k input is not a constant, cannot extract diagonal value.");
        return false;
    }

    Tensor tensor;
    if (srcNode.GetAttr("value", tensor) != GRAPH_SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to get value attr from Const node.");
        return false;
    }

    auto tensorDesc = tensor.GetTensorDesc();
    auto shape = tensorDesc.GetShape();
    auto dims = shape.GetDims();

    if (dims.size() > 1 || (dims.size() == 1 && dims[0] != 1)) {
        OP_LOGE(kFusionPassName.c_str(), "Invalid k shape, expected scalar or 1D with size 1.");
        return false;
    }

    DataType dtype = tensorDesc.GetDataType();
    const uint8_t* dataPtr = tensor.GetData();
    if (dataPtr == nullptr) {
        OP_LOGE(kFusionPassName.c_str(), "k tensor data is null.");
        return false;
    }

    if (dtype == DT_INT32) {
        diagonalValue = *reinterpret_cast<const int32_t*>(dataPtr);
    } else if (dtype == DT_INT64) {
        diagonalValue = static_cast<int32_t>(*reinterpret_cast<const int64_t*>(dataPtr));
    } else {
        OP_LOGE(kFusionPassName.c_str(), "k tensor dtype %d not supported, only int32/int64.", dtype);
        return false;
    }

    OP_LOGD(kFusionPassName.c_str(), "Extracted diagonal value: %d", diagonalValue);
    return true;
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OP_LOGD(kFusionPassName.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

static void GetInputsInfo(const std::vector<SubgraphInput> &subGraphInputs, std::vector<Shape> &inputShapes,
    std::vector<DataType> &inputDtpyes, std::vector<Format> &inputFormats)
 {
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNode = subGraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        AscendString nodeType;
        matchNode.node.GetType(nodeType);
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtpyes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

std::vector<PatternUniqPtr> TriluFusionPass::Patterns()
{
    OP_LOGD(kFusionPassName.c_str(), "Enter Patterns for TriluFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder0 = es::EsGraphBuilder("TriluXFusionPass");
    auto x0 = graphBuilder0.CreateInput(0);
    auto output0 = es::Trilu(x0);
    auto graph0 = graphBuilder0.BuildAndReset(std::vector<es::EsTensorHolder>{output0});
    auto pattern0 = std::make_unique<Pattern>(std::move(*graph0));
    pattern0->CaptureTensor({*output0.GetProducer(), 0});  // Capture the Trilu node
    patternGraphs.emplace_back(std::move(pattern0));

    auto graphBuilder1 = es::EsGraphBuilder("TriluXKConstFusionPass");
    auto x1 = graphBuilder1.CreateInput(0);
    auto k0 = graphBuilder1.CreateConst(std::vector<int32_t>{0},std::vector<int64_t>{0});
    auto output1 = es::Trilu(x1, k0);
    auto graph1 = graphBuilder1.BuildAndReset(std::vector<es::EsTensorHolder>{output1});
    auto pattern1 = std::make_unique<Pattern>(std::move(*graph1));
    pattern1->CaptureTensor({*output1.GetProducer(), 0});  // Capture the Trilu node
    patternGraphs.emplace_back(std::move(pattern1));
    return patternGraphs;
}

bool TriluFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kFusionPassName.c_str(), "Enter MeetRequirements for TriluFusionPass");
    auto patternGraph = matchResult->GetPatternGraph();
    AscendString patternName;

    if (patternGraph.GetName(patternName) != GRAPH_SUCCESS){
        OP_LOGE(kFusionPassName.c_str(), "Failed to get patternName.");  
        return false; 
    }

    if (!IsSupportSoc()) {
        OP_LOGE(kFusionPassName.c_str(), "Platform not supported.");
        return false;
    }

    NodeIo triluNodeIo;
    if (unlikely(matchResult->GetCapturedTensor(kCaptureIdxTriluNode, triluNodeIo) != SUCCESS)) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to get captured tensor.");
        return false;
    }
    AscendString nodeType;
    triluNodeIo.node.GetType(nodeType);
    std::string typeStr = nodeType.GetString();
    if (typeStr != "Trilu") {
        OP_LOGE(kFusionPassName.c_str(), "Node type %s is not Trilu, skip.", typeStr.c_str());
        return false;
    }

    int32_t upper = 0;
    if (triluNodeIo.node.GetAttr("upper", upper) != GRAPH_SUCCESS) {
        // Attribute not set, use default value 0
        upper = 0;
        OP_LOGE(kFusionPassName.c_str(), "upper attribute not set, use default value 0.");
    }
    if (upper != 0 && upper != 1) {
        OP_LOGE(kFusionPassName.c_str(), "upper value %d is not 0 or 1, skip.", upper);
        return false;
    }

    // 4. Check and extract diagonal value from k input
    int32_t diagonal = 0;
    if (!GetDiagonalValue(triluNodeIo.node, diagonal)) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to get diagonal value from k input, skip.");
        return false;
    }

    return true;
}

GraphUniqPtr TriluFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kFusionPassName.c_str(), "Enter Replacement for TriluFusionPass");
    AscendString patternName;
    auto patternGraph = matchResult->GetPatternGraph();
    patternGraph.GetName(patternName);

    NodeIo triluNodeIo;
    matchResult->GetCapturedTensor(kCaptureIdxTriluNode, triluNodeIo);
    int32_t upper = 0;
    triluNodeIo.node.GetAttr("upper", upper);
    int32_t diagonal = 0;
    GetDiagonalValue(triluNodeIo.node, diagonal);
    
    std::vector<SubgraphInput> subGraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subGraphInputs);
    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtpyes;
    std::vector<Format> inputFormats;    
    GetInputsInfo(subGraphInputs, inputShapes, inputDtpyes, inputFormats);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto xTensor = replaceGraphBuilder.CreateInput(0, "x", inputDtpyes[0], inputFormats[0], inputShapes[0].GetDims());
    auto res = (upper == 0) ? es::Tril(xTensor, diagonal) : es::Triu(xTensor, diagonal);

    GNode triluNode = *res.GetProducer();
    auto triluNodeFormat = inputFormats[0];
    TensorDesc triluInputDesc;
    triluNode.GetInputDesc(0, triluInputDesc);
    triluInputDesc.SetFormat(triluNodeFormat);
    triluNode.UpdateInputDesc(0, triluInputDesc);
    
    auto replaceGraph = replaceGraphBuilder.BuildAndReset({res});


    if (InferShape(replaceGraph, subGraphInputs) != SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }

    return replaceGraph;
}
REG_FUSION_PASS(TriluFusionPass).Stage(CustomPassStage::kCompatibleInherited);
} // namespace ops