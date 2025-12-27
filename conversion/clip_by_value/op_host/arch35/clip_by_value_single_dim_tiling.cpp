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
 * \file clip_by_value_single_dim_tiling.cpp
 * \brief
 */
#include <vector>
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "clip_by_value_tiling.h"
#include "clip_by_value_single_dim_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include <string>

using namespace ge;
using namespace std;
namespace ClipByValueSigDim {
constexpr int64_t INOUT_PARAM_NUM = 4;
constexpr int64_t X_INDEX = 0;
constexpr int64_t MIN_INDEX = 1;
constexpr int64_t MAX_INDEX = 2;
constexpr int64_t Y_INDEX = 3;

constexpr uint64_t OPEN_DB_SIZE = 2;
constexpr uint64_t HALF_CORE_NUM_DIVIDE = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t INT64_BYTES = 8;

constexpr int32_t ASS_ALIVE_NODE_NUM = 2;
constexpr int32_t SIG_DIM_ALIVE_NODE_NUM = 4;

constexpr int64_t PER_CORE_MIN_UB_BYTE = 8 * 1024;

constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
} // namespace ClipByValueSigDim

namespace optiling {

static constexpr uint64_t CLIP_BY_VALUE_SINGLE_DIM_TILING_KEY = 2000100;
static constexpr uint64_t CLIP_BY_VALUE_ASS_TILING_KEY = 2001100;

static constexpr uint64_t CLIP_BY_VALUE_SINGLE_DIM_TILING_PRIORITY = 1;

string ClipByValueTensorDesc2String(const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
    oss << "(shape:" << Ops::Base::ToString(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << Ops::Base::ToString(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToSerialString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";

    return oss.str();
}

string ClipByValueDebugTilingContext(gert::TilingContext* context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << ClipByValueTensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << ClipByValueTensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

bool ClipByValueTilingSingleDim::IsCapable()
{
    return isSigDim;
}

ge::graphStatus ClipByValueTilingSingleDim::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const ClipByValueCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    opName = context_->GetNodeName();
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        coreNum = static_cast<int32_t>(compileInfo->coreNum);
        ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "TilingContext: %s", optiling::ClipByValueDebugTilingContext(context_).c_str());
    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype = xDesc->GetDataType();

    auto minDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minDesc);
    auto minDtype = minDesc->GetDataType();

    auto maxDesc = context_->GetInputDesc(2); // 2 is max param
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxDesc);
    auto maxDtype = maxDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    if (xDtype != minDtype || xDtype != maxDtype || xDtype != yDtype) {
        OP_LOGE(
            context_->GetNodeName(), "xDataType: %d minDataType: %d maxDataType: %d yDataType: %d check failed.",
            xDtype, minDtype, maxDtype, yDtype);
        return ge::GRAPH_FAILED;
    }

    if (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_INT32 && xDtype != ge::DataType::DT_FLOAT16 &&
        xDtype != ge::DataType::DT_BF16 && xDtype != ge::DataType::DT_INT64) {
        OP_LOGE(context_->GetNodeName(), "dataType: %d Not supported datatype.", xDtype);
        return ge::GRAPH_FAILED;
    }
    dTypeSize = ge::GetSizeByDataType(xDtype);

    auto xShape = context_->GetInputShape(ClipByValueSigDim::X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::Base::EnsureNotScalar(xShape->GetStorageShape());

    auto minShape = context_->GetInputShape(ClipByValueSigDim::MIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minShape);
    auto minStorageShape = Ops::Base::EnsureNotScalar(minShape->GetStorageShape());

    auto maxShape = context_->GetInputShape(ClipByValueSigDim::MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxShape);
    auto maxStorageShape = Ops::Base::EnsureNotScalar(maxShape->GetStorageShape());

    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::Base::EnsureNotScalar(yShape->GetStorageShape());

    std::vector<gert::Shape> inShapes;
    inShapes.push_back(xStorageShape);
    inShapes.push_back(minStorageShape);
    inShapes.push_back(maxStorageShape);
    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;
    ge::graphStatus res = Ops::Base::DimensionCollapse(inShapes, yStorageShape, dims, strides);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "DimensionCollapse failed."), return res);

    OP_CHECK_IF(
        (dims.size() != ClipByValueSigDim::INOUT_PARAM_NUM),
        OP_LOGE(context_->GetNodeName(), "DimensionCollapse failed. check dims is illegal. dim num: %lu", dims.size()),
        return ge::GRAPH_FAILED);

    isSigDim = false;
    if (dims[ClipByValueSigDim::X_INDEX].size() == 1 && dims[ClipByValueSigDim::MIN_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::MAX_INDEX].size() == 1 && dims[ClipByValueSigDim::Y_INDEX].size() == 1) {
        isSigDim = true;
        xDim = dims[ClipByValueSigDim::X_INDEX].front();
        minDim = dims[ClipByValueSigDim::MIN_INDEX].front();
        maxDim = dims[ClipByValueSigDim::MAX_INDEX].front();
        yDim = dims[ClipByValueSigDim::Y_INDEX].front();

        isAss = false;
        if (minDim == 1 && maxDim == 1) {
            isAss = true;
        }
    }

    OP_LOGD(
        context_->GetNodeName(), "isSigDim: %d isAss: %d xDim: %ld minDim: %ld maxDim: %ld yDim: %ld", isSigDim, isAss,
        xDim, minDim, maxDim, yDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::DoOpTiling()
{
    // 先分ub，再分核
    // 开DB，cacheline 对齐
    int64_t dbUbSize = ubSize / ClipByValueSigDim::OPEN_DB_SIZE;
    int32_t aliveNum = isAss ? ClipByValueSigDim::ASS_ALIVE_NODE_NUM : ClipByValueSigDim::SIG_DIM_ALIVE_NODE_NUM;
    int64_t ubFormerByte = dbUbSize / aliveNum;
    int64_t ubFormerByteFloorAlign = (ubFormerByte / cacheline) * cacheline;
    int64_t ubFormer = ubFormerByteFloorAlign / dTypeSize;
    OP_CHECK_IF(
        (ubFormer == 0),
        OP_LOGE(
            context_->GetNodeName(), "ub size or cacheline check failed. ubSize: %lu cacheline: %lu dTypeSize: %u",
            ubSize, cacheline, dTypeSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (coreNum == 0), OP_LOGE(context_->GetNodeName(), "core num check is 0, check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (yDim == 0), OP_LOGE(context_->GetNodeName(), "tensor check is empty, check failed"), return ge::GRAPH_FAILED);

    // 为方便kernel计算，减少kernel scalar，对于整除无尾场景，也认为有个整数的尾块
    int64_t ubOuter = (yDim + ubFormer - 1) / ubFormer;
    int64_t ubTail = yDim % ubFormer;
    ubTail = (ubTail == 0) ? ubFormer : ubTail;
    int64_t blockFormer = (ubOuter + coreNum - 1) / coreNum;
    int64_t blockTail = ubOuter % blockFormer;
    blockTail = (blockTail == 0) ? blockFormer : blockTail;
    int64_t blockNum = (ubOuter + blockFormer - 1) / blockFormer;

    // 如果blockNum小于核数一半，则尝试分配到核数的一半
    if (static_cast<uint64_t>(blockNum) < (coreNum / ClipByValueSigDim::HALF_CORE_NUM_DIVIDE) &&
        (ubFormer * dTypeSize * aliveNum) > ClipByValueSigDim::PER_CORE_MIN_UB_BYTE) {
        int64_t dimPerCore = yDim * 2 / coreNum; // 1/2 的coreNum进行分配
        // 向上对齐到128Byte，理论上不会超过原来计算的ubFormer
        int64_t alignDimPerCore = ((((dimPerCore * dTypeSize) + cacheline - 1) / cacheline) * cacheline) / dTypeSize;
        ubFormer = (ubFormer > alignDimPerCore) ? alignDimPerCore : ubFormer;

        // 如果分配到核数一半后，小于8k(开DB后)，则直接按照 8k 计算。进入该分支，说明前面已经判断过开db后ub大小大于8KB
        int64_t lowestUbFormer =
            (((ClipByValueSigDim::PER_CORE_MIN_UB_BYTE / aliveNum) / cacheline) * cacheline) / dTypeSize;
        if (ubFormer < lowestUbFormer) {
            ubFormer = lowestUbFormer;
        }

        // 重新计算分核参数
        ubOuter = (yDim + ubFormer - 1) / ubFormer;
        ubTail = yDim % ubFormer;
        ubTail = (ubTail == 0) ? ubFormer : ubTail;
        blockFormer = (ubOuter + coreNum - 1) / coreNum;
        blockTail = ubOuter % blockFormer;
        blockTail = (blockTail == 0) ? blockFormer : blockTail;
        blockNum = (ubOuter + blockFormer - 1) / blockFormer;
    }

    tilingData.set_blockNum(blockNum);
    tilingData.set_ubFormer(ubFormer);
    tilingData.set_ubTail(ubTail);
    tilingData.set_blockFormer(blockFormer);
    tilingData.set_blockTail(blockTail);
    tilingData.set_xDim(xDim);
    tilingData.set_minDim(minDim);
    tilingData.set_maxDim(maxDim);
    tilingData.set_yDim(yDim);

    OP_LOGI(
        context_->GetNodeName(),
        "ClipByValue do tiling finish. coreNum: %u ubSize: %lu cacheline: %lu "
        "blockNum: %ld ubFormer: %ld ubTail: %ld blockFormer: %ld blockTail: %ld "
        "xDim: %ld minDim: %ld maxDim: %ld yDim: %ld",
        coreNum, ubSize, cacheline, blockNum, ubFormer, ubTail, blockFormer, blockTail, xDim, minDim, maxDim, yDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTilingSingleDim::GetTilingKey() const
{
    if (isAss) {
        return CLIP_BY_VALUE_ASS_TILING_KEY;
    } else {
        return CLIP_BY_VALUE_SINGLE_DIM_TILING_KEY;
    }
}

ge::graphStatus ClipByValueTilingSingleDim::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = ClipByValueSigDim::MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData.get_blockNum());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ClipByValue, ClipByValueTilingSingleDim, CLIP_BY_VALUE_SINGLE_DIM_TILING_PRIORITY);

} // namespace optiling