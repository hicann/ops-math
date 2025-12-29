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
 * \file diag_part_tiling.cc
 * \brief tiling for DiagPart
 */

#include "diag_part_tiling_arch35.h"

namespace optiling {
ge::graphStatus DiagPartTiling::Init()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init DiagPartTiling.");
    auto compileInfo = reinterpret_cast<const DiagCompileInfo*>(tilingContext_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    coreNum_ = compileInfo->core_num;
    OP_CHECK_IF(
        (coreNum_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);
    ubSize_ = compileInfo->ub_size;
    OP_CHECK_IF(
        (ubSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_->GetNodeName(), "Init DiagPartTiling sucess.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DiagPartTiling::RunDiagPartTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start RunDiagPartTiling.");

    OP_CHECK_IF(
        DiagPartVerifying() != ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext_->GetNodeName(), "DiagPartTiling failed to verify params!"), return ge::GRAPH_FAILED);

    return RunDiagPartGatherTiling();
}

ge::graphStatus DiagPartTiling::DiagPartVerifying()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DiagPartVerifying.");
    auto xStorageShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xStorageShape);
    auto xShape = xStorageShape->GetStorageShape();
    auto xDimNum = xShape.GetDimNum();

    // limit input dim > 0 and dim % 2 == 0
    OP_CHECK_IF(
        (xDimNum <= 0 || (xDimNum % TWO) != 0),
        OP_LOGE(
            tilingContext_->GetNodeName(),
            "Invalid x shape dim num, it should be an even number and greater than 0, but dim num is %lu.", xDimNum),
        return ge::GRAPH_FAILED);

    // limit the dimensions corresponding to the half and half of the input shape are the same
    for (uint64_t i = 0; i < xDimNum / TWO; i++) {
        OP_CHECK_IF(
            (xShape.GetDim(i) != xShape.GetDim(i + xDimNum / TWO)),
            OP_LOGE(
                tilingContext_->GetNodeName(),
                "Invalid x shape, dimension:%lu and dimension:%lu should be equal, but got %ld and %ld.", i,
                i + xDimNum / TWO, xShape.GetDim(i), xShape.GetDim(xDimNum / TWO)),
            return ge::GRAPH_FAILED);
        sideLength_ *= xShape.GetDim(i);
    }

    // check the validity of the output shape.
    auto yStorageShape = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yStorageShape);
    auto yShape = yStorageShape->GetStorageShape();
    auto yDimNum = yShape.GetDimNum();
    OP_CHECK_IF(
        (yDimNum != xDimNum / TWO),
        OP_LOGE(
            tilingContext_->GetNodeName(),
            "Invalid y shape dim num, it should be equal to half of the dim num of the x shape, but got %lu.", yDimNum),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < yDimNum; i++) {
        OP_CHECK_IF(
            (xShape.GetDim(i) != yShape.GetDim(i)),
            OP_LOGE(
                tilingContext_->GetNodeName(),
                "Invalid y shape, x and y dimension:%lu should be equal, but got %ld and %ld.", i, xShape.GetDim(i),
                yShape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(tilingContext_->GetNodeName(), "DiagPartVerifying sucess.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DiagPartTiling::RunDiagPartGatherTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start RunDiagPartTiling.");
    tilingData_.set_sideLength(sideLength_);
    auto xDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    auto dataType = xDesc->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t sideLengthFactor = 1;
    switch (typeSize) {
        case B8_BYTES:
            sideLengthFactor = B8_SPLIT_ELEMENT;
            break;
        case B16_BYTES:
            sideLengthFactor = B16_SPLIT_ELEMENT;
            break;
        case B32_BYTES:
            sideLengthFactor = B32_SPLIT_ELEMENT;
            break;
        case B64_BYTES:
            sideLengthFactor = B64_SPLIT_ELEMENT;
            break;
        default:
            break;
    }

    int64_t blockNum = Ops::Base::CeilDiv(sideLength_, sideLengthFactor);
    blockNum_ = blockNum;
    if (blockNum >= coreNum_) {
        tilingData_.set_realCoreNum(coreNum_);
    } else if (blockNum == 0) { // Empty Tensor
        tilingData_.set_realCoreNum(1);
    } else {
        tilingData_.set_realCoreNum(blockNum);
    }
    // main block and tail block side length
    tilingData_.set_numPerCore(sideLengthFactor);
    tilingData_.set_tailNum(sideLength_ % sideLengthFactor);
    tilingData_.set_ubSize(ubSize_);

    SetTilingData();
    PrintTilingData();
    OP_LOGD(tilingContext_->GetNodeName(), "RunDiagPartTiling success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DiagPartTiling::SetTilingData()
{
    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    tilingContext_->SetBlockDim(tilingData_.get_realCoreNum());
    tilingContext_->SetTilingKey(TILING_KEY_GATHER);
    if (blockNum_ > coreNum_) {
        tilingContext_->SetTilingKey(TILING_KEY_SIMT);
    }

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

void DiagPartTiling::PrintTilingData()
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is sideLength:%ld, \
            realCoreNum:%ld, numPerCore:%ld, tailNum:%ld",
        tilingData_.get_sideLength(), tilingData_.get_realCoreNum(), tilingData_.get_numPerCore(),
        tilingData_.get_tailNum());
}

ge::graphStatus DiagPartTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start DiagPartTilingForAscendC.");
    DiagPartTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunDiagPartTiling();
}
} // namespace optiling