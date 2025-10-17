/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_lora_tiling.cpp
 * \brief
 */

#include "add_lora_tiling.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"

namespace {
constexpr uint64_t WORKSIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t SOC_TILINGKEY_OFFSET = 100000UL;
constexpr uint32_t VECTORDOUBLE = 2;
constexpr uint32_t CUBESIZE = 16;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;
constexpr uint32_t INDEXFOUR = 4;
constexpr uint32_t INDEXFIVE = 5;
struct AddLoraCompileInfo {};
} // namespace

namespace optiling {
void AddLoraTiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

ge::graphStatus AddLoraTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("AddLora", "context is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool AddLoraTiling::IsCapable()
{
    return true;
}

bool AddLoraTiling::CheckTilingValidation()
{
    OP_CHECK_IF(
        addLoraFlag && tilingData_.get_R() > MAX_RANK_SIZE,
        OP_LOGE(context_->GetNodeName(), "AddLora rank size exceeds limit 128."), return false);

    OP_CHECK_IF(
        tilingData_.get_wBatch() > MAX_WEIGHT_NUM,
        OP_LOGE(context_->GetNodeName(), "AddLora weight num exceeds limit 32."), return false);

    OP_CHECK_IF(
        tilingData_.get_H1() % CUBESIZE != 0, OP_LOGE(context_->GetNodeName(), "AddLora H1 should be align to 16."),
        return false);

    OP_CHECK_IF(
        tilingData_.get_H2() % CUBESIZE != 0, OP_LOGE(context_->GetNodeName(), "AddLora H2 should be align to 16."),
        return false);

    OP_CHECK_IF(
        tilingData_.get_R() % CUBESIZE != 0, OP_LOGE(context_->GetNodeName(), "AddLora R should be align to 16."),
        return false);

    return true;
}

ge::graphStatus AddLoraTiling::DoOpTiling()
{
    OP_CHECK_IF(
        !SetPlatformInfoForTiling(), OP_LOGE(context_->GetNodeName(), "SetPlatformInfoForTiling fail"),
        return ge::GRAPH_FAILED);
    auto wa_Ptr = context_->GetOptionalInputShape(INDEXFOUR);
    uint32_t wBatch = GetShape(INDEXTWO).GetDim(INDEXZERO);
    if (wa_Ptr) {
        if (wa_Ptr->GetStorageShape().GetDim(INDEXZERO) == wBatch) {
            addLoraFlag = static_cast<uint32_t>(1);
        }
    }
    uint32_t batch = static_cast<uint32_t>(GetShape(INDEXZERO).GetDim(INDEXZERO));
    uint32_t H2 = GetShape(INDEXTWO).GetDim(INDEXTWO);
    uint32_t y_column = GetShape(INDEXZERO).GetDim(INDEXONE);
    uint32_t H1 = static_cast<uint32_t>(GetShape(INDEXONE).GetDim(INDEXONE));
    uint32_t layer = GetShape(INDEXTWO).GetDim(INDEXONE);
    uint32_t R = GetShape(INDEXTWO).GetDim(INDEXTHREE);
    uint32_t maxCore = static_cast<uint32_t>(coreNum_) * VECTORDOUBLE;
    if (isAscend310P_) {
        maxCore = static_cast<uint32_t>(coreNum_);
    }

    uint32_t usedCoreNum = static_cast<uint32_t>(coreNum_);
    uint32_t taskNumPerCore = batch / maxCore;
    uint32_t H2PerCore = Ops::Base::CeilDiv(H2, CUBESIZE) / maxCore * CUBESIZE;
    tilingData_.set_layer(layer);
    tilingData_.set_batch(batch);
    tilingData_.set_H2(H2);
    tilingData_.set_H1(H1);
    tilingData_.set_wBatch(wBatch);
    tilingData_.set_usedCoreNum(usedCoreNum);
    tilingData_.set_R(R);
    auto layer_idx = *context_->GetAttrs()->GetAttrPointer<int>(INDEXZERO);
    auto scale = *context_->GetAttrs()->GetAttrPointer<float>(INDEXONE);
    auto y_offset = *context_->GetAttrs()->GetAttrPointer<int>(INDEXTWO);
    auto y_slice_size = *context_->GetAttrs()->GetAttrPointer<int>(INDEXTHREE);
    tilingData_.set_layer_idx(layer_idx);
    tilingData_.set_scale(scale);
    tilingData_.set_y_offset(y_offset);
    tilingData_.set_y_slice_size(y_slice_size);
    tilingData_.set_taskNumPerCore(taskNumPerCore);
    tilingData_.set_H2PerCore(H2PerCore);
    tilingData_.set_addLoraFlag(addLoraFlag);
    tilingData_.set_y_column(y_column);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::DoLibApiTiling()
{
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t AddLoraTiling::GetTilingKey() const
{
    uint64_t tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_DEFAULT_SCENE);
    uint32_t batch = context_->GetInputShape(0)->GetStorageShape().GetDim(0);
    if (batch <= static_cast<uint32_t>(coreNum_)) {
        tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_SPARSE_SCENE);
    }

    uint32_t socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_910);
    if (isAscend310P_) {
        socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_310);

        if (!addLoraFlag) {
            tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_BGMV_SCENE);
        }
    }

    tilingKey += static_cast<uint64_t>(socVersionFlag) * SOC_TILINGKEY_OFFSET;
    return tilingKey;
}

ge::graphStatus AddLoraTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t sysWorkspaceSize = WORKSIZE;
    /* usrWorkspaceSize = workspace for other gm */
    size_t usrWorkspaceSize =
        tilingData_.get_batch() * tilingData_.get_H1() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * tilingData_.get_R() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * tilingData_.get_H2() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * sizeof(uint32_t) * 3 + 1024 * 1024;
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::PostTiling()
{
    OP_CHECK_IF(
        !CheckTilingValidation(), OP_LOGE(context_->GetNodeName(), "check tiling data validation failed."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "final tiling data size: %zu", tilingData_.GetDataSize());
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size [%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    auto blockDim = tilingData_.get_usedCoreNum();
    context_->SetBlockDim(blockDim);
    context_->SetTilingKey(GetTilingKey());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool AddLoraTiling::SetPlatformInfoForTiling()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAic();
    isAscend310P_ = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;

    return true;
}

REGISTER_TILING_TEMPLATE("AddLora", AddLoraTiling, 0);

const gert::Shape AddLoraTiling::GetShape(const size_t index)
{
    return context_->GetInputShape(index)->GetStorageShape();
}

void AddLoraTiling::PrintTilingData()
{
    std::stringstream ss;
    ss << " layer: " << tilingData_.get_layer() << " batch: " << tilingData_.get_batch()
       << " H2: " << tilingData_.get_H2() << " H1: " << tilingData_.get_H1() << " wBatch: " << tilingData_.get_wBatch()
       << " usedCoreNum: " << tilingData_.get_usedCoreNum() << " R: " << tilingData_.get_R()
       << " layer_idx: " << tilingData_.get_layer_idx() << "scale: " << tilingData_.get_scale()
       << " y_offset: " << tilingData_.get_y_offset() << "y_slice_size:  " << tilingData_.get_y_slice_size();
    OP_LOGD(context_->GetNodeName(), "api tiling: %s", ss.str().c_str());
}

static ge::graphStatus AddLoraTilingFunc(gert::TilingContext* context)
{
    AddLoraTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForAddLora([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddLora).Tiling(AddLoraTilingFunc).TilingParse<AddLoraCompileInfo>(TilingParseForAddLora);
} // namespace optiling