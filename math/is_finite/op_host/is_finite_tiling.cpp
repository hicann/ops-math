/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file is_finite_tiling.cpp
 * \brief
 */
#include "is_finite_tiling.h"
#include "is_finite_tiling_arch35.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

using namespace ge;

namespace optiling {
constexpr uint32_t DATA_BLOCK = 32;
constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class IsFiniteTiling
{
public:
    explicit IsFiniteTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();

private:
    uint8_t GetDataTypeSize() const;
    uint64_t GetTilingKeyVal() const;

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) const;
    uint32_t GetUsableUbMemory(uint64_t ubSizePlatForm);
    void AssignDataToEachCore();
    void FillTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;
    IsFiniteTilingData tilingData;

    uint8_t dataBlockSize = 0;

    uint64_t totalDataCount = 1;
    uint32_t usableUbSize = 0;
    uint32_t needCoreNum = 0;
    uint64_t perCoreDataCount = 0;
    uint64_t tailDataCoreNum = 0;
    uint64_t lastCoreDataCount = 0;
};

ge::graphStatus IsFiniteTiling::RunBigKernelTiling()
{
    auto compileInfo = reinterpret_cast<const IsFiniteCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(tilingContext, "compileInfo == nullptr"), return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm = compileInfo->ubSize;
    uint64_t coreNumPlatForm = compileInfo->totalCoreNum;

    // Get dtype information, and the total number of data.
    auto tempInputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(tempInputDesc == nullptr, OP_LOGE(tilingContext, "InputDesc == nullptr"), return ge::GRAPH_FAILED);
    dataType = tempInputDesc->GetDataType();
    uint8_t dataTypeSize = GetDataTypeSize();
    dataBlockSize = DATA_BLOCK * dataTypeSize;
    const gert::StorageShape* shape = tilingContext->GetInputShape(0);
    OP_CHECK_IF(shape == nullptr, OP_LOGE(tilingContext, "InputShape == nullptr"), return ge::GRAPH_FAILED);
    for (uint16_t i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
        totalDataCount *= shape->GetStorageShape().GetDim(i);
    }

    usableUbSize = GetUsableUbMemory(ubSizePlatForm);
    needCoreNum = GetNeedCoreNum(coreNumPlatForm);

    AssignDataToEachCore();
    FillTilingData();
    tilingContext->SetBlockDim(needCoreNum);

    tilingContext->SetTilingKey(GetTilingKeyVal());
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = WORK_SPACE_SIZE;
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint8_t IsFiniteTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t IsFiniteTiling::GetTilingKeyVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_BF16:
            return TILING_KEY_BFLOAT16;
        default:
            return 0;
    }
}

uint32_t IsFiniteTiling::GetNeedCoreNum(uint32_t coreNumPlatform) const
{
    uint32_t tempCoreNum = static_cast<uint32_t>(Ops::Base::CeilDiv(totalDataCount, static_cast<uint64_t>(DATA_BLOCK)));
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void IsFiniteTiling::AssignDataToEachCore()
{
    perCoreDataCount = totalDataCount / needCoreNum;
    perCoreDataCount = perCoreDataCount / DATA_BLOCK * DATA_BLOCK;
    uint64_t tempTailDataCount = totalDataCount - perCoreDataCount * needCoreNum;
    tailDataCoreNum = tempTailDataCount / DATA_BLOCK;
    lastCoreDataCount = perCoreDataCount + tempTailDataCount % DATA_BLOCK;
}

uint32_t IsFiniteTiling::GetUsableUbMemory(uint64_t ubSizePlatForm)
{
    // The remaining UB size is split in two, double buffer enabled, input and output, and rounded down 32 data.
    uint32_t canUseUbSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize()) / UB_DIVIDER_FOR_TEMP_CASTING;
    canUseUbSize = canUseUbSize / dataBlockSize * dataBlockSize;
    return canUseUbSize;
}

void IsFiniteTiling::FillTilingData()
{
    tilingData.set_totalDataCount(totalDataCount);
    tilingData.set_usableUbSize(usableUbSize);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_perCoreDataCount(perCoreDataCount);
    tilingData.set_tailDataCoreNum(tailDataCoreNum);
    tilingData.set_lastCoreDataCount(lastCoreDataCount);
}

static ge::graphStatus TilingPrepare4IsFiniteTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<IsFiniteCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    auto socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->isRegbase = (socVersion == platform_ascendc::SocVersion::ASCEND910_95) ? true : false;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "IsFinite GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
            compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingIsFiniteTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IsFiniteTiling", "Tiling context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Entering TilingIsFiniteTiling");
    auto compileInfo = reinterpret_cast<const IsFiniteCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    if (compileInfo->isRegbase) {
        OP_LOGD(context, "Entering IsFiniteRegbaseTiling");
        IsFiniteRegbaseTiling isFiniteOpTiling(context);
        return isFiniteOpTiling.RunTiling();
    }
    IsFiniteTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(IsFinite).Tiling(TilingIsFiniteTiling).TilingParse<IsFiniteCompileInfo>(TilingPrepare4IsFiniteTiling);
} // namespace optiling