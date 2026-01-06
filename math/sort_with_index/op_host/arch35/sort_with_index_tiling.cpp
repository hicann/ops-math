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
 * \file sort_with_index_tiling.cc
 * \brief sort_with_index ac tiling impl
 */
#include "sort_with_index_tiling.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_util.h"

namespace optiling {
constexpr size_t WORK_SPACE_SIZE = 16777216;  // 16 * 1024 * 1024;
const uint32_t BIN_NUM = 256;
const uint32_t TILE_DATA_NUM = 4096;
const uint32_t TMP_UB = 1024; // 暂预留给sort高级api的大小
const uint32_t CONST_10 = 10;
const uint32_t CONST_1 = 1;
const uint32_t CONST_2 = 2;
const uint32_t CONST_3 = 3;
const uint32_t INT64_BYTE = 8;
const uint32_t INT32_BYTE = 4;
const uint32_t TILE_DATA_NUM_B64 = 2048;
const uint32_t NEED_UB_SIZE_BYTE = 221184; // 预留了32k给simt使用
const uint32_t SMALL_SORT_MAX_DATA_SIZE = 512;
const uint32_t AGLIN_VALUE = 32;
const uint32_t MERGE_SORT_TILING_OFFSET = 10000;
struct SortTileInfo {
    uint32_t coreNumNeed = 0;
    uint32_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t oneCoreRowNum = 1;
};

static const std::map<ge::DataType, uint32_t> tilingDataTypeKeyMap = {
    {ge::DT_INT64, 1004},  {ge::DT_INT32, 1003},   {ge::DT_INT16, 1002},  {ge::DT_INT8, 1001},
    {ge::DT_UINT64, 2004}, {ge::DT_UINT32, 2003},  {ge::DT_UINT16, 2002}, {ge::DT_UINT8, 2001},
    {ge::DT_FLOAT, 3003},  {ge::DT_FLOAT16, 3002}, {ge::DT_BF16, 4002}};
static const std::map<ge::DataType, uint32_t> tilingDataTypeBitMap = {
    {ge::DT_INT64, 8},  {ge::DT_INT32, 4},   {ge::DT_INT16, 2},  {ge::DT_INT8, 1},
    {ge::DT_UINT64, 8}, {ge::DT_UINT32, 4},  {ge::DT_UINT16, 2}, {ge::DT_UINT8, 1},
    {ge::DT_FLOAT, 4},  {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
static const std::map<ge::DataType, uint32_t> optDataTypeBitMap = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};

void SetSortTmpSizeOfIdx(
    ge::DataType dataType, int64_t lastAxisNum, uint32_t tileData, bool isDescend, bool hasIndex,
    SortWithIndexTilingDataSimt& sortTilingData)
{
    int64_t reanLen = std::min(lastAxisNum, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = {reanLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    // SortWithIndex and no need to cut axis
    config.hasSrcIndex = hasIndex && (reanLen == lastAxisNum);
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGW("RadixSortTilingForAscendC", "Allocal buffer element len = %lu ac sort api", reanLen);
    OP_LOGW("RadixSortTilingForAscendC", "Need tmp buffer %u byte for ac sort api", maxValue);
    sortTilingData.set_sortAcApiNeedBufferSize(maxValue);
}

void SetMergeSortTmpSizeOfIdx(
    gert::TilingContext* context, ge::DataType dataType, int64_t lastAxisNum,
    SortWithIndexTilingDataSimt& sortTilingData)
{
    uint32_t reanLen = 0;
    if ((lastAxisNum <= SMALL_SORT_MAX_DATA_SIZE) && (optDataTypeBitMap.count(dataType) != 0)) {
        reanLen = std::min(static_cast<uint32_t>(lastAxisNum), SMALL_SORT_MAX_DATA_SIZE);
    }
    uint32_t aglinDataSize = static_cast<uint32_t>((reanLen + AGLIN_VALUE - 1) / AGLIN_VALUE * AGLIN_VALUE);
    uint32_t dataTypeSize = (dataType == ge::DT_BF16) ? optDataTypeBitMap.find(ge::DT_FLOAT)->second :
                                                        optDataTypeBitMap.find(dataType)->second;
    auto platform_info = context->GetPlatformInfo();
    if (nullptr == platform_info) {
        OP_LOGE("RadixSortTilingForAscendC", "platform_info is nullptr.");
    }
    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, aglinDataSize, dataTypeSize);
    OP_LOGW("RadixSortTilingForAscendC", "Allocal buffer mergesort element len = %u ac sort api", reanLen);
    OP_LOGW("RadixSortTilingForAscendC", "Merge sort need tmp buffer %u byte for ac api", dataSizeNeed);
    sortTilingData.set_mergSortAcApiNeedBufferSize(dataSizeNeed);
}

void TileModeSmallSizeOptimOfIdx(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, uint32_t tileData,
    SortWithIndexTilingDataSimt& sortTilingData, SortTileInfo& sortTileInfo)
{
    uint32_t aglinNum = static_cast<uint32_t>((lastAxisNum + AGLIN_VALUE - 1) / AGLIN_VALUE * AGLIN_VALUE);
    uint32_t oneCoreRowNum = static_cast<uint32_t>((tileData / 2) / aglinNum);
    oneCoreRowNum = static_cast<uint32_t>(oneCoreRowNum == 0 ? 1 : oneCoreRowNum);
    uint32_t virUnsortedDimNum = static_cast<uint32_t>((unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum);
    uint32_t coreNumNeed = 0;
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TileModeSmallSizeOptimOfIdx", "maxCoreNum is zero"), return);
    uint32_t sortLoopTimes = static_cast<uint32_t>((virUnsortedDimNum + maxCoreNum - 1) / maxCoreNum);
    if (sortLoopTimes == 1u) {
        uint32_t realCoreNum = virUnsortedDimNum % maxCoreNum;
        if (realCoreNum == 0u) {
            realCoreNum = maxCoreNum;
        }
        coreNumNeed = realCoreNum;
    } else {
        coreNumNeed = maxCoreNum;
    }
    sortTilingData.set_sortLoopTimes(sortLoopTimes);
    sortTilingData.set_lastDimTileNum(1);
    sortTilingData.set_unsortedDimParallel(coreNumNeed);
    sortTilingData.set_lastDimNeedCore(1);
    sortTilingData.set_numTileDataSize(lastAxisNum);
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.oneCoreRowNum = oneCoreRowNum;
    OP_LOGI("RadixSortTilingForAscendC", "Small size opt mode oneCoreRowNum=%u", oneCoreRowNum);
    OP_LOGI(
        "RadixSortTilingForAscendC", "Small size opt mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        sortLoopTimes, lastAxisNum);
}

void TileModeSmallSizeOfIdx(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, SortWithIndexTilingDataSimt& sortTilingData,
    SortTileInfo& sortTileInfo)
{
    uint32_t coreNumNeed = 0;
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TileModeSmallSizeOfIdx", "maxCoreNum is zero"), return);
    uint32_t sortLoopTimes = static_cast<uint32_t>((unsortedDimNum + maxCoreNum - 1) / maxCoreNum);
    if (sortLoopTimes == 1u) {
        uint32_t realCoreNum = unsortedDimNum % maxCoreNum;
        if (realCoreNum == 0u) {
            realCoreNum = maxCoreNum;
        }
        coreNumNeed = realCoreNum;
    } else {
        coreNumNeed = maxCoreNum;
    }
    sortTilingData.set_sortLoopTimes(sortLoopTimes);
    sortTilingData.set_lastDimTileNum(1);
    sortTilingData.set_unsortedDimParallel(coreNumNeed);
    sortTilingData.set_lastDimNeedCore(1);
    sortTilingData.set_numTileDataSize(static_cast<uint32_t>(lastAxisNum));
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    OP_LOGI(
        "RadixSortTilingForAscendC", "Small size mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        sortLoopTimes, lastAxisNum);
}

void TileModeMediumSizeOfIdx(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, uint32_t tileData,
    SortWithIndexTilingDataSimt& sortTilingData, SortTileInfo& sortTileInfo)
{
    OP_CHECK_IF(tileData == 0, OP_LOGE("TileModeMediumSizeOfIdx", "tileData is zero"), return);
    uint32_t lastDimTileNum = static_cast<uint32_t>((lastAxisNum + tileData - 1) / tileData);
    uint32_t unsortedDimParallel = maxCoreNum / lastDimTileNum;
    uint32_t coreNumNeed = lastDimTileNum * unsortedDimParallel;
    uint32_t sortLoopTimes = static_cast<uint32_t>((unsortedDimNum + unsortedDimParallel - 1) / unsortedDimParallel);
    if (sortLoopTimes == 1u) {
        coreNumNeed = unsortedDimNum * lastDimTileNum;
        unsortedDimParallel = unsortedDimNum;
    }
    sortTilingData.set_sortLoopTimes(sortLoopTimes);
    sortTilingData.set_lastDimTileNum(lastDimTileNum);
    sortTilingData.set_unsortedDimParallel(unsortedDimParallel);
    sortTilingData.set_lastDimNeedCore(lastDimTileNum);
    sortTilingData.set_numTileDataSize(tileData);
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.lastDimTileNum = lastDimTileNum;
    sortTileInfo.unsortedDimParallel = unsortedDimParallel;
    OP_LOGI("RadixSortTilingForAscendC", "Medium size mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        sortLoopTimes, lastAxisNum);
    OP_LOGI("RadixSortTilingForAscendC", "Medium size mode lastDimTileNum=%u unsortedDimParallel=%u lastDimRealCore=%u",
        lastDimTileNum, unsortedDimParallel, lastDimTileNum);
}

void TileModeBigSizeOfIdx(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, uint32_t tileData,
    SortWithIndexTilingDataSimt& sortTilingData, SortTileInfo& sortTileInfo)
{
    OP_CHECK_IF(tileData == 0, OP_LOGE("TileModeBigSizeOfIdx", "tileData is zero"), return);
    uint32_t lastDimTileNum = static_cast<uint32_t>((lastAxisNum + tileData - 1) / tileData);
    uint32_t coreNumNeed = std::min(static_cast<uint32_t>(maxCoreNum), lastDimTileNum);
    sortTilingData.set_sortLoopTimes(unsortedDimNum);
    sortTilingData.set_lastDimTileNum(lastDimTileNum);
    sortTilingData.set_unsortedDimParallel(1);
    sortTilingData.set_lastDimNeedCore(coreNumNeed);
    sortTilingData.set_numTileDataSize(tileData);
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.lastDimTileNum = lastDimTileNum;
    sortTileInfo.unsortedDimParallel = 1u;
    OP_LOGI("RadixSortTilingForAscendC", "Big size mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        unsortedDimNum, lastAxisNum);
    OP_LOGI("RadixSortTilingForAscendC", "Big size mode lastDimTileNum=%u unsortedDimParallel=1 lastDimRealCore=%u",
        lastDimTileNum, coreNumNeed);
}

void PrintTilingDataOfIdx(SortTileInfo& sortTileInfo, SortWithIndexTilingDataSimt& sortTilingData)
{
    OP_LOGI(
        "RadixSortPrintTilingData",
        "coreNum is %u, lastAxisNum is %ld, isInInt32Range is %u, "
        "sortLoopTimes is %u, unsortedDimParallel is %u, unsortedDimNum is %u, "
        "lastDimTileNum is %u, lastDimNeedCore is %u, numTileDataSize is %u, "
        "sortAcApiNeedBufferSize is %u, mergSortAcApiNeedBufferSize is %u, "
        "oneCoreRowNum is %u, outputLastDimValue is %u ",
        sortTileInfo.coreNumNeed, sortTilingData.get_lastAxisNum(), sortTilingData.get_isInInt32Range(),
        sortTilingData.get_sortLoopTimes(), sortTilingData.get_unsortedDimParallel(),
        sortTilingData.get_unsortedDimNum(), sortTilingData.get_lastDimTileNum(), sortTilingData.get_lastDimNeedCore(),
        sortTilingData.get_numTileDataSize(), sortTilingData.get_sortAcApiNeedBufferSize(),
        sortTilingData.get_mergSortAcApiNeedBufferSize(), sortTilingData.get_oneCoreRowNum(),
        sortTilingData.get_outputLastDimValue());
    return;
}

ge::graphStatus RadixSortTilingOfIdx(gert::TilingContext* context, int32_t maxCoreNum)
{
    OP_LOGI("RadixSortTilingForAscendC", "RadixSortTIling start");
    SortWithIndexTilingDataSimt sortTilingData;
    const gert::Shape inputShape = Ops::Math::OpTiling::EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    auto dataType = context->GetInputDesc(0)->GetDataType();
    auto y2DType = context->GetOutputDesc(1)->GetDataType();

    OP_CHECK_IF(tilingDataTypeKeyMap.count(dataType) == 0, OP_LOGE(context->GetNodeName(), "Not support data type"), return ge::GRAPH_FAILED);
    auto tilingKey = tilingDataTypeKeyMap.find(dataType)->second;
    std::string opType(context->GetNodeType());

    OP_LOGW("RadixSortTilingForAscendC", "Get op_type[%s]", opType.c_str());

    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t xDimNum = inputShape.GetDimNum();
    const bool* isDescending = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, isDescending);
    OP_LOGI(context->GetNodeName(), "isDescending=%u", *isDescending);
    int64_t sortAxisNum = inputShape.GetDim(xDimNum - 1);
    uint32_t unSortDimNum = 1;
    for (uint32_t i = 0u; i < static_cast<uint32_t>(xDimNum - 1); i++) {
        unSortDimNum *= inputShape.GetDim(i);
    }

    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint32_t isInInt32Range = static_cast<uint32_t>(sortAxisNum <= int32Max);
    sortTilingData.set_isInInt32Range(isInInt32Range);
    OP_LOGI("[isInInt32Range]", "sortAxisNum: %ld, int32Max: %ld, isInInt32Range: %u", sortAxisNum, int32Max, isInInt32Range);

    SortTileInfo sortTileInfo;
    uint32_t tileData = TILE_DATA_NUM;
    if (dataType == ge::DT_UINT64 || dataType == ge::DT_INT64) {
        tileData = TILE_DATA_NUM_B64;
    } else {
        tileData = TILE_DATA_NUM;
    }
    const uint32_t halfSortedDimData = (tileData * static_cast<uint32_t>(maxCoreNum)) / CONST_2;
    if (sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE && optDataTypeBitMap.count(dataType) != 0) {
        uint32_t tileDataS = TILE_DATA_NUM;
        TileModeSmallSizeOptimOfIdx(unSortDimNum, maxCoreNum, sortAxisNum, tileDataS, sortTilingData, sortTileInfo);
        SetMergeSortTmpSizeOfIdx(context, dataType, sortAxisNum, sortTilingData);
        tilingKey += MERGE_SORT_TILING_OFFSET;
        SetSortTmpSizeOfIdx(dataType, sortAxisNum, tileData, *isDescending, true, sortTilingData);
    } else if (sortAxisNum <= static_cast<int64_t>(tileData)) {
        TileModeSmallSizeOfIdx(unSortDimNum, maxCoreNum, sortAxisNum, sortTilingData, sortTileInfo);
    } else if (sortAxisNum <= static_cast<int64_t>(halfSortedDimData)) {
        TileModeMediumSizeOfIdx(unSortDimNum, maxCoreNum, sortAxisNum, tileData, sortTilingData, sortTileInfo);
    } else {
        TileModeBigSizeOfIdx(unSortDimNum, maxCoreNum, sortAxisNum, tileData, sortTilingData, sortTileInfo);
    }
    SetSortTmpSizeOfIdx(dataType, sortAxisNum, tileData, *isDescending, true, sortTilingData);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(sortTileInfo.coreNumNeed);
    context->SetScheduleMode(1);
    sortTilingData.set_isDescend(*isDescending);
    sortTilingData.set_lastAxisNum(sortAxisNum);
    sortTilingData.set_unsortedDimNum(unSortDimNum);
    sortTilingData.set_oneCoreRowNum(sortTileInfo.oneCoreRowNum);
    sortTilingData.set_outputLastDimValue(sortAxisNum);
    sortTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(sortTilingData.GetDataSize());

    size_t excusiveBinsGmSize = BIN_NUM * tilingDataTypeBitMap.find(dataType)->second * sortTileInfo.unsortedDimParallel * sizeof(int64_t);
    size_t globalHistGmSize = BIN_NUM * sortTileInfo.lastDimTileNum * sortTileInfo.unsortedDimParallel * sizeof(int64_t);
    if (isInInt32Range) {
        excusiveBinsGmSize = BIN_NUM * tilingDataTypeBitMap.find(dataType)->second * sortTileInfo.unsortedDimParallel * sizeof(int32_t);
        globalHistGmSize = BIN_NUM * sortTileInfo.lastDimTileNum * sortTileInfo.unsortedDimParallel * sizeof(int32_t);
    }
    size_t valueIndexBufferSize = sortAxisNum * tilingDataTypeBitMap.find(y2DType)->second;
    size_t valueBufferSize = sortAxisNum * tilingDataTypeBitMap.find(dataType)->second;
    size_t usrSize = globalHistGmSize + excusiveBinsGmSize + sortTileInfo.unsortedDimParallel * (valueIndexBufferSize + valueBufferSize);

    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = usrSize + WORK_SPACE_SIZE;
    context->SetLocalMemorySize(NEED_UB_SIZE_BYTE);
    PrintTilingDataOfIdx(sortTileInfo, sortTilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortWithIndexTilingSimt(gert::TilingContext* context, int32_t maxCoreNum)
{
    return RadixSortTilingOfIdx(context, maxCoreNum);
}

static ge::graphStatus Tiling4SortWithIndex(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const SortWithIndexCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    OP_LOGD(context->GetNodeName(), "AscendC SortWithIndex simt tiling");
    OP_CHECK_IF(SortWithIndexTilingSimt(context, compile_info->core_num) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "The simt tiling function failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SortWithIndex(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "AscendC Tiling starting GRAPH_SUCCESS");
    auto compileInfo = context->GetCompiledInfo<SortWithIndexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->core_num <= 0),
        OP_LOGE(context->GetNodeName(), "The core num is invaild."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SortWithIndex).Tiling(Tiling4SortWithIndex).TilingParse<SortWithIndexCompileInfo>(TilingPrepare4SortWithIndex);
} // namespace optiling
