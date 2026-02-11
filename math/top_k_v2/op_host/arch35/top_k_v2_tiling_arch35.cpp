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
 * \file top_k_v2_tiling_arch35.cpp
 * \brief top_k_v2 impl
 */
#include "top_k_v2_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {
namespace topkV2DataInfo {
const uint32_t CONST_TWO = 2;
const uint32_t CONST_THREE = 3;
const uint32_t MAX_K_FOR_INT64 = 2000;
const uint32_t BIN_NUM = 256;
const uint32_t TILE_SIZE_DECREASING_FACTOR = 32;
const uint32_t TMP_DATA_NUM = 7680;     // 默认UB一次性能处理的非64位数据的个数，可根据场景动态调整
const uint32_t TMP_DATA_NUM_B64 = 5120; // 默认UB一次性能处理的64位数据的个数，可根据场景动态调整
const uint64_t AGLIN_FACTOR = 32;
const uint32_t SMALL_MAX_DATA_SZIE = 1024;
const uint32_t MERGE_SORT_TILING_OFFSET = 10000;
const uint32_t SINGLE_CORE_MODE = 1;
const uint32_t MULT_CORE_MODE = 2;
const uint32_t MULT_CORE_OPTIM_MODE = 4;
const uint32_t SINGLE_BLOCK_MODE = 3;
const uint32_t INT64_BYTE = 8;
const uint32_t INT32_BYTE = 4;
const uint32_t SINGLE_CORE_THRESHOLD =
    10000000; // 网络case，测试结果发现singlecore时间为2600，而老模板性能为9000，因此调整走SingleCore的阈值
const uint32_t CONST_SIMT_SPACE = 32768; // 获取到的UB大小需要预留32KB给simt
const uint32_t SUPPORT_SORT_MAX_BYTE_SIZE = 8000;

constexpr size_t SYS_WORK_SPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
struct TopkTileInfo {
    uint32_t coreNumNeed = 0;
    int64_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t ubRealLoadDataNum = 0;
    uint32_t oneCoreRowNum = 1;
    uint32_t batchNumInUb = 1;
    uint32_t tailLoopBatchNum = 0;
    uint32_t tailBatchNum = 0;
    uint32_t tailTileNum = 0;
    int64_t topKOutLastAxisNum = 0;
};
struct TopkComputingNowTileSizeInfo {
    ge::DataType dataType;
    ge::DataType indicesDType;
    bool isLargest = true;
    bool isSort = true;
    bool isInInt32Range = true;
    int64_t lastAxisNum = 0;
    int64_t kValue = 0;
    uint32_t maxCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t ubBlockAlignSize = 0; // ub的对齐数值，当前为32
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
static const std::map<ge::DataType, uint32_t> b64DataTypeBitMap = {{ge::DT_INT64, 8}, {ge::DT_UINT64, 8}};
} // namespace topkV2DataInfo

ge::graphStatus GetTopkApiTmpBufferSize(
    gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, uint32_t needDataNum, int64_t kValue,
    bool isLargest, ge::DataType dtype, bool isSort, uint32_t nowTileSize)
{
    uint32_t maxBufferSize = 0;
    uint32_t minBufferSize = 0;
    uint32_t aglinInnerValue = ((needDataNum + topkV2DataInfo::AGLIN_FACTOR - 1) / topkV2DataInfo::AGLIN_FACTOR) *
                               topkV2DataInfo::AGLIN_FACTOR;
    uint32_t aglinKValue = 0;
    if (topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) {
        aglinKValue = std::min(static_cast<int64_t>(needDataNum), kValue);
    } else {
        aglinKValue = std::min(static_cast<int64_t>(nowTileSize), kValue);
    }
    AscendC::TopKConfig topkConfig;
    topkConfig.algo = AscendC::TopKAlgo::RADIX_SELECT;
    topkConfig.order = AscendC::TopKOrder::UNSET;
    topkConfig.sorted = isSort;
    bool isSuccess = AscendC::GetTopKMaxMinTmpSize(
        aglinInnerValue, 1, aglinKValue, false, false, AscendC::TopKMode::TOPK_NORMAL, isLargest, dtype, topkConfig,
        maxBufferSize, minBufferSize);
    OP_LOGI("TopKV2TilingForAscendC", "Need tmp buffer %u byte for ac sort topk api", maxBufferSize);
    OP_LOGI(
        "TopKV2TilingForAscendC", "Init kValue=%ld aglinKValue=%u aglinInnerValue=%u", kValue, aglinKValue,
        aglinInnerValue);
    OP_CHECK_IF(
        false == isSuccess, OP_LOGE(context->GetNodeName(), "Get topk api temp buffer fail"), return ge::GRAPH_FAILED);
    topkTilingData.set_topkAcApiTmpBufferSize(maxBufferSize);
    return ge::GRAPH_SUCCESS;
}

uint32_t GetTopkTempBuffer(
    int64_t lastAxisNum, uint32_t tileData, uint32_t maxCoreNum, uint32_t xDtypeSize, uint32_t indexToDtypeSize,
    uint32_t indexDtypeSize, int64_t kValue, uint64_t ubSizePlatForm)
{
    OP_CHECK_IF(tileData == 0, OP_LOGE("TopkV2", "tileData is 0"), return ge::GRAPH_FAILED);
    uint32_t lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + tileData - 1) / tileData;
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TopkV2", "maxCoreNum is 0"), return ge::GRAPH_FAILED);
    uint32_t lastDimTileNumTimes = (lastDimTileNum + maxCoreNum - 1) / maxCoreNum;
    uint64_t initUb = indexDtypeSize * topkV2DataInfo::BIN_NUM * (lastDimTileNumTimes + 1) +
                      Ops::Base::CeilAlign(
                          static_cast<uint64_t>(sizeof(uint32_t) * lastDimTileNumTimes), topkV2DataInfo::AGLIN_FACTOR) *
                          topkV2DataInfo::CONST_TWO;
    uint32_t factor = xDtypeSize * topkV2DataInfo::CONST_TWO + indexDtypeSize + indexToDtypeSize;

    if (tileData < kValue) {
        factor += xDtypeSize + indexToDtypeSize + sizeof(int32_t);
    } else {
        initUb += Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)), topkV2DataInfo::AGLIN_FACTOR) +
                  Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), topkV2DataInfo::AGLIN_FACTOR) +
                  Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize), topkV2DataInfo::AGLIN_FACTOR);
    }
    OP_LOGI(
        "TopKV2TilingForAscendC", "GetTopkTempBuffer tileData=%u, initUb=%u, factor = %u, ubSizePlatForm=%u", tileData,
        initUb, factor, ubSizePlatForm);
    return ubSizePlatForm - initUb - factor * tileData;
}

uint32_t ComputeTopkTileData(
    gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, ge::DataType dataType,
    ge::DataType indicesDType, bool isLargest, bool isSort, int64_t lastAxisNum, int64_t kValue, uint32_t maxCoreNum,
    uint64_t ubSizePlatForm)
{
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    uint32_t indexToDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(indicesDType)->second);
    uint32_t indexDtypeSize = indexToDtypeSize;
    uint32_t tileData = (topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0) ? topkV2DataInfo::TMP_DATA_NUM_B64 :
                                                                                   topkV2DataInfo::TMP_DATA_NUM;
    uint32_t tmpUb = GetTopkTempBuffer(
        lastAxisNum, tileData, maxCoreNum, xDtypeSize, indexToDtypeSize, indexDtypeSize, kValue, ubSizePlatForm);
    OP_CHECK_IF(tileData == 0, OP_LOGE("TopkV2", "tileData is 0"), return ge::GRAPH_FAILED);
    int64_t lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + tileData - 1) / tileData;
    int64_t maxInputK = std::max(lastDimTileNum, kValue);
    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, maxInputK, isLargest, dataType, isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    while (topkAcApiNeedBuffer > tmpUb) {
        tileData = tileData - topkV2DataInfo::BIN_NUM;
        OP_CHECK_IF(tileData == 0, OP_LOGE("TopkV2", "tileData is 0"), return ge::GRAPH_FAILED);
        lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + tileData - 1) / tileData;
        maxInputK = std::max(lastDimTileNum, kValue);
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, maxInputK, isLargest, dataType, isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        tmpUb = GetTopkTempBuffer(
            lastAxisNum, tileData, maxCoreNum, xDtypeSize, indexToDtypeSize, indexDtypeSize, kValue, ubSizePlatForm);
    }

    OP_LOGI(
        "TopKV2TilingForAscendC", "tileData %u TempBuffer %u ApiTempBuffer %u", tileData, tmpUb, topkAcApiNeedBuffer);
    return tileData;
}

uint32_t ComputeMergeSortTileData(
    TopKV2TilingDataSimd& topkTilingData, ge::DataType dataType, ge::DataType indicesDType, int64_t lastAxisNum,
    uint32_t maxCoreNum, uint32_t unsortedDimNum, uint64_t ubSizePlatForm)
{
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    uint32_t indexToDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(indicesDType)->second);
    uint32_t convertTypeSize = (dataType == ge::DT_BF16) ?
                                   topkV2DataInfo::optDataTypeBitMap.find(ge::DT_FLOAT)->second :
                                   topkV2DataInfo::optDataTypeBitMap.find(dataType)->second;

    uint32_t mergeSortAcApiNeedBuffer = topkTilingData.get_mergSortAcApiNeedBufferSize();
    uint32_t initUb = ubSizePlatForm - mergeSortAcApiNeedBuffer;
    uint32_t aglinNum = (static_cast<uint32_t>(lastAxisNum) + topkV2DataInfo::AGLIN_FACTOR - 1) /
                        topkV2DataInfo::AGLIN_FACTOR * topkV2DataInfo::AGLIN_FACTOR;
    uint32_t oneCoreRowNumSize = initUb - aglinNum * sizeof(uint32_t) -
                                 aglinNum * topkV2DataInfo::CONST_TWO * convertTypeSize * topkV2DataInfo::INT64_BYTE;
    uint32_t oneCoreRowNumMax =
        oneCoreRowNumSize / (aglinNum * topkV2DataInfo::CONST_TWO *
                             (topkV2DataInfo::CONST_TWO * xDtypeSize + indexToDtypeSize + convertTypeSize));
    uint32_t tileMaxData = oneCoreRowNumMax * aglinNum * 2;

    // 思路：1.占满核,均匀分核 2.循环数尽可能小
    uint32_t tileData = topkV2DataInfo::TMP_DATA_NUM;
    uint32_t oneCoreRowNum = (tileData / topkV2DataInfo::CONST_TWO) / aglinNum;
    oneCoreRowNum = (oneCoreRowNum == 0 ? 1 : oneCoreRowNum);
    uint32_t virUnsortedDimNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    if (virUnsortedDimNum < maxCoreNum) {
        // 均匀分核
        OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TopkV2", "maxCoreNum is 0"), return ge::GRAPH_FAILED);
        oneCoreRowNum = (unsortedDimNum + maxCoreNum - 1) / maxCoreNum;
        oneCoreRowNum = (oneCoreRowNum == 0 ? 1 : oneCoreRowNum);
        virUnsortedDimNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
        // 取整误差可能致使大于满核,应去除这种情况导致的性能降低
        while (virUnsortedDimNum > maxCoreNum) {
            oneCoreRowNum += 1;
            virUnsortedDimNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
        }
        tileData = oneCoreRowNum * topkV2DataInfo::CONST_TWO * aglinNum;
        tileData = std::min(tileData, tileMaxData - topkV2DataInfo::BIN_NUM);
    } else {
        // 原先占满了核则增大tileData，但必须使核占满
        while (virUnsortedDimNum >= maxCoreNum && tileData < tileMaxData - topkV2DataInfo::BIN_NUM) {
            tileData += topkV2DataInfo::BIN_NUM;
            oneCoreRowNum = (tileData / topkV2DataInfo::CONST_TWO) / aglinNum;
            oneCoreRowNum = (oneCoreRowNum == 0 ? 1 : oneCoreRowNum);
            virUnsortedDimNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
        }
    }

    return tileData;
}

void SetMergeSortTmpSize(
    gert::TilingContext* context, ge::DataType dataType, int64_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData)
{
    uint32_t aglinDataSize = (static_cast<uint32_t>(lastAxisNum) + topkV2DataInfo::AGLIN_FACTOR - 1) /
                             topkV2DataInfo::AGLIN_FACTOR * topkV2DataInfo::AGLIN_FACTOR;
    uint32_t dataTypeSize = (dataType == ge::DT_BF16) ? topkV2DataInfo::optDataTypeBitMap.find(ge::DT_FLOAT)->second :
                                                        topkV2DataInfo::optDataTypeBitMap.find(dataType)->second;
    auto platform_info = context->GetPlatformInfo();
    if (nullptr == platform_info) {
        OP_LOGE("TopKV2TilingForAscendC", "platform_info is nullptr.");
    }
    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, aglinDataSize, dataTypeSize);
    OP_LOGI("TopKV2TilingForAscendC", "Allocal buffer mergesort element len = %ld ac merge api", lastAxisNum);
    OP_LOGI("TopKV2TilingForAscendC", "Merge sort need tmp buffer %u byte for ac merge api", dataSizeNeed);
    topkTilingData.set_mergSortAcApiNeedBufferSize(dataSizeNeed);
}

void TileModeSmallSizeOptim(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkTileInfo& topkTileInfo, uint32_t nowTileSize)
{
    uint32_t aglinNum = (static_cast<uint32_t>(lastAxisNum) + topkV2DataInfo::AGLIN_FACTOR - 1) /
                        topkV2DataInfo::AGLIN_FACTOR * topkV2DataInfo::AGLIN_FACTOR;
    uint32_t oneCoreRowNum = (nowTileSize / 2) / aglinNum;
    oneCoreRowNum = (oneCoreRowNum == 0 ? 1 : oneCoreRowNum);
    uint32_t virUnsortedDimNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    uint32_t coreNumNeed = 0;
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TopkV2", "maxCoreNum is 0"), return);
    uint32_t sortLoopTimes = (virUnsortedDimNum + maxCoreNum - 1) / maxCoreNum;
    if (sortLoopTimes == 1) {
        uint32_t realCoreNum = virUnsortedDimNum % maxCoreNum;
        if (realCoreNum == 0) {
            realCoreNum = maxCoreNum;
        }
        coreNumNeed = realCoreNum;
    } else {
        coreNumNeed = maxCoreNum;
    }
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_lastDimTileNum(1);
    topkTilingData.set_unsortedDimParallel(coreNumNeed);
    topkTilingData.set_lastDimNeedCore(1);
    topkTilingData.set_numTileDataSize(lastAxisNum);
    topkTileInfo.ubRealLoadDataNum = lastAxisNum;
    topkTileInfo.coreNumNeed = coreNumNeed;
    topkTileInfo.lastDimTileNum = 1;
    topkTileInfo.unsortedDimParallel = coreNumNeed;
    topkTileInfo.oneCoreRowNum = oneCoreRowNum;
    OP_LOGI("TopKV2TilingForAscendC", "Small size opt mode oneCoreRowNum=%u", oneCoreRowNum);
    OP_LOGI(
        "TopKV2TilingForAscendC", "Small size opt mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        sortLoopTimes, lastAxisNum);
}

uint64_t GetSingleBlockTopkRunTimeNeedSpace(
    int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize, uint32_t indexToDtypeSize, int64_t kValue)
{
    OP_CHECK_IF(lastAxisNum == 0, OP_LOGE("TopkV2", "lastAxisNum is 0"), return ge::GRAPH_FAILED);
    uint32_t batchNumInUb = tileData / lastAxisNum;
    uint64_t aglinkValue = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue), topkV2DataInfo::AGLIN_FACTOR);
    uint64_t aglinkValueMultDtypeSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), topkV2DataInfo::AGLIN_FACTOR);
    uint64_t aglinkValueMultIndexDtypeSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize), topkV2DataInfo::AGLIN_FACTOR);
    uint64_t aglinIndicesOutTbuf =
        Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)), topkV2DataInfo::AGLIN_FACTOR);
    uint64_t initUb = batchNumInUb * (aglinkValue * xDtypeSize + aglinkValueMultDtypeSize +
                                      aglinkValueMultIndexDtypeSize + aglinIndicesOutTbuf);
    return initUb;
}

uint32_t ComputeSingleBlockTileData(
    gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, ge::DataType dataType,
    ge::DataType indicesDType, bool isLargest, bool isSort, int64_t lastAxisNum, int64_t kValue,
    uint64_t ubSizePlatForm)
{
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    uint32_t indexToDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(indicesDType)->second);
    uint32_t tileData = (topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0) ? topkV2DataInfo::TMP_DATA_NUM_B64 :
                                                                                   topkV2DataInfo::TMP_DATA_NUM;
    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    uint64_t needSpace =
        GetSingleBlockTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue);
    while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm) {
        tileData = tileData - topkV2DataInfo::BIN_NUM;
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        needSpace = GetSingleBlockTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue);
    }
    OP_LOGI(
        "TopKV2TilingForAscendC", "single block model tileData=%u, TempBuffer=%lu, ApiTempBuffer=%u", tileData,
        needSpace, topkAcApiNeedBuffer);
    return tileData;
}

uint64_t GetTopkMultiCoreOptimModeRunTimeNeedSpace(
    int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize, uint32_t indexToDtypeSize, int64_t kValue,
    uint64_t ubBlockAlignSize)
{
    uint64_t dataSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(tileData), ubBlockAlignSize) * xDtypeSize;
    uint64_t indexSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(tileData), ubBlockAlignSize) * sizeof(int32_t);
    uint64_t topkOutDataSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), ubBlockAlignSize);
    uint64_t topkOutIndexSpace =
        Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)), ubBlockAlignSize);
    uint64_t tempConversionSpace =
        Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize), ubBlockAlignSize);
    uint64_t initUb = dataSpace + indexSpace + topkOutDataSpace + topkOutIndexSpace + tempConversionSpace;
    OP_LOGI(
        "TopKV2TilingForAscendC",
        "compute runTime space lastAxisNum =%u, tileData=%lu, xDtypeSize=%u, indexToDtypeSize=%u, kValue=%u, initUb=%u",
        lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, initUb);
    return initUb;
}

bool IsMultiCoreOptimMode(
    gert::TilingContext* context, uint32_t& inputNowTileSize, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkComputingNowTileSizeInfo& computingNowTileSizeInfo)
{
    // get variable
    ge::DataType dataType = computingNowTileSizeInfo.dataType;
    ge::DataType indicesDType = computingNowTileSizeInfo.indicesDType;
    int64_t kValue = computingNowTileSizeInfo.kValue;
    bool isLargest = computingNowTileSizeInfo.isLargest;
    bool isSort = computingNowTileSizeInfo.isSort;
    int64_t lastAxisNum = computingNowTileSizeInfo.lastAxisNum;
    uint64_t ubBlockAlignSize = computingNowTileSizeInfo.ubBlockAlignSize;

    // computing runtime local tensor need space and topk api need space
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    uint32_t indexToDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(indicesDType)->second);
    int32_t tileData = (topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0) ? topkV2DataInfo::TMP_DATA_NUM_B64 :
                                                                                  topkV2DataInfo::TMP_DATA_NUM;
    if (tileData < kValue) {
        OP_LOGD("TopKV2TilingForAscendC", "k is greater than init tileData.");
        return false;
    }
    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    uint64_t needSpace = GetTopkMultiCoreOptimModeRunTimeNeedSpace(
        lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, ubBlockAlignSize);
    OP_LOGD(
        "TopKV2TilingForAscendC",
        "multi core optim model init tileData=%u, init tempBuffer=%lu, init apiTempBuffer=%u, xDtypeSize=%u, "
        "indexToDtypeSize=%u",
        tileData, needSpace, topkAcApiNeedBuffer, xDtypeSize, indexToDtypeSize);
    while (topkAcApiNeedBuffer + needSpace > computingNowTileSizeInfo.ubSizePlatForm) {
        tileData = tileData - topkV2DataInfo::TILE_SIZE_DECREASING_FACTOR;
        if (tileData < kValue) {
            OP_LOGD("TopKV2TilingForAscendC", "k value is greater than tilingData.");
            return false;
        }
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        needSpace = GetTopkMultiCoreOptimModeRunTimeNeedSpace(
            lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, ubBlockAlignSize);
        OP_LOGD(
            "TopKV2TilingForAscendC",
            "multi core optim model now tileData=%u, now tempBuffer=%lu, now apiTempBuffer=%u.", tileData, needSpace,
            topkAcApiNeedBuffer);
    }

    // 在确定正确的tileData之后，必须确保尾轴是多核模式，否则会出现多核的tiling模式，走的是singleBlock的模板
    if (tileData >= lastAxisNum) {
        OP_LOGD("TopKV2TilingForAscendC", "tileData is greater than lastAxisNum.");
        return false;
    }

    // compute tileNum multiply by kValue
    OP_CHECK_IF(tileData == 0, OP_LOGE("TopkV2", "tileData is 0"), return ge::GRAPH_FAILED);
    uint32_t lastDimTileNum = (lastAxisNum + tileData - 1) / tileData;
    uint32_t inputTopkSize = kValue * lastDimTileNum;
    if (inputTopkSize <= static_cast<uint32_t>(tileData)) {
        inputNowTileSize = tileData;
        OP_LOGI(
            "TopKV2TilingForAscendC", "multi core optim model final inputNowTileSize=%u, inputTopkSize=%u",
            inputNowTileSize, inputTopkSize);
        return true;
    }
    return false;
}

uint64_t GetSingleCoreTopkRunTimeNeedSpace(
    int64_t lastAxisNum, uint32_t nowTileSize, uint32_t xDtypeSize, uint32_t indexToDtypeSize, int64_t kValue,
    bool isSort)
{
    OP_CHECK_IF(nowTileSize == 0, OP_LOGE("TopkV2", "nowTileSize is 0"), return ge::GRAPH_FAILED);
    uint32_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    uint32_t tileNum = lastAxisNum / lastDimTileNum;
    uint32_t tailTileNum = lastAxisNum % lastDimTileNum;
    tileNum = tailTileNum == 0 ? tileNum : tileNum + 1;
    uint32_t outQueueNum = std::min(tileNum, static_cast<uint32_t>(kValue));

    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint32_t isInInt32Range = static_cast<uint32_t>(lastAxisNum <= int32Max);
    uint32_t indexTypeSize = isInInt32Range ? sizeof(int32_t) : sizeof(int64_t);

    uint64_t initUb = 0;
    uint64_t inputXQueValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(tileNum), topkV2DataInfo::AGLIN_FACTOR) * xDtypeSize;
    initUb += inputXQueValue;
    uint64_t valuesQueValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(outQueueNum * xDtypeSize), topkV2DataInfo::AGLIN_FACTOR);
    initUb += valuesQueValue;
    uint64_t indicesQueValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(outQueueNum * indexToDtypeSize), topkV2DataInfo::AGLIN_FACTOR);
    initUb += indicesQueValue;
    uint64_t outBufValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(outQueueNum * sizeof(int32_t)), topkV2DataInfo::AGLIN_FACTOR);
    initUb += outBufValue;

    uint64_t tileCusumTBufValue = topkV2DataInfo::BIN_NUM * sizeof(int32_t);
    initUb += tileCusumTBufValue;
    uint64_t cusumTBufValue = topkV2DataInfo::BIN_NUM * indexTypeSize;
    initUb += cusumTBufValue;
    uint64_t tileKTBufValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(lastDimTileNum * sizeof(int32_t)), topkV2DataInfo::AGLIN_FACTOR);
    initUb += tileKTBufValue;
    uint64_t unsignedInputXTBufValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(tileNum * sizeof(int32_t)), topkV2DataInfo::AGLIN_FACTOR);
    initUb += unsignedInputXTBufValue;
    uint64_t tileCusumInt64TBufValue = topkV2DataInfo::BIN_NUM * indexTypeSize;
    initUb += tileCusumInt64TBufValue;

    uint64_t sortSrcIndexTBufValue = 0;
    if (isSort && kValue * xDtypeSize <= topkV2DataInfo::SUPPORT_SORT_MAX_BYTE_SIZE) {
        sortSrcIndexTBufValue =
            Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize), topkV2DataInfo::AGLIN_FACTOR);
    }
    initUb += sortSrcIndexTBufValue;
    return initUb;
}

uint32_t ComputeSingleCoreTileData(
    gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, ge::DataType dataType,
    ge::DataType indicesDType, bool isLargest, bool isSort, int64_t lastAxisNum, int64_t kValue,
    uint64_t ubSizePlatForm)
{
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    uint32_t indexToDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(indicesDType)->second);
    uint32_t tileData = (topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0) ? topkV2DataInfo::TMP_DATA_NUM_B64 :
                                                                                   topkV2DataInfo::TMP_DATA_NUM;
    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    uint64_t needSpace =
        GetSingleCoreTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, isSort);
    OP_LOGI(
        "TopKV2TilingForAscendC", "single core model init tileData=%u, init TempBuffer=%lu, now ApiTempBuffer=%u",
        tileData, needSpace, topkAcApiNeedBuffer);
    while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm) {
        tileData = tileData - topkV2DataInfo::BIN_NUM;
        OP_CHECK_IF(tileData == 0, OP_LOGE("TopkV2", "tileData is 0"), return ge::GRAPH_FAILED);
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        needSpace =
            GetSingleCoreTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, isSort);
        OP_LOGI(
            "TopKV2TilingForAscendC", "single core model now tileData=%u, now TempBuffer=%lu, now ApiTempBuffer=%u",
            tileData, needSpace, topkAcApiNeedBuffer);
    }
    OP_LOGI(
        "TopKV2TilingForAscendC", "single core model tileData=%u, TempBuffer=%lu, ApiTempBuffer=%u", tileData,
        needSpace, topkAcApiNeedBuffer);
    return tileData;
}

void TileModeSmallSize(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkTileInfo& topkTileInfo, uint32_t nowTileSize)
{
    OP_CHECK_IF(lastAxisNum == 0, OP_LOGE("TopkV2", "lastAxisNum is 0"), return);
    uint32_t batchNumInUb = nowTileSize / lastAxisNum;
    uint32_t batchNumSingleLoop = maxCoreNum * batchNumInUb;
    uint32_t sortLoopTimes = unsortedDimNum / batchNumSingleLoop;

    uint32_t tailBatchNumTotal = unsortedDimNum % batchNumSingleLoop;
    uint32_t tailBatchNumSingleCore = 0;
    uint32_t tailBatchNum = 0;
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TopkV2", "maxCoreNum is 0"), return);
    uint32_t coreNumNeed = maxCoreNum;
    if (tailBatchNumTotal != 0) {
        tailBatchNumSingleCore = tailBatchNumTotal / maxCoreNum;
        tailBatchNum = tailBatchNumTotal % maxCoreNum;
        sortLoopTimes += 1;
    }
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_unsortedDimParallel(coreNumNeed);
    topkTilingData.set_numTileDataSize(lastAxisNum);
    topkTilingData.set_lastDimTileNum(1);
    topkTilingData.set_lastDimNeedCore(1);
    topkTileInfo.batchNumInUb = batchNumInUb;
    topkTileInfo.tailLoopBatchNum = tailBatchNumSingleCore;
    topkTileInfo.tailBatchNum = tailBatchNum;

    topkTileInfo.ubRealLoadDataNum = lastAxisNum;
    topkTileInfo.coreNumNeed = coreNumNeed;
    topkTileInfo.lastDimTileNum = 1;
    topkTileInfo.unsortedDimParallel = coreNumNeed;
    OP_LOGI(
        "TopKV2TilingForAscendC", "Small size mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        sortLoopTimes, lastAxisNum);
}

void TileModeSingleCore(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, uint32_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkTileInfo& topkTileInfo, uint32_t nowTileSize)
{
    OP_CHECK_IF(nowTileSize == 0, OP_LOGE("TopkV2", "nowTileSize is 0"), return);
    uint32_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TopkV2", "maxCoreNum is 0"), return);
    uint32_t sortLoopTimes = unsortedDimNum / maxCoreNum;
    uint32_t tailBatchNum = unsortedDimNum % maxCoreNum;
    uint32_t tileNum = lastAxisNum / lastDimTileNum;
    uint32_t tailTileNum = lastAxisNum % lastDimTileNum;

    topkTilingData.set_lastDimTileNum(lastDimTileNum);
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_numTileDataSize(tileNum);
    topkTilingData.set_unsortedDimParallel(maxCoreNum);

    topkTilingData.set_modeType(topkV2DataInfo::SINGLE_CORE_MODE);

    topkTileInfo.tailBatchNum = tailBatchNum;
    topkTileInfo.tailTileNum = tailTileNum;

    topkTileInfo.ubRealLoadDataNum = tileNum + 1;
    topkTileInfo.coreNumNeed = maxCoreNum;
    topkTileInfo.lastDimTileNum = lastDimTileNum;
    topkTileInfo.unsortedDimParallel = maxCoreNum;

    OP_LOGI(
        "TopKV2TilingForAscendC", "Single core mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%u",
        topkTileInfo.coreNumNeed, sortLoopTimes, lastAxisNum);
}

void TileModeMediumSize(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkTileInfo& topkTileInfo, uint32_t nowTileSize)
{
    if (topkTileInfo.topKOutLastAxisNum > topkV2DataInfo::MAX_K_FOR_INT64) {
        nowTileSize /= topkV2DataInfo::CONST_TWO;
    }
    OP_CHECK_IF(nowTileSize == 0, OP_LOGE("TopkV2", "nowTileSize is 0"), return);
    uint32_t lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + nowTileSize - 1) / nowTileSize;
    uint32_t unsortedDimParallel = maxCoreNum / lastDimTileNum;
    uint32_t coreNumNeed = lastDimTileNum * unsortedDimParallel;
    uint32_t sortLoopTimes = (unsortedDimNum + unsortedDimParallel - 1) / unsortedDimParallel;
    if (sortLoopTimes == 1) {
        coreNumNeed = unsortedDimNum * lastDimTileNum;
        unsortedDimParallel = unsortedDimNum;
    }
    topkTilingData.set_modeType(topkV2DataInfo::MULT_CORE_MODE);
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_lastDimTileNum(lastDimTileNum);
    topkTilingData.set_unsortedDimParallel(unsortedDimParallel);
    topkTilingData.set_lastDimNeedCore(lastDimTileNum);
    topkTilingData.set_numTileDataSize(nowTileSize);
    topkTileInfo.ubRealLoadDataNum = nowTileSize;
    topkTileInfo.coreNumNeed = coreNumNeed;
    topkTileInfo.lastDimTileNum = lastDimTileNum;
    topkTileInfo.unsortedDimParallel = unsortedDimParallel;
    OP_LOGI(
        "TopKV2TilingForAscendC", "Medium size mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        sortLoopTimes, lastAxisNum);
    OP_LOGI(
        "TopKV2TilingForAscendC", "Medium size mode lastDimTileNum=%u unsortedDimParallel=%u lastDimRealCore=%u",
        lastDimTileNum, unsortedDimParallel, lastDimTileNum);
}
void TileModeBigSize(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkTileInfo& topkTileInfo, uint32_t nowTileSize)
{
    if (topkTileInfo.topKOutLastAxisNum > topkV2DataInfo::MAX_K_FOR_INT64) {
        nowTileSize /= topkV2DataInfo::CONST_TWO;
    }
    OP_CHECK_IF(nowTileSize == 0, OP_LOGE("TopkV2", "nowTileSize is 0"), return);
    int64_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    uint32_t coreNumNeed = static_cast<uint32_t>(std::min(static_cast<int64_t>(maxCoreNum), lastDimTileNum));
    topkTilingData.set_modeType(topkV2DataInfo::MULT_CORE_MODE);
    topkTilingData.set_sortLoopTimes(unsortedDimNum);
    topkTilingData.set_lastDimTileNum(lastDimTileNum);
    topkTilingData.set_unsortedDimParallel(1);
    topkTilingData.set_lastDimNeedCore(coreNumNeed);
    topkTilingData.set_numTileDataSize(nowTileSize);
    topkTileInfo.ubRealLoadDataNum = nowTileSize;
    topkTileInfo.coreNumNeed = coreNumNeed;
    topkTileInfo.lastDimTileNum = lastDimTileNum;
    topkTileInfo.unsortedDimParallel = 1;
    OP_LOGI(
        "TopKV2TilingForAscendC", "Big size mode coreNumNeed=%u sortLoopTimes=%u lastAxisNum=%ld", coreNumNeed,
        unsortedDimNum, lastAxisNum);
    OP_LOGI(
        "TopKV2TilingForAscendC", "Big size mode lastDimTileNum=%ld unsortedDimParallel=1 lastDimRealCore=%u",
        lastDimTileNum, coreNumNeed);
}

void TileMultiCoreOptimSize(
    uint32_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, TopKV2TilingDataSimd& topkTilingData,
    topkV2DataInfo::TopkTileInfo& topkTileInfo, uint32_t nowTileSize)
{
    const uint32_t sortedDimParallelData = (nowTileSize * maxCoreNum) / 2;
    if (topkTileInfo.topKOutLastAxisNum > topkV2DataInfo::MAX_K_FOR_INT64) {
        nowTileSize *= topkV2DataInfo::CONST_TWO;
    }
    if (lastAxisNum <= sortedDimParallelData) {
        TileModeMediumSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else {
        TileModeBigSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    }
}

ge::graphStatus IsValidParam(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TopKV2", "Tiling Context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr,
        OP_LOGE(context->GetNodeName(), "Input desc is nullptr"), return ge::GRAPH_FAILED);
    auto inputDataType = context->GetInputDesc(0)->GetDataType();
    auto kDataType = context->GetInputDesc(1)->GetDataType();
    OP_CHECK_IF(
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr,
        OP_LOGE(context->GetNodeName(), "Output Shape is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape outValueShape = context->GetOutputShape(0)->GetStorageShape();
    const gert::Shape outIndexShape = context->GetOutputShape(1)->GetStorageShape();
    OP_CHECK_IF(
        context->GetOutputDesc(0) == nullptr || context->GetOutputDesc(1) == nullptr,
        OP_LOGE(context->GetNodeName(), "Output desc is nullptr"), return ge::GRAPH_FAILED);
    auto outputValueDataType = context->GetOutputDesc(0)->GetDataType();
    auto outputIndexDataType = context->GetOutputDesc(1)->GetDataType();
    OP_CHECK_IF(
        topkV2DataInfo::tilingDataTypeKeyMap.count(inputDataType) == 0,
        OP_LOGE(context->GetNodeName(), "Not support data type"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputDataType != outputValueDataType,
        OP_LOGE(context->GetNodeName(), "Input data type should equal to output value type"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        kDataType != ge::DT_INT32 && kDataType != ge::DT_INT64,
        OP_LOGE(context->GetNodeName(), "Input k data type should equal to int32 or int64"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputIndexDataType != ge::DT_INT32 && outputIndexDataType != ge::DT_INT64,
        OP_LOGE(context->GetNodeName(), "Output index data type should equal to int32 or int64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outValueShape != outIndexShape,
        OP_LOGE(context->GetNodeName(), "Output value shape should equal to output index shape"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool IsModeSingleCore(uint32_t unsortedDimNum, uint32_t maxCoreNum, uint32_t lastAxisNum)
{
    // 排序轴如果大于singlecore的阈值，则不走该模板
    if (lastAxisNum > topkV2DataInfo::SINGLE_CORE_THRESHOLD) {
        return false;
    }
    // B轴小于核数，则不走该模板
    if (unsortedDimNum < maxCoreNum) {
        return false;
    }
    // B轴均匀分核确定是有性能收益；
    // 均匀分核场景，需要考虑尾行处理的时间与核间同步时间的均衡；
    // 目前测试非均匀分核性能也有提升，故不区分是否均匀分核，直接返回true，后面如果有性能走这个模板有性能劣化可以考虑这一点
    return true;
}

ge::graphStatus TopKV2Tiling(gert::TilingContext* context, int32_t maxCoreNum)
{
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2Tiling start");
    TopKV2TilingDataSimd topkTilingData;
    OP_CHECK_IF(
        IsValidParam(context) == ge::GRAPH_FAILED, OP_LOGE("TopkV2", "Input param is invalid"),
        return ge::GRAPH_FAILED);
    const gert::Shape inputShape = context->GetInputShape(0)->GetStorageShape();
    auto dataType = context->GetInputDesc(0)->GetDataType();
    const gert::Shape outShape = context->GetOutputShape(0)->GetStorageShape();
    auto dataTypeKey = topkV2DataInfo::tilingDataTypeKeyMap.find(dataType)->second;
    auto indicesDType = context->GetOutputDesc(1)->GetDataType();
    std::string opType(context->GetNodeType());
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isSorted = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isSorted);
    OP_LOGI(context->GetNodeName(), "isSorted=%u", *isSorted);
    const int* dimValuePtr = attrs->GetAttrPointer<int>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimValuePtr);
    OP_LOGI(context->GetNodeName(), "dimValuePtr=%d", *dimValuePtr);
    const bool* isLargest = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, isLargest);
    OP_LOGI(context->GetNodeName(), "isLargest=%d", *isLargest);
    // check the indices_dtype attr and actual value of indices output
    const int* indicesDTypeValuePtr = attrs->GetAttrPointer<int>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDTypeValuePtr);
    OP_LOGI(context->GetNodeName(), "indicesDTypeValuePtr=%d", *indicesDTypeValuePtr);
    if ((*indicesDTypeValuePtr) != static_cast<int64_t>(indicesDType)) {
        OP_LOGE("TopKV2TilingForAscendC", "The indices_dtype attr differs from the actual dtype of output indices.");
        return ge::GRAPH_FAILED;
    }

    size_t inputDimNum = inputShape.GetDimNum();
    OP_CHECK_IF(
        *dimValuePtr < static_cast<int32_t>(-inputDimNum) || *dimValuePtr >= static_cast<int32_t>(inputDimNum),
        OP_LOGE(context->GetNodeName(), "Attr dim is out of range"), return ge::GRAPH_FAILED);
    int64_t lastAxisNum = inputShape.GetDim(inputDimNum - 1);
    uint32_t unsortedDimNum = 1;
    for (uint32_t i = 0; i < (inputDimNum - 1); i++) {
        unsortedDimNum *= inputShape.GetDim(i);
    }
    size_t outputDimNum = outShape.GetDimNum();
    int64_t outLastAxisNum = outShape.GetDim(outputDimNum - 1);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint32_t isInInt32Range = static_cast<uint32_t>(lastAxisNum <= int32Max);
    // Get Platform Info
    uint64_t ubSizePlatForm = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context));
    OP_LOGI("[TopKV2Tiling]", "ubBlock size = : %u", ubBlock);

    // 校验ubSizePlatForm
    ubSizePlatForm -= topkV2DataInfo::CONST_SIMT_SPACE;
    uint32_t nowTileSize = ComputeTopkTileData(
        context, topkTilingData, dataType, indicesDType, *isLargest, *isSorted, lastAxisNum, outLastAxisNum, maxCoreNum,
        ubSizePlatForm);
    topkV2DataInfo::TopkTileInfo topkTileInfo;
    topkTileInfo.topKOutLastAxisNum = outLastAxisNum;

    topkV2DataInfo::TopkComputingNowTileSizeInfo computingNowTileSizeInfo;
    computingNowTileSizeInfo.isLargest = *isLargest;
    computingNowTileSizeInfo.isSort = *isSorted;
    computingNowTileSizeInfo.isInInt32Range = lastAxisNum <= int32Max;
    computingNowTileSizeInfo.lastAxisNum = lastAxisNum;
    computingNowTileSizeInfo.kValue = outLastAxisNum;
    computingNowTileSizeInfo.maxCoreNum = maxCoreNum;
    computingNowTileSizeInfo.ubSizePlatForm = ubSizePlatForm;
    computingNowTileSizeInfo.dataType = dataType;
    computingNowTileSizeInfo.indicesDType = indicesDType;
    computingNowTileSizeInfo.ubBlockAlignSize = ubBlock;
    OP_LOGI(
        "[TopKV2Tiling]",
        "computingNowTileSizeInfo isLargest: %u, isSort: %u, isInInt32Range: %u, lastAxisNum: %u, kValue: %u, "
        "maxCoreNum: %u, ubSizePlatForm: %u",
        computingNowTileSizeInfo.isLargest, computingNowTileSizeInfo.isSort, computingNowTileSizeInfo.isInInt32Range,
        computingNowTileSizeInfo.lastAxisNum, computingNowTileSizeInfo.kValue, computingNowTileSizeInfo.maxCoreNum,
        computingNowTileSizeInfo.ubSizePlatForm);

    const uint32_t sortedDimParallelData = (nowTileSize * maxCoreNum) / 2;
    if (lastAxisNum <= topkV2DataInfo::SMALL_MAX_DATA_SZIE && topkV2DataInfo::optDataTypeBitMap.count(dataType) != 0) {
        SetMergeSortTmpSize(context, dataType, lastAxisNum, topkTilingData);
        uint32_t nowTileSizeTmp = ComputeMergeSortTileData(
            topkTilingData, dataType, indicesDType, lastAxisNum, maxCoreNum, unsortedDimNum, ubSizePlatForm);
        TileModeSmallSizeOptim(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSizeTmp);
        dataTypeKey += topkV2DataInfo::MERGE_SORT_TILING_OFFSET;
    } else if (lastAxisNum <= nowTileSize) {
        uint32_t nowTileSizeTmp = ComputeSingleBlockTileData(
            context, topkTilingData, dataType, indicesDType, *isLargest, *isSorted, lastAxisNum, outLastAxisNum,
            ubSizePlatForm);
        TileModeSmallSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSizeTmp);
    } else if (IsModeSingleCore(unsortedDimNum, maxCoreNum, static_cast<uint32_t>(lastAxisNum))) {
        uint32_t nowTileSizeTmp = ComputeSingleCoreTileData(
            context, topkTilingData, dataType, indicesDType, *isLargest, *isSorted, lastAxisNum, outLastAxisNum,
            ubSizePlatForm);
        OP_CHECK_IF(nowTileSizeTmp == 0, OP_LOGE("TopkV2", "nowTileSizeTmp is 0"), return ge::GRAPH_FAILED);
        TileModeSingleCore(
            unsortedDimNum, maxCoreNum, static_cast<uint32_t>(lastAxisNum), topkTilingData, topkTileInfo,
            nowTileSizeTmp);
    } else if (IsMultiCoreOptimMode(context, nowTileSize, topkTilingData, computingNowTileSizeInfo)) {
        // topk核间模板优化，修改modelType, tilling不变
        TileMultiCoreOptimSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
        topkTilingData.set_modeType(topkV2DataInfo::MULT_CORE_OPTIM_MODE);
    } else if (lastAxisNum <= sortedDimParallelData) {
        TileModeMediumSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else {
        TileModeBigSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize); // 核间模板
    }
    int64_t maxInputK = std::max(outLastAxisNum, topkTileInfo.lastDimTileNum);

    if (topkTilingData.get_modeType() == topkV2DataInfo::MULT_CORE_MODE &&
        topkTileInfo.topKOutLastAxisNum > topkV2DataInfo::MAX_K_FOR_INT64) {
        nowTileSize /= topkV2DataInfo::CONST_TWO;
    }

    GetTopkApiTmpBufferSize(
        context, topkTilingData, topkTileInfo.ubRealLoadDataNum, maxInputK, *isLargest, dataType, *isSorted,
        nowTileSize);
    OP_CHECK_IF(maxCoreNum == 0, OP_LOGE("TopkV2", "maxCoreNum is 0"), return ge::GRAPH_FAILED);
    int64_t lastDimTileNumTimesValue = (topkTileInfo.lastDimTileNum + maxCoreNum - 1) / maxCoreNum;
    OP_LOGI(
        "[TopKV2Tiling]", "lastAxisNum: %ld, int32Max: %ld, isInInt32Range: %u, nowTileSize: %u", lastAxisNum, int32Max,
        isInInt32Range, nowTileSize);
    context->SetTilingKey(dataTypeKey);
    context->SetBlockDim(topkTileInfo.coreNumNeed);
    context->SetScheduleMode(1);
    topkTilingData.set_isLargest(*isLargest);
    topkTilingData.set_isSort(*isSorted);
    topkTilingData.set_isInInt32Range(isInInt32Range);
    topkTilingData.set_lastAxisNum(lastAxisNum);
    topkTilingData.set_unsortedDimNum(unsortedDimNum);
    topkTilingData.set_lastDimTileNumTimes(lastDimTileNumTimesValue);
    topkTilingData.set_platformCoreNum(maxCoreNum);
    topkTilingData.set_topKRealValue(outLastAxisNum);
    topkTilingData.set_oneCoreRowNum(topkTileInfo.oneCoreRowNum);
    topkTilingData.set_outputLastDimValue(outLastAxisNum);
    topkTilingData.set_batchNumInUb(topkTileInfo.batchNumInUb);
    topkTilingData.set_tailLoopBatchNum(topkTileInfo.tailLoopBatchNum);
    topkTilingData.set_tailBatchNum(topkTileInfo.tailBatchNum);
    topkTilingData.set_tailTileNum(topkTileInfo.tailTileNum);
    topkTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());

    OP_LOGI(
        context->GetNodeName(),
        "TopK V2 tilingData tilingKey is %u, isLargest is %u, modelType is %u,"
        "isSorted is %u, lastAxisNum is %ld, unsortedDimNum is %u, lastDimTileNumTimesValue is %ld, outLastAxisNum is "
        "%ld,"
        "oneCoreRowNum is %u, sortLoopTimes is %u, lastDimTileNum is %ld, unsortedDimParallel is %u, modeType is %u,"
        "lastDimNeedCore is %u, numTileDataSize is %u, batchNumInUb is %u, tailLoopBatchNum is %u, tailBatchNum is %u,"
        "tailTileNum is %u, coreNumNeed is %u",
        dataTypeKey, topkTilingData.get_isLargest(), topkTilingData.get_modeType(), topkTilingData.get_isSort(),
        topkTilingData.get_lastAxisNum(), topkTilingData.get_unsortedDimNum(), topkTilingData.get_lastDimTileNumTimes(),
        topkTilingData.get_topKRealValue(), topkTilingData.get_oneCoreRowNum(), topkTilingData.get_sortLoopTimes(),
        topkTilingData.get_lastDimTileNum(), topkTilingData.get_unsortedDimParallel(), topkTilingData.get_modeType(),
        topkTilingData.get_lastDimNeedCore(), topkTilingData.get_numTileDataSize(), topkTilingData.get_batchNumInUb(),
        topkTilingData.get_tailLoopBatchNum(), topkTilingData.get_tailBatchNum(), topkTilingData.get_tailTileNum(),
        topkTileInfo.coreNumNeed);

    size_t usrSize = 0;
    if (isInInt32Range) {
        if (topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) {
            usrSize = topkTileInfo.lastDimTileNum * topkV2DataInfo::BIN_NUM * topkTileInfo.unsortedDimParallel *
                      sizeof(int32_t);
            OP_LOGI("[TopKV2Tiling]", "TopK V2 Workspace size is : %u", usrSize);
        } else if (topkTilingData.get_modeType() == topkV2DataInfo::MULT_CORE_OPTIM_MODE) {
            uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
            size_t tempSortResultNeedSize = Ops::Base::CeilAlign(
                static_cast<uint64_t>(
                    xDtypeSize * topkTileInfo.lastDimTileNum * topkTileInfo.unsortedDimParallel *
                    topkTilingData.get_topKRealValue()),
                topkV2DataInfo::AGLIN_FACTOR);
            size_t tempSortIndexNeedSize = Ops::Base::CeilAlign(
                static_cast<uint64_t>(
                    sizeof(int32_t) * topkTileInfo.lastDimTileNum * topkTileInfo.unsortedDimParallel *
                    topkTilingData.get_topKRealValue()),
                topkV2DataInfo::AGLIN_FACTOR);
            usrSize = tempSortResultNeedSize + tempSortIndexNeedSize;
            OP_LOGI("[TopKV2Tiling]", "Workspace size for Radix Sort perf template within Int32Range is : %u", usrSize);
        } else {
            size_t dataSetCumSumHistNeedSize =
                topkV2DataInfo::BIN_NUM * sizeof(int32_t) * topkTileInfo.unsortedDimParallel;
            size_t dataSetTileTopkNeedSize = Ops::Base::CeilAlign(
                static_cast<uint64_t>(topkTileInfo.lastDimTileNum * sizeof(int32_t) * topkTileInfo.unsortedDimParallel),
                topkV2DataInfo::AGLIN_FACTOR);
            usrSize = dataSetCumSumHistNeedSize + dataSetTileTopkNeedSize * topkV2DataInfo::CONST_TWO;
            OP_LOGI("[TopKV2Tiling]", "Workspace size for Radix Sort more core within Int32Range is : %u", usrSize);
        }
    } else {
        if (topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) {
            usrSize = topkTileInfo.lastDimTileNum * topkV2DataInfo::BIN_NUM * topkTileInfo.unsortedDimParallel *
                      sizeof(int64_t);
            OP_LOGI("[TopKV2Tiling]", "TopK V2 Workspace size is : %u", usrSize);
        } else {
            size_t dataSetCumSumHistNeedSize =
                topkV2DataInfo::BIN_NUM * sizeof(int64_t) * topkTileInfo.unsortedDimParallel;
            size_t dataSetTileTopkNeedSize = Ops::Base::CeilAlign(
                topkTileInfo.lastDimTileNum * sizeof(int64_t) * topkTileInfo.unsortedDimParallel,
                topkV2DataInfo::AGLIN_FACTOR);
            usrSize = dataSetCumSumHistNeedSize + dataSetTileTopkNeedSize * topkV2DataInfo::CONST_TWO;
            OP_LOGI("[TopKV2Tiling]", "Workspace size for Radix Sort more core beyond Int32Range is : %u", usrSize);
        }
    }
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = usrSize + topkV2DataInfo::SYS_WORK_SPACE_SIZE;
    OP_LOGI("[TopKV2Tiling]", "user & system WorkSpace Size is : %u", userWorkSpaceSize[0]);
    context->SetLocalMemorySize(ubSizePlatForm);
    context->SetScheduleMode(1);
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2Tiling end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForTopKV2(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "AscendC Tiling starting GRAPH_SUCCESS");
    auto compileInfo = context->GetCompiledInfo<TopKV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invaild."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4TopKV2(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const TopKV2CompileInfo*>(context->GetCompileInfo());
    OP_LOGI(context->GetNodeName(), "AscendC topk simd tiling");
    OP_CHECK_IF(
        TopKV2Tiling(context, compile_info->coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Topk simd tiling function failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TopKV2).Tiling(Tiling4TopKV2).TilingParse<TopKV2CompileInfo>(TilingPrepareForTopKV2);
} // namespace optiling
