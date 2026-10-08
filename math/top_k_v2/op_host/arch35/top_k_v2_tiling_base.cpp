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
 * \file top_k_v2_tiling_base.cpp
 * \brief top_k_v2 common tiling helpers implementation
 */
#include "top_k_v2_tiling_base.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "log/log.h"
#include "util/platform_util.h"

namespace optiling {
namespace topkV2 {

// ==================== Helper Functions ====================

uint32_t GetDataTypeSize(ge::DataType dataType) { return topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second; }

bool IsDataType64Bit(ge::DataType dataType) { return topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0; }

uint32_t GetDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::TMP_DATA_NUM_B64 : topkV2DataInfo::TMP_DATA_NUM;
}

uint32_t GetSingleBlockModelDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::SINGLE_BLOCK_DATA_NUM_B64 :
                                       topkV2DataInfo::SINGLE_BLOCK_DATA_NUM;
}

uint32_t GetSingleCoreModelDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::SINGLE_CORE_DATA_NUM_B64 : topkV2DataInfo::SINGLE_CORE_DATA_NUM;
}

// ==================== Common Align Helpers ====================

bool TopkCeilAlignUint32(uint64_t sizeToAlign, uint32_t alignment, uint32_t& alignedOut)
{
    uint64_t alignedResult = Ops::Base::CeilAlign(sizeToAlign, static_cast<uint64_t>(alignment));
    if (alignedResult > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedOut = static_cast<uint32_t>(alignedResult);
    return true;
}

namespace {

constexpr uint32_t TOPK_SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT = 4095;
constexpr uint32_t TOPK_TWO_STAGE_RANK_INVERSE_MAX_N = 64;
constexpr uint32_t TOPK_SMALL_AXIS_TIER_COUNT = 4;
constexpr uint32_t TOPK_SMALL_AXIS_TIER_WIDTH = 2;
constexpr uint32_t TOPK_TWO_STAGE_VALUE_BUFFER_COUNT = 2;
constexpr uint32_t TOPK_TWO_STAGE_RANK_INVERSE_INDEX_BUFFER_COUNT = 2;
constexpr uint32_t TOPK_TWO_STAGE_RADIX_INDEX_BUFFER_COUNT = 3;

struct TopkSmallAxisRule {
    ge::DataType dtype;
    uint32_t insertionAxisLimit;
    uint32_t twoStageAxisLimit;
    uint32_t insertionSegTiers[TOPK_SMALL_AXIS_TIER_COUNT][TOPK_SMALL_AXIS_TIER_WIDTH];
    uint32_t twoStageSegTiers[TOPK_SMALL_AXIS_TIER_COUNT][TOPK_SMALL_AXIS_TIER_WIDTH];
};

// Each tier is {maximum axis length, minimum segments per core}; the values are empirical route thresholds.
constexpr TopkSmallAxisRule TOPK_SMALL_AXIS_RULES[] = {
    {ge::DT_INT64, 16, 512, {{8, 1}, {16, 4}, {0, 0}}, {{15, 8}, {128, 4}, {512, 8}, {0, 0}}},
    {ge::DT_UINT64, 16, 512, {{8, 1}, {16, 4}, {0, 0}}, {{15, 8}, {128, 4}, {512, 8}, {0, 0}}},
    {ge::DT_INT32, 11, 384, {{8, 2}, {11, 4}, {0, 0}}, {{11, 8}, {64, 4}, {384, 8}, {0, 0}}},
    {ge::DT_UINT32, 11, 384, {{8, 2}, {11, 4}, {0, 0}}, {{11, 8}, {64, 4}, {384, 8}, {0, 0}}},
    {ge::DT_INT16, 8, 192, {{4, 2}, {8, 4}, {0, 0}}, {{7, 8}, {64, 4}, {192, 12}, {0, 0}}},
    {ge::DT_UINT16, 8, 192, {{4, 2}, {8, 4}, {0, 0}}, {{7, 8}, {64, 4}, {192, 12}, {0, 0}}},
    {ge::DT_INT8, 8, 128, {{4, 2}, {8, 7}, {0, 0}}, {{3, 8}, {64, 7}, {128, 16}, {0, 0}}},
    {ge::DT_UINT8, 8, 128, {{4, 2}, {8, 7}, {0, 0}}, {{3, 8}, {64, 7}, {128, 16}, {0, 0}}},
    {ge::DT_FLOAT, 8, 24, {{4, 16}, {8, 48}, {0, 0}}, {{24, 64}, {0, 0}}},
    {ge::DT_BF16, 8, 54, {{4, 16}, {8, 48}, {0, 0}}, {{54, 64}, {0, 0}}},
    {ge::DT_FLOAT16, 8, 54, {{4, 16}, {8, 48}, {0, 0}}, {{54, 64}, {0, 0}}},
};

struct TopkTwoStageBatchPlan {
    uint32_t rowsPerBatch = 0;
    uint32_t batchTotal = 0;
    uint32_t coreDim = 0;
    uint32_t tempUbBytes = 0;
};

bool TopkCeilDivUint32(uint64_t numerator, uint64_t divisorValue, uint32_t& divResult)
{
    if (divisorValue == 0U) {
        return false;
    }
    uint64_t ceilQuotient = Ops::Base::CeilDiv(numerator, divisorValue);
    if (ceilQuotient > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    divResult = static_cast<uint32_t>(ceilQuotient);
    return true;
}

uint32_t TopkLookupMinSegs(const uint32_t (*segTiers)[TOPK_SMALL_AXIS_TIER_WIDTH], uint32_t axisSize)
{
    for (uint32_t tierIdx = 0; tierIdx < TOPK_SMALL_AXIS_TIER_COUNT && segTiers[tierIdx][0] != 0U; ++tierIdx) {
        if (axisSize <= segTiers[tierIdx][0]) {
            return segTiers[tierIdx][1];
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

const TopkSmallAxisRule* TopkFindSmallAxisRule(ge::DataType dtype)
{
    for (const TopkSmallAxisRule& routeRule : TOPK_SMALL_AXIS_RULES) {
        if (routeRule.dtype == dtype) {
            return &routeRule;
        }
    }
    return nullptr;
}

bool TopkUseTwoStageRankInverse(uint32_t axisLength) { return axisLength <= TOPK_TWO_STAGE_RANK_INVERSE_MAX_N; }

static bool ComputeTopkBf16InsertionBytesPerSeg(uint32_t axisLength, uint64_t idxAlignedBytes, uint32_t ubAlignSize,
                                                uint64_t& segBytes)
{
    uint64_t rawCastBytes = 0U;
    if (ge::MulOverflow(axisLength, sizeof(int16_t), rawCastBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeTopkInsertionBytesPerSeg", "axisLength",
                                              std::to_string(axisLength).c_str(),
                                              "The value of axisLength must not cause cast raw byte size overflow.");
        return false;
    }
    uint64_t castSegBytes = Ops::Base::CeilAlign<uint64_t>(rawCastBytes, ubAlignSize);
    if (castSegBytes == 0U) {
        return false;
    }
    uint64_t castRowElemCount = castSegBytes / sizeof(int16_t);
    uint64_t floatCastBytes = 0U;
    if (ge::MulOverflow(castRowElemCount, sizeof(float), floatCastBytes) ||
        ge::AddOverflow(floatCastBytes, idxAlignedBytes, segBytes) ||
        ge::AddOverflow(segBytes, castSegBytes, segBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeTopkInsertionBytesPerSeg", "segBytes",
                                              (std::to_string(castRowElemCount) + ", " +
                                               std::to_string(idxAlignedBytes) + ", " + std::to_string(castSegBytes))
                                                  .c_str(),
                                              "The value of bf16 segBytes must not overflow.");
        return false;
    }
    return true;
}

uint32_t ComputeTopkInsertionBytesPerSeg(ge::DataType dtype, uint32_t axisLength, uint32_t elemSize,
                                         uint32_t idxElemSize, uint32_t ubAlignSize)
{
    uint64_t rawValueBytes = 0U;
    uint64_t rawIdxBytes = 0U;
    if (ge::MulOverflow(axisLength, elemSize, rawValueBytes) || ge::MulOverflow(axisLength, idxElemSize, rawIdxBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeTopkInsertionBytesPerSeg", "axisLength",
                                              std::to_string(axisLength).c_str(),
                                              "The value of axisLength must not cause raw byte size overflow.");
        return 0U;
    }
    uint64_t alignedValueBytes = Ops::Base::CeilAlign<uint64_t>(rawValueBytes, ubAlignSize);
    uint64_t alignedIdxBytes = Ops::Base::CeilAlign<uint64_t>(rawIdxBytes, ubAlignSize);
    if (alignedValueBytes == 0U || alignedIdxBytes == 0U) {
        return 0U;
    }
    uint64_t segBytes = 0U;
    if (ge::AddOverflow(alignedValueBytes, alignedIdxBytes, segBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "ComputeTopkInsertionBytesPerSeg", "segBytes",
            (std::to_string(alignedValueBytes) + ", " + std::to_string(alignedIdxBytes)).c_str(),
            "The value of alignedValueBytes plus alignedIdxBytes must not overflow.");
        return 0U;
    }
    if (dtype == ge::DT_BF16 &&
        !ComputeTopkBf16InsertionBytesPerSeg(axisLength, alignedIdxBytes, ubAlignSize, segBytes)) {
        return 0U;
    }
    if (segBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeTopkInsertionBytesPerSeg", "segBytes",
                                              std::to_string(segBytes).c_str(),
                                              "The value of segBytes must be less than or equal to uint32 max.");
        return 0U;
    }
    return static_cast<uint32_t>(segBytes);
}

static bool ComputeTopkNonLastBatchNum(int64_t outerCount, int64_t innerCount, uint32_t chunkWidth,
                                       uint32_t& batchCount)
{
    if (outerCount <= 0 || innerCount <= 0 || chunkWidth == 0U) {
        return false;
    }
    uint32_t innerLoopCount = 0U;
    if (!TopkCeilDivUint32(static_cast<uint64_t>(innerCount), static_cast<uint64_t>(chunkWidth), innerLoopCount)) {
        return false;
    }
    uint64_t batchCount64 = static_cast<uint64_t>(outerCount) * static_cast<uint64_t>(innerLoopCount);
    if (batchCount64 == 0U || batchCount64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    batchCount = static_cast<uint32_t>(batchCount64);
    return true;
}

bool QueryTopkSortTmpSizeRadix(ge::DataType dtype, uint32_t sortElemCount, uint32_t& tempUbSize)
{
    std::vector<int64_t> shapeList = {static_cast<int64_t>(sortElemCount)};
    ge::Shape radixShape(shapeList);
    AscendC::SortConfig sortCfg;
    sortCfg.type = AscendC::SortType::RADIX_SORT;
    sortCfg.isDescend = false;
    sortCfg.hasSrcIndex = false;
    sortCfg.hasDstIndex = true;
    uint32_t maxTmpBytes = 0U;
    uint32_t minTmpBytes = 0U;
    AscendC::GetSortMaxMinTmpSize(radixShape, dtype, ge::DT_UINT32, false, sortCfg, maxTmpBytes, minTmpBytes);
    tempUbSize = maxTmpBytes;
    return maxTmpBytes > 0U;
}

uint32_t TopkMaxTwoStageU16SafeBatch(uint32_t axisLength)
{
    if (axisLength == 0U || axisLength > static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())) {
        return 0U;
    }
    uint32_t maxBatchSize = static_cast<uint32_t>(
        std::sqrt(static_cast<double>(std::numeric_limits<uint16_t>::max()) / axisLength));
    while (static_cast<uint64_t>(maxBatchSize) * maxBatchSize * axisLength >
           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
        --maxBatchSize;
    }
    while (static_cast<uint64_t>(maxBatchSize + 1U) * (maxBatchSize + 1U) * axisLength <=
           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
        ++maxBatchSize;
    }
    return maxBatchSize;
}

bool ComputeTopkTwoStageSortTmpUb(ge::DataType dtype, uint32_t axisLength, uint32_t totalElemCount,
                                  uint32_t ubAlignSize, uint32_t& tempUbSize)
{
    tempUbSize = 0U;
    QueryTopkSortTmpSizeRadix(dtype, totalElemCount, tempUbSize);
    uint32_t alignedTemp = 0U;
    if (!TopkCeilAlignUint32(tempUbSize, ubAlignSize, alignedTemp)) {
        return false;
    }
    tempUbSize = alignedTemp;
    bool rankInverse = TopkUseTwoStageRankInverse(axisLength);
    if (!rankInverse) {
        uint32_t stage2Temp = 0U;
        QueryTopkSortTmpSizeRadix(ge::DT_UINT16, totalElemCount, stage2Temp);
        uint32_t stage2AlignedTemp = 0U;
        if (!TopkCeilAlignUint32(stage2Temp, ubAlignSize, stage2AlignedTemp)) {
            return false;
        }
        tempUbSize = std::max(tempUbSize, stage2AlignedTemp);
    }
    return true;
}

uint64_t EstimateTopkTwoStageUbBytes(const TopKSmallAxisRouteInfo& info, uint32_t totalElemCount, uint32_t sortTempUb)
{
    uint64_t rawValueBytes = 0U;
    uint64_t rawIdxBytes = 0U;
    uint64_t rawFinalIdxBytes = 0U;
    if (ge::MulOverflow(totalElemCount, info.dtypeSize, rawValueBytes) ||
        ge::MulOverflow(totalElemCount, sizeof(uint32_t), rawIdxBytes) ||
        ge::MulOverflow(totalElemCount, info.y2DtypeSize, rawFinalIdxBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("EstimateTopkTwoStageUbBytes", "totalElemCount",
                                              std::to_string(totalElemCount).c_str(),
                                              "The value of totalElemCount must not cause raw byte size overflow.");
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t alignedValueBytes = Ops::Base::CeilAlign<uint64_t>(rawValueBytes, info.blockUbSize);
    uint64_t alignedIdxBytes = Ops::Base::CeilAlign<uint64_t>(rawIdxBytes, info.blockUbSize);
    uint64_t alignedFinalIdxBytes = Ops::Base::CeilAlign<uint64_t>(rawFinalIdxBytes, info.blockUbSize);
    if (alignedValueBytes == 0U || alignedIdxBytes == 0U || alignedFinalIdxBytes == 0U) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint32_t indexBufCount = TopkUseTwoStageRankInverse(static_cast<uint32_t>(info.lastAxis)) ?
                                 TOPK_TWO_STAGE_RANK_INVERSE_INDEX_BUFFER_COUNT :
                                 TOPK_TWO_STAGE_RADIX_INDEX_BUFFER_COUNT;
    uint64_t totalUbBytes = 0U;
    uint64_t idxTotalUbBytes = 0U;
    if (ge::MulOverflow(alignedValueBytes, TOPK_TWO_STAGE_VALUE_BUFFER_COUNT, totalUbBytes) ||
        ge::MulOverflow(alignedIdxBytes, indexBufCount, idxTotalUbBytes) ||
        ge::AddOverflow(totalUbBytes, idxTotalUbBytes, totalUbBytes) ||
        ge::AddOverflow(totalUbBytes, alignedFinalIdxBytes, totalUbBytes) ||
        ge::AddOverflow(totalUbBytes, sortTempUb, totalUbBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "EstimateTopkTwoStageUbBytes", "totalUbBytes",
            (std::to_string(alignedValueBytes) + ", " + std::to_string(alignedIdxBytes) + ", " +
             std::to_string(indexBufCount) + ", " + std::to_string(alignedFinalIdxBytes) + ", " +
             std::to_string(sortTempUb))
                .c_str(),
            "The value of totalUbBytes must not overflow.");
        return std::numeric_limits<uint64_t>::max();
    }
    return totalUbBytes;
}

bool PrepareTopkTwoStageBatchCandidate(const TopKSmallAxisRouteInfo& info, uint32_t candidateBatch,
                                       uint32_t& totalElemCount, uint32_t& tempUbSize, bool& rankInverse,
                                       uint64_t& totalUbBytes)
{
    uint32_t axisLength = static_cast<uint32_t>(info.lastAxis);
    uint64_t totalElemCount64 = static_cast<uint64_t>(candidateBatch) * axisLength;
    if (totalElemCount64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    totalElemCount = static_cast<uint32_t>(totalElemCount64);
    tempUbSize = 0U;
    if (!ComputeTopkTwoStageSortTmpUb(info.dataType, axisLength, totalElemCount, info.blockUbSize, tempUbSize)) {
        return false;
    }
    rankInverse = TopkUseTwoStageRankInverse(axisLength);
    totalUbBytes = EstimateTopkTwoStageUbBytes(info, totalElemCount, tempUbSize);
    return true;
}

bool SearchTopkTwoStageBatchPlan(uint32_t maxBatchSize,
                                 std::function<bool(uint32_t, TopkTwoStageBatchPlan&)> tryBatchFn,
                                 TopkTwoStageBatchPlan& chosenPlan)
{
    if (maxBatchSize == 0U) {
        return false;
    }
    TopkTwoStageBatchPlan firstValidPlan;
    TopkTwoStageBatchPlan bestFitPlan;
    bool hasFirstValid = false;
    uint32_t firstPerCoreBatches = 0U;
    uint64_t leastIdleSlots = std::numeric_limits<uint64_t>::max();
    for (uint32_t candidateBatch = maxBatchSize; candidateBatch >= 1U; --candidateBatch) {
        TopkTwoStageBatchPlan candPlan;
        if (!tryBatchFn(candidateBatch, candPlan)) {
            continue;
        }
        if (!hasFirstValid) {
            firstValidPlan = candPlan;
            hasFirstValid = true;
            firstPerCoreBatches = Ops::Base::CeilDiv(firstValidPlan.batchTotal, firstValidPlan.coreDim);
            bestFitPlan = firstValidPlan;
            leastIdleSlots = static_cast<uint64_t>(firstPerCoreBatches) * firstValidPlan.coreDim -
                             firstValidPlan.batchTotal;
            continue;
        }
        uint32_t perCoreBatches = Ops::Base::CeilDiv(candPlan.batchTotal, candPlan.coreDim);
        if (perCoreBatches != firstPerCoreBatches) {
            break;
        }
        uint64_t idleSlotCount = static_cast<uint64_t>(perCoreBatches) * candPlan.coreDim - candPlan.batchTotal;
        if (idleSlotCount < leastIdleSlots) {
            bestFitPlan = candPlan;
            leastIdleSlots = idleSlotCount;
        }
    }
    if (!hasFirstValid) {
        return false;
    }
    chosenPlan = bestFitPlan;
    return true;
}

static bool ComputeTopkSmallAxisInsertionBatchParams(const TopKSmallAxisRouteInfo& info, uint32_t axisLength,
                                                     uint32_t& segBytes, uint32_t& availableUb, uint32_t& ubBatchLimit)
{
    if (info.ubSize <= topkV2DataInfo::SIMT_UB) {
        return false;
    }
    segBytes = ComputeTopkInsertionBytesPerSeg(info.dataType, axisLength, info.dtypeSize, info.y2DtypeSize,
                                               info.blockUbSize);
    if (segBytes == 0U) {
        return false;
    }
    availableUb = info.ubSize - topkV2DataInfo::SIMT_UB;
    ubBatchLimit = availableUb / segBytes;
    return true;
}

template <typename ComputeBatchCountFn>
static bool EstimateTopkSmallAxisInsertionBatching(const TopKSmallAxisRouteInfo& info, uint32_t batchSizeLimit,
                                                   ComputeBatchCountFn computeBatchCount, SmallAxisRoutePlan& routePlan)
{
    uint32_t axisLength = static_cast<uint32_t>(info.lastAxis);
    uint32_t segBytes = 0U;
    uint32_t availableUb = 0U;
    uint32_t ubBatchLimit = 0U;
    if (!ComputeTopkSmallAxisInsertionBatchParams(info, axisLength, segBytes, availableUb, ubBatchLimit) ||
        ubBatchLimit == 0U) {
        return false;
    }
    uint32_t batchValue = std::min({batchSizeLimit, ubBatchLimit, TOPK_SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT});
    if (batchValue == 0U) {
        return false;
    }
    routePlan.batchSize = batchValue;
    if (!computeBatchCount(batchValue, routePlan.batchNum)) {
        return false;
    }
    routePlan.blockDim = std::min(info.maxCoreNum, routePlan.batchNum);
    return routePlan.batchNum > 0U && routePlan.blockDim > 0U;
}

template <typename ComputeBatchCountFn>
static bool TryTopkSmallAxisTwoStageBatchCandidate(const TopKSmallAxisRouteInfo& info, uint32_t candidateBatch,
                                                   ComputeBatchCountFn computeBatchCount, SmallAxisRoutePlan& routePlan)
{
    // Reject candidates unsupported by the batch mapping before querying Sort temporary UB.
    routePlan.batchSize = candidateBatch;
    if (!computeBatchCount(candidateBatch, routePlan.batchNum)) {
        return false;
    }
    uint32_t totalElemCount = 0U;
    uint32_t tempUbSize = 0U;
    bool rankInverse = false;
    uint64_t totalUbBytes = 0U;
    if (!PrepareTopkTwoStageBatchCandidate(info, candidateBatch, totalElemCount, tempUbSize, rankInverse,
                                           totalUbBytes)) {
        return false;
    }
    if (totalUbBytes + topkV2DataInfo::SIMT_UB > info.ubSize) {
        return false;
    }
    routePlan.blockDim = std::min(info.maxCoreNum, routePlan.batchNum);
    routePlan.tmpUbSize = tempUbSize;
    routePlan.useRankInverse = rankInverse;
    return routePlan.batchNum > 0U && routePlan.blockDim > 0U;
}

template <typename ComputeBatchCountFn>
static bool EstimateTopkSmallAxisTwoStageBatching(const TopKSmallAxisRouteInfo& info, uint32_t batchSizeLimit,
                                                  ComputeBatchCountFn computeBatchCount, SmallAxisRoutePlan& routePlan)
{
    uint32_t axisLength = static_cast<uint32_t>(info.lastAxis);
    if (info.ubSize <= topkV2DataInfo::SIMT_UB || axisLength == 0U || batchSizeLimit == 0U) {
        return false;
    }
    bool rankInverse = TopkUseTwoStageRankInverse(axisLength);
    uint32_t indexBufCount = rankInverse ? TOPK_TWO_STAGE_RANK_INVERSE_INDEX_BUFFER_COUNT :
                                           TOPK_TWO_STAGE_RADIX_INDEX_BUFFER_COUNT;
    uint64_t minBytesPerElement = static_cast<uint64_t>(info.dtypeSize) * TOPK_TWO_STAGE_VALUE_BUFFER_COUNT +
                                  static_cast<uint64_t>(sizeof(uint32_t)) * indexBufCount + info.y2DtypeSize;
    uint64_t ubElemLimit = (info.ubSize - topkV2DataInfo::SIMT_UB) / minBytesPerElement;
    uint64_t ubBatchLimit = ubElemLimit / axisLength;
    uint32_t maxBatchSize = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(batchSizeLimit), ubBatchLimit));
    if (!rankInverse) {
        maxBatchSize = std::min(maxBatchSize, TopkMaxTwoStageU16SafeBatch(axisLength));
    }

    TopkTwoStageBatchPlan chosenPlan;
    auto tryBatchFn = [&info, &computeBatchCount](uint32_t candidateBatch, TopkTwoStageBatchPlan& candPlan) -> bool {
        SmallAxisRoutePlan candRoute;
        if (!TryTopkSmallAxisTwoStageBatchCandidate(info, candidateBatch, computeBatchCount, candRoute)) {
            return false;
        }
        candPlan.rowsPerBatch = candRoute.batchSize;
        candPlan.batchTotal = candRoute.batchNum;
        candPlan.coreDim = candRoute.blockDim;
        candPlan.tempUbBytes = candRoute.tmpUbSize;
        return true;
    };
    if (!SearchTopkTwoStageBatchPlan(maxBatchSize, tryBatchFn, chosenPlan)) {
        return false;
    }
    routePlan.batchSize = chosenPlan.rowsPerBatch;
    routePlan.batchNum = chosenPlan.batchTotal;
    routePlan.blockDim = chosenPlan.coreDim;
    routePlan.tmpUbSize = chosenPlan.tempUbBytes;
    routePlan.useRankInverse = rankInverse;
    return true;
}

static bool PickTopkSmallAxisRouteImpl(const TopKSmallAxisRouteInfo& info, uint32_t batchSizeLimit,
                                       std::function<bool(uint32_t, uint32_t&)> computeBatchCount,
                                       SmallAxisRoutePlan& routePlan)
{
    uint32_t axisLength = static_cast<uint32_t>(info.lastAxis);
    if (axisLength <= 1U) {
        return false;
    }
    const TopkSmallAxisRule* routeRule = TopkFindSmallAxisRule(info.dataType);
    if (routeRule == nullptr) {
        return false;
    }
    uint32_t perCoreSegCount = 0U;
    if (!TopkCeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum),
                           perCoreSegCount)) {
        return false;
    }
    SmallAxisRoutePlan twoStageRoute;
    if (routeRule->twoStageAxisLimit > 0U && axisLength <= routeRule->twoStageAxisLimit &&
        axisLength <= SMALL_AXIS_THRESHOLD &&
        perCoreSegCount >= TopkLookupMinSegs(routeRule->twoStageSegTiers, axisLength) &&
        EstimateTopkSmallAxisTwoStageBatching(info, batchSizeLimit, computeBatchCount, twoStageRoute)) {
        routePlan = twoStageRoute;
        routePlan.kind = SmallAxisRouteKind::TWO_STAGE;
        return true;
    }
    if (axisLength > routeRule->insertionAxisLimit ||
        perCoreSegCount < TopkLookupMinSegs(routeRule->insertionSegTiers, axisLength)) {
        return false;
    }
    SmallAxisRoutePlan insertionRoute;
    if (!EstimateTopkSmallAxisInsertionBatching(info, batchSizeLimit, computeBatchCount, insertionRoute)) {
        return false;
    }
    routePlan = insertionRoute;
    routePlan.kind = SmallAxisRouteKind::INSERTION;
    return true;
}

} // namespace

bool PickTopkSmallAxisRoute(const TopKSmallAxisRouteInfo& info, SmallAxisRoutePlan& routePlan)
{
    uint32_t perCoreSegCount = 0U;
    if (!TopkCeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum),
                           perCoreSegCount)) {
        return false;
    }
    auto computeBatchCount = [&info](uint32_t batchValue, uint32_t& batchCount) -> bool {
        return TopkCeilDivUint32(static_cast<uint64_t>(info.unsortedDim), batchValue, batchCount);
    };
    return PickTopkSmallAxisRouteImpl(info, perCoreSegCount, computeBatchCount, routePlan);
}

bool PickTopkNonLastSmallAxisRoute(const TopKSmallAxisRouteInfo& info, SmallAxisRoutePlan& routePlan)
{
    uint32_t batchSizeLimit = static_cast<uint32_t>(
        std::min<int64_t>(info.innerSize, static_cast<int64_t>(std::numeric_limits<uint32_t>::max())));
    auto computeBatchCount = [&info](uint32_t batchValue, uint32_t& batchCount) -> bool {
        return ComputeTopkNonLastBatchNum(info.outerSize, info.innerSize, batchValue, batchCount);
    };
    return PickTopkSmallAxisRouteImpl(info, batchSizeLimit, computeBatchCount, routePlan);
}

// ==================== FP32 MergeSort Helpers ====================

uint32_t AlignTopkMergeMoreCoreWorkspaceElems(int64_t elementNum)
{
    if (elementNum <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(elementNum * topkV2DataInfo::SORT_STRUCT_SIZE_FP32),
                             topkV2DataInfo::AGLIN_FACTOR) /
        topkV2DataInfo::SORT_STRUCT_SIZE_FP32);
}

uint32_t ComputeTopkMergeMoreCoreOnceMaxElements(uint64_t ubSizePlatForm, ge::DataType indicesDType)
{
    uint32_t indexBytes = GetDataTypeSize(indicesDType);
    uint32_t bytesPerElem = topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * topkV2DataInfo::SORT_STRUCT_SIZE_FP32 *
                            topkV2DataInfo::CONST_TWO;
    bytesPerElem += topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * static_cast<uint32_t>(sizeof(uint32_t));
    bytesPerElem += topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * static_cast<uint32_t>(sizeof(float));
    if (indexBytes == topkV2DataInfo::INT64_BYTE) {
        bytesPerElem += topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * indexBytes;
    }
    return bytesPerElem == 0 ? 0 : static_cast<uint32_t>(ubSizePlatForm / bytesPerElem);
}

uint32_t ComputeTopkMergeIntraCoreBlockSortSize(uint64_t ubSizePlatForm)
{
    constexpr uint32_t phase2Bytes = topkV2DataInfo::CONST_TWO * topkV2DataInfo::SORT_STRUCT_SIZE_FP32 *
                                     topkV2DataInfo::CONST_TWO * topkV2DataInfo::CONST_TWO;
    uint32_t blockSortElems = static_cast<uint32_t>(ubSizePlatForm / phase2Bytes);
    return (blockSortElems / topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN) * topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN;
}

uint32_t ComputeTopkMergeIntraCoreExtractChunkSize(uint64_t ubSizePlatForm, ge::DataType indicesDType)
{
    uint32_t idxElemBytes = GetDataTypeSize(indicesDType);
    uint32_t bytesPerElement = (topkV2DataInfo::SORT_STRUCT_SIZE_FP32 + sizeof(float) + sizeof(int32_t) +
                                idxElemBytes) *
                               topkV2DataInfo::CONST_TWO;
    uint32_t extractChunkElems = static_cast<uint32_t>(ubSizePlatForm / bytesPerElement);
    return (extractChunkElems / topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN) *
           topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN;
}

// ==================== NonLastSmallAxis Helpers ====================

uint32_t GetTopkPreferredInnerChunk(ge::DataType dtype, uint32_t candidateIdx)
{
    static constexpr uint32_t TOPK_CHUNK_CANDIDATES[][topkV2DataInfo::MAX_INNER_CHUNK_CANDIDATES] = {
        {4, 2, 1, 0, 0, 0},
        {8, 4, 2, 1, 0, 0},
        {16, 8, 4, 2, 1, 0},
        {32, 16, 8, 4, 2, 1},
    };
    static constexpr uint32_t TOPK_CHUNK_VALID_COUNT[] = {3, 4, 5, 6};
    uint32_t chunkGroup = 0;
    if (dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
        chunkGroup = topkV2DataInfo::INNER_CHUNK_GROUP_8BYTE;
    } else if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) {
        chunkGroup = topkV2DataInfo::INNER_CHUNK_GROUP_4BYTE;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_INT16 || dtype == ge::DT_UINT16) {
        chunkGroup = topkV2DataInfo::INNER_CHUNK_GROUP_2BYTE;
    } else if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
        chunkGroup = topkV2DataInfo::INNER_CHUNK_GROUP_1BYTE;
    } else {
        return 0;
    }
    return candidateIdx < TOPK_CHUNK_VALID_COUNT[chunkGroup] ? TOPK_CHUNK_CANDIDATES[chunkGroup][candidateIdx] : 0;
}

bool UseTopkNonLastMergeSort(ge::DataType dtype, uint32_t axisLength)
{
    return dtype == ge::DT_FLOAT ||
           ((dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) && axisLength <= topkV2DataInfo::SMALL_MAX_DATA_SZIE);
}

ge::DataType GetTopkNonLastSortDtype(ge::DataType dtype, bool mergeSortOn)
{
    return mergeSortOn && dtype == ge::DT_BF16 ? ge::DT_FLOAT : dtype;
}

uint32_t GetTopkNonLastSortDtypeSize(uint32_t elemSize, bool mergeSortOn, ge::DataType dtype)
{
    return mergeSortOn && dtype == ge::DT_BF16 ? static_cast<uint32_t>(sizeof(float)) : elemSize;
}

bool GetTopkNonLastSortTmpSize(ge::DataType dtype, uint32_t sortElemCount, bool mergeSortOn, bool descendOn,
                               uint32_t& tempUbSize)
{
    std::vector<int64_t> shapeList = {static_cast<int64_t>(sortElemCount)};
    ge::Shape nonLastShape(shapeList);
    AscendC::SortConfig sortCfg;
    sortCfg.type = mergeSortOn ? AscendC::SortType::MERGE_SORT : AscendC::SortType::RADIX_SORT;
    sortCfg.isDescend = descendOn;
    sortCfg.hasSrcIndex = false;
    sortCfg.hasDstIndex = true;
    uint32_t maxTmpBytes = 0;
    uint32_t minTmpBytes = 0;
    ge::DataType effectiveDtype = GetTopkNonLastSortDtype(dtype, mergeSortOn);
    AscendC::GetSortMaxMinTmpSize(nonLastShape, effectiveDtype, ge::DT_UINT32, true, sortCfg, maxTmpBytes, minTmpBytes);
    tempUbSize = maxTmpBytes;
    return maxTmpBytes > 0;
}

void ComputeTopkAxisDimProducts(const gert::Shape& shape, int64_t sortAxisIdx, TopkNonLastSmallAxisTileInfo& info)
{
    int64_t shapeRank = shape.GetDimNum();
    if (sortAxisIdx < 0 || sortAxisIdx >= shapeRank) {
        return;
    }
    int64_t outerProduct = 1;
    int64_t innerProduct = 1;
    for (int64_t dimIdx = 0; dimIdx < shapeRank; ++dimIdx) {
        int64_t dimValue = shape.GetDim(dimIdx);
        if (dimIdx < sortAxisIdx) {
            outerProduct *= dimValue;
        } else if (dimIdx > sortAxisIdx) {
            innerProduct *= dimValue;
        }
    }
    info.outerSize = outerProduct;
    info.innerSize = innerProduct;
    info.lastAxis = shape.GetDim(sortAxisIdx);
    info.unsortedDim = outerProduct * innerProduct;
}

// ==================== TopK API Buffer Calculation ====================

ge::graphStatus GetTopkApiTmpBufferSize(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                        uint32_t needDataNum, int64_t kValue, bool isLargest, ge::DataType dtype,
                                        bool isSort, uint32_t nowTileSize)
{
    int32_t aglinInnerValue = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(needDataNum), topkV2DataInfo::AGLIN_FACTOR));

    uint32_t aglinKValue = (topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) ?
                               std::min(static_cast<int64_t>(needDataNum), kValue) :
                               std::min(static_cast<int64_t>(nowTileSize), kValue);

    AscendC::TopKConfig topkConfig;
    topkConfig.algo = AscendC::TopKAlgo::RADIX_SELECT;
    topkConfig.order = AscendC::TopKOrder::UNSET;
    topkConfig.sorted = isSort;

    uint32_t maxBufferSize = 0;
    uint32_t minBufferSize = 0;
    bool isSuccess = AscendC::GetTopKMaxMinTmpSize(aglinInnerValue, 1, aglinKValue, false, false,
                                                   AscendC::TopKMode::TOPK_NORMAL, isLargest, dtype, topkConfig,
                                                   maxBufferSize, minBufferSize);

    OP_LOGI("TopKV2TilingForAscendC", "TopK API buffer: kValue=%ld, alignedK=%u, alignedInner=%u, bufferSize=%u",
            kValue, aglinKValue, aglinInnerValue, maxBufferSize);

    OP_CHECK_IF(!isSuccess,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GetTopKMaxMinTmpSize", "false",
                                                      "The value of GetTopKMaxMinTmpSize must be true."),
                return ge::GRAPH_FAILED);

    topkTilingData.set_topkAcApiTmpBufferSize(maxBufferSize);
    return ge::GRAPH_SUCCESS;
}

// ==================== Runtime Space Calculation ====================

uint64_t GetTopkMultiCoreRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t maxCoreNum,
                                          uint32_t xDtypeSize, uint32_t indexToDtypeSize, uint32_t indexDtypeSize,
                                          int64_t kValue)
{
    OP_CHECK_IF(tileData == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "tileData", std::to_string(tileData).c_str(),
                                                      "The value of tileData must be greater than 0."),
                return ge::GRAPH_FAILED);

    uint64_t aglinFactor = topkV2DataInfo::AGLIN_FACTOR;
    uint32_t lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + tileData - 1) / tileData;
    uint32_t lastDimTileNumTimes = (lastDimTileNum + maxCoreNum - 1) / maxCoreNum;
    uint64_t lastDimTileNumTimesAlign = Ops::Base::CeilAlign(
        static_cast<uint64_t>(sizeof(uint32_t) * lastDimTileNumTimes), aglinFactor);
    uint64_t initUb = indexDtypeSize * topkV2DataInfo::BIN_NUM * (lastDimTileNumTimes + 1) +
                      lastDimTileNumTimesAlign * topkV2DataInfo::CONST_TWO;

    uint32_t factor = xDtypeSize * topkV2DataInfo::CONST_TWO + indexDtypeSize + indexToDtypeSize;

    if (tileData < kValue) {
        factor += xDtypeSize + indexToDtypeSize + sizeof(int32_t);
    } else {
        initUb += Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)), aglinFactor) +
                  Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), aglinFactor) +
                  Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize), aglinFactor);
    }
    OP_LOGI("TopKV2TilingForAscendC", "tileData=%u, initUb=%u, factor = %u", tileData, initUb, factor);
    return initUb + factor * tileData;
}

uint64_t GetSingleBlockTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                            uint32_t indexToDtypeSize, int64_t kValue)
{
    OP_CHECK_IF(lastAxisNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "lastAxisNum", std::to_string(lastAxisNum).c_str(),
                                                      "The value of lastAxisNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    uint32_t batchNumInUb = tileData / lastAxisNum;
    uint64_t alignTileData = Ops::Base::CeilAlign(static_cast<uint64_t>(lastAxisNum), topkV2DataInfo::AGLIN_FACTOR);
    uint64_t alignkValueMultDtypeSize = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize),
                                                             topkV2DataInfo::AGLIN_FACTOR);
    uint64_t alignkValueMultIndexDtypeSize = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize),
                                                                  topkV2DataInfo::AGLIN_FACTOR);
    uint64_t alignIndicesOutTbuf = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)),
                                                        topkV2DataInfo::AGLIN_FACTOR);
    uint64_t initUb = batchNumInUb * (alignTileData * xDtypeSize + alignkValueMultDtypeSize +
                                      alignkValueMultIndexDtypeSize + alignIndicesOutTbuf);
    OP_LOGD("TopKV2TilingForAscendC",
            "compute single block alignTileData=%u, alignkValueMultDtypeSize=%u, "
            "alignkValueMultIndexDtypeSize=%u, alignIndicesOutTbuf=%u.",
            alignTileData, alignkValueMultDtypeSize, alignkValueMultIndexDtypeSize, alignIndicesOutTbuf);
    return initUb;
}

uint64_t GetTopkMultiCoreOptimModeRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                                   uint32_t indexToDtypeSize, int64_t kValue, uint64_t ubBlockAlignSize)
{
    uint64_t dataSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(tileData), ubBlockAlignSize) * xDtypeSize;
    uint64_t indexSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(tileData), ubBlockAlignSize) * sizeof(int32_t);
    uint64_t topkOutDataSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), ubBlockAlignSize);
    uint64_t topkOutIndexSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)),
                                                      ubBlockAlignSize);
    uint64_t tempConversionSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize),
                                                        ubBlockAlignSize);
    uint64_t initUb = dataSpace + indexSpace + topkOutDataSpace + topkOutIndexSpace + tempConversionSpace;
    OP_LOGI(
        "TopKV2TilingForAscendC",
        "compute runTime space lastAxisNum =%u, tileData=%lu, xDtypeSize=%u, indexToDtypeSize=%u, kValue=%u, initUb=%u",
        lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, initUb);
    return initUb;
}

uint64_t GetSingleCoreTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t nowTileSize, uint32_t xDtypeSize,
                                           uint32_t indexToDtypeSize, int64_t kValue, bool isSort)
{
    OP_CHECK_IF(nowTileSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "nowTileSize", std::to_string(nowTileSize).c_str(),
                                                      "The value of nowTileSize must be greater than 0."),
                return ge::GRAPH_FAILED);

    auto ceilAlign = [](uint64_t value) -> uint64_t {
        return Ops::Base::CeilAlign(value, topkV2DataInfo::AGLIN_FACTOR);
    };

    int64_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    uint32_t tileNum = lastAxisNum / lastDimTileNum;
    uint32_t tailTileNum = lastAxisNum % lastDimTileNum;
    tileNum = tailTileNum == 0 ? tileNum : tileNum + 1;
    uint32_t outQueueNum = std::min(tileNum, static_cast<uint32_t>(kValue));

    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint32_t indexTypeSize = (lastAxisNum <= int32Max) ? sizeof(int32_t) : sizeof(int64_t);

    uint64_t initUb = 0;
    initUb += ceilAlign(tileNum) * xDtypeSize;
    initUb += ceilAlign(outQueueNum * xDtypeSize);
    initUb += ceilAlign(outQueueNum * indexToDtypeSize);
    initUb += ceilAlign(outQueueNum * sizeof(int32_t));
    initUb += topkV2DataInfo::BIN_NUM * sizeof(int32_t);
    initUb += topkV2DataInfo::BIN_NUM * indexTypeSize;
    initUb += ceilAlign(static_cast<uint64_t>(lastDimTileNum) * sizeof(int32_t));
    initUb += ceilAlign(tileNum * sizeof(int32_t));
    initUb += topkV2DataInfo::BIN_NUM * indexTypeSize;

    if (isSort && kValue * xDtypeSize <= topkV2DataInfo::SUPPORT_SORT_MAX_BYTE_SIZE) {
        initUb += ceilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize));
    }

    return initUb;
}

// ==================== MergeSort Helpers ====================

bool IsLastLoopCoreUtilizationSuccess(uint64_t unsortedDimNum, uint32_t tmpOneCoreRowNum, uint32_t maxCoreNum)
{
    uint64_t virUnsortedDimNeedCoreNum = (unsortedDimNum + tmpOneCoreRowNum - 1) / tmpOneCoreRowNum;
    uint64_t sortLoopTimes = (virUnsortedDimNeedCoreNum + maxCoreNum - 1) / maxCoreNum;
    uint32_t lastLoopDimNum = static_cast<uint32_t>(unsortedDimNum %
                                                    (static_cast<uint64_t>(maxCoreNum) * tmpOneCoreRowNum));
    uint32_t lastLoopDimNeedCoreNum = lastLoopDimNum / tmpOneCoreRowNum;
    if (lastLoopDimNum == 0) {
        return true;
    }
    bool loopTimesCondition = sortLoopTimes >= topkV2DataInfo::SMALL_LOOP_LOWER_NUM &&
                              sortLoopTimes <= topkV2DataInfo::SMALL_LOOP_UPPER_NUM;
    bool utilizationCondition = lastLoopDimNeedCoreNum < maxCoreNum * topkV2DataInfo::LAST_LOOP_CORE_UTILIZATION;
    if (loopTimesCondition && utilizationCondition) {
        return false;
    }
    return true;
}

uint32_t GetTileDataForMergeSort(uint64_t unsortedDimNum, uint32_t maxCoreNum, uint32_t tileMaxData, uint32_t bufferNum,
                                 uint32_t aglinNum)
{
    OP_CHECK_IF(bufferNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "bufferNum", std::to_string(bufferNum).c_str(),
                                                      "The value of bufferNum must be greater than 0."),
                return topkV2DataInfo::SMALL_MAX_DATA_SZIE);
    OP_CHECK_IF(aglinNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "aglinNum", std::to_string(aglinNum).c_str(),
                                                      "The value of aglinNum must be greater than 0."),
                return topkV2DataInfo::SMALL_MAX_DATA_SZIE);

    uint32_t tileData = topkV2DataInfo::TMP_DATA_NUM;
    uint32_t oneCoreRowNum = (tileData / bufferNum) / aglinNum;
    oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
    uint64_t virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;

    if (virUnsortedDimNeedCoreNum < maxCoreNum) {
        oneCoreRowNum = (unsortedDimNum + maxCoreNum - 1) / maxCoreNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
        tileData = oneCoreRowNum * bufferNum * aglinNum;
        tileData = std::min(tileData, tileMaxData - topkV2DataInfo::BIN_NUM);
        return tileData;
    }

    while (virUnsortedDimNeedCoreNum >= maxCoreNum && topkV2DataInfo::BIN_NUM + tileData < tileMaxData) {
        tileData += topkV2DataInfo::BIN_NUM;
        oneCoreRowNum = (tileData / bufferNum) / aglinNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    }

    uint32_t tmpTileData = tileData;
    while (!IsLastLoopCoreUtilizationSuccess(unsortedDimNum, oneCoreRowNum, maxCoreNum)) {
        if (tileData < topkV2DataInfo::BIN_NUM) {
            OP_LOGD("TopKV2TilingForAscendC", "tileData optimization =%u", tmpTileData);
            return tmpTileData;
        }
        tileData -= topkV2DataInfo::BIN_NUM;
        oneCoreRowNum = (tileData / bufferNum) / aglinNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    }

    return tileData;
}

void SetMergeSortTmpSize(gert::TilingContext* context, ge::DataType dataType, int64_t lastAxisNum,
                         TopKV2TilingDataSimd& topkTilingData)
{
    auto platform_info = context->GetPlatformInfo();
    if (nullptr == platform_info) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platform_info");
    }

    uint32_t alignDataSize = (static_cast<uint32_t>(lastAxisNum) + topkV2DataInfo::AGLIN_FACTOR - 1) /
                             topkV2DataInfo::AGLIN_FACTOR * topkV2DataInfo::AGLIN_FACTOR;
    uint32_t dataTypeSize = (dataType == ge::DT_BF16) ? GetDataTypeSize(ge::DT_FLOAT) : GetDataTypeSize(dataType);

    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, alignDataSize, dataTypeSize);
    OP_LOGI("TopKV2TilingForAscendC", "Allocal buffer mergesort element len = %ld ac merge api", lastAxisNum);
    OP_LOGI("TopKV2TilingForAscendC", "Merge sort need tmp buffer %u byte for ac merge api", dataSizeNeed);
    topkTilingData.set_mergSortAcApiNeedBufferSize(dataSizeNeed);
}

// ==================== Mode Judgment Functions ====================

bool needSortWithIndex(TopKV2TilingDataSimd& topkTilingData, bool isSorted, ge::DataType dataType)
{
    if (isSorted && topkTilingData.get_modeType() == topkV2DataInfo::MULT_CORE_MODE) {
        if (topkTilingData.get_topKRealValue() <= topkV2DataInfo::SUPPORT_SORT_MAX_SIZE) {
            return false;
        }
        return true;
    }
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    if (isSorted && topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) {
        if (topkTilingData.get_topKRealValue() <= topkV2DataInfo::SUPPORT_SORT_MAX_SIZE &&
            topkTilingData.get_topKRealValue() * xDtypeSize <= topkV2DataInfo::SUPPORT_SORT_MAX_BYTE_SIZE) {
            return false;
        }
        return true;
    }
    return false;
}

bool IsBitonicSmallTopkMode(int64_t kValue, int64_t sortPolicy, bool isSort)
{
    return isSort && static_cast<uint64_t>(kValue) >= topkV2DataInfo::BITONIC_SMALL_TOPK_MIN_K &&
           static_cast<uint64_t>(kValue) <= topkV2DataInfo::BITONIC_SMALL_TOPK_MAX_K &&
           static_cast<uint64_t>(sortPolicy) == topkV2DataInfo::BITONIC_SMALL_TOPK_POLICY;
}

// ==================== NonLastSmallAxis Calculation Helpers ====================

bool SearchTopkNonLastSmallAxisPlan(
    const TopkNonLastSmallAxisTileInfo& info, uint64_t availableUb,
    std::function<bool(TopkNonLastSmallAxisTileInfo&, uint32_t, uint64_t&, TopkNonLastSmallAxisCandidate&)>
        estimateUbFn,
    TopkNonLastSmallAxisCandidate& bestCand, TopkNonLastSmallAxisTileInfo* chosenInfo)
{
    for (uint32_t candIdx = 0; candIdx < topkV2DataInfo::MAX_INNER_CHUNK_CANDIDATES; ++candIdx) {
        uint32_t innerChunkSize = GetTopkPreferredInnerChunk(info.dataType, candIdx);
        if (innerChunkSize == 0U) {
            break;
        }
        innerChunkSize = static_cast<uint32_t>(
            std::min<uint64_t>(innerChunkSize, static_cast<uint64_t>(info.innerSize)));
        if (innerChunkSize == 0U) {
            return false;
        }
        uint64_t innerLoopCount64 = (static_cast<uint64_t>(info.innerSize) + innerChunkSize - 1U) / innerChunkSize;
        if (innerLoopCount64 == 0U || innerLoopCount64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }
        TopkNonLastSmallAxisCandidate curCand;
        curCand.innerChunk = innerChunkSize;
        curCand.innerLoopNum = static_cast<uint32_t>(innerLoopCount64);
        curCand.tileCount = static_cast<uint64_t>(info.outerSize) * innerLoopCount64;
        if (curCand.tileCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }
        TopkNonLastSmallAxisTileInfo candInfo = info;
        if (!estimateUbFn(candInfo, innerChunkSize, curCand.peakUb, curCand) || curCand.peakUb > availableUb) {
            continue;
        }
        curCand.activeCore = static_cast<uint32_t>(
            std::min<uint64_t>(static_cast<uint64_t>(info.maxCoreNum), curCand.tileCount));
        bool moreActiveCores = curCand.activeCore > bestCand.activeCore;
        bool sameCoresPreferLargerChunk = curCand.activeCore == bestCand.activeCore &&
                                          curCand.innerChunk > bestCand.innerChunk;
        if (moreActiveCores || sameCoresPreferLargerChunk) {
            bestCand = curCand;
            if (chosenInfo != nullptr) {
                *chosenInfo = candInfo;
            }
        }
    }
    return bestCand.innerChunk != 0U && bestCand.tileCount != 0U && bestCand.activeCore != 0U;
}

bool ComputeTopkNonLastLayout(const TopkNonLastSmallAxisTileInfo& info, uint32_t kValue, uint32_t innerChunk,
                              bool useMergeSort, topkV2DataInfo::NonLastSmallAxisTopkLayout& layout)
{
    if (innerChunk == 0U || kValue == 0U || info.dtypeSize == 0U || info.blockUbSize == 0U) {
        return false;
    }
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint32_t sortCount = Ops::Base::CeilAlign(axisLen, topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN);
    uint32_t sortDtypeSize = GetTopkNonLastSortDtypeSize(info.dtypeSize, useMergeSort, info.dataType);
    uint64_t valueAxisRawBytes = static_cast<uint64_t>(sortCount) * sortDtypeSize;
    uint32_t outputCount = useMergeSort ? sortCount : kValue;
    if (useMergeSort) {
        valueAxisRawBytes = std::max(valueAxisRawBytes,
                                     static_cast<uint64_t>(sortCount) * topkV2DataInfo::SORT_STRUCT_BYTES);
    }

    uint64_t valueOutputRawBytes = static_cast<uint64_t>(outputCount) * sortDtypeSize;
    if (useMergeSort) {
        valueOutputRawBytes = std::max(valueOutputRawBytes,
                                       static_cast<uint64_t>(outputCount) * topkV2DataInfo::SORT_STRUCT_BYTES);
    }
    if (!TopkCeilAlignUint32(static_cast<uint64_t>(innerChunk) * info.dtypeSize, info.blockUbSize,
                             layout.inputRowBytes) ||
        !TopkCeilAlignUint32(valueAxisRawBytes, info.blockUbSize, layout.axisRowBytes) ||
        !TopkCeilAlignUint32(valueOutputRawBytes, info.blockUbSize, layout.valueRowBytes) ||
        !TopkCeilAlignUint32(static_cast<uint64_t>(outputCount) * sizeof(uint32_t), info.blockUbSize,
                             layout.indexRowBytes)) {
        return false;
    }

    uint64_t inputRowElems = static_cast<uint64_t>(layout.inputRowBytes) / info.dtypeSize;
    uint64_t axisRowElems = static_cast<uint64_t>(layout.axisRowBytes) / sortDtypeSize;
    if (info.dtypeSize <= sizeof(uint16_t) &&
        ((static_cast<uint64_t>(axisLen) - 1U) * inputRowElems > std::numeric_limits<uint16_t>::max() ||
         static_cast<uint64_t>(innerChunk - 1U) * axisRowElems > std::numeric_limits<uint16_t>::max())) {
        return false;
    }
    return true;
}

bool EstimateTopkNonLastSmallAxisUb(TopkNonLastSmallAxisTileInfo& info, uint32_t kValue, uint32_t innerChunk,
                                    bool useMergeSort, uint64_t& peakUb, TopkNonLastSmallAxisCandidate& candidate)
{
    topkV2DataInfo::NonLastSmallAxisTopkLayout layout;
    if (!ComputeTopkNonLastLayout(info, kValue, innerChunk, useMergeSort, layout)) {
        return false;
    }
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint32_t sortCount = Ops::Base::CeilAlign(axisLen, topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN);

    uint32_t inputCastRowBytes = 0;
    if (useMergeSort && info.dataType == ge::DT_BF16 &&
        !TopkCeilAlignUint32(static_cast<uint64_t>(sortCount) * info.dtypeSize, info.blockUbSize, inputCastRowBytes)) {
        return false;
    }

    peakUb = static_cast<uint64_t>(axisLen) * layout.inputRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.axisRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.valueRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.indexRowBytes +
             static_cast<uint64_t>(innerChunk) * inputCastRowBytes + static_cast<uint64_t>(info.tmpUbSize);
    candidate.inputRowBytes = layout.inputRowBytes;
    candidate.valueAxisBytes = layout.axisRowBytes;
    candidate.indexAxisBytes = layout.valueRowBytes;
    candidate.outputIndexRowBytes = layout.indexRowBytes;
    info.inputRowBytes = layout.inputRowBytes;
    info.valueAxisBytes = layout.axisRowBytes;
    info.indexAxisBytes = layout.valueRowBytes;
    info.outputIndexRowBytes = layout.indexRowBytes;
    return true;
}

// ==================== NonLastSmallAxis Init and Search ====================

bool InitTopkNonLastSmallAxisInfo(gert::TilingContext* context, const gert::Shape& inputShape, int32_t axis,
                                  const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                  TopkNonLastSmallAxisTileInfo& info)
{
    info.rank = inputShape.GetDimNum();
    info.sortAxis = axis;
    info.maxCoreNum = computeInfo.maxCoreNum;
    info.dataType = computeInfo.dataType;
    info.dtypeSize = GetDataTypeSize(computeInfo.dataType);
    info.y2DtypeSize = GetDataTypeSize(computeInfo.indicesDType);
    info.blockUbSize = static_cast<uint32_t>(computeInfo.ubBlockAlignSize);
    ComputeTopkAxisDimProducts(inputShape, axis, info);

    if (info.lastAxis <= 0 || info.innerSize <= 0 || info.outerSize <= 0 ||
        info.lastAxis > topkV2DataInfo::NON_LAST_SMALL_AXIS_THRESHOLD || computeInfo.kValue <= 0 ||
        computeInfo.kValue > info.lastAxis) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "axis, k",
            (std::to_string(info.lastAxis) + ", " + std::to_string(computeInfo.kValue)).c_str(),
            "The value of axis must be positive and less than or equal to threshold, and the value of k must be within "
            "the range [1, axis].");
        return false;
    }
    return true;
}

ge::graphStatus SetupTopkNonLastSmallAxisTmpUb(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                               const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                               TopkNonLastSmallAxisTileInfo& info, bool useMergeSort,
                                               uint32_t sortCount)
{
    if (useMergeSort) {
        uint32_t tmpUbSize = 0;
        if (!GetTopkNonLastSortTmpSize(computeInfo.dataType, sortCount, useMergeSort, computeInfo.isLargest,
                                       tmpUbSize)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GetTopkNonLastSortTmpSize", "false",
                                                  "The value of GetTopkNonLastSortTmpSize must be true.");
            return ge::GRAPH_FAILED;
        }
        info.tmpUbSize = tmpUbSize;
        topkTilingData.set_tmpUbSize(tmpUbSize);
        topkTilingData.set_topkAcApiTmpBufferSize(0);
    } else {
        if (GetTopkApiTmpBufferSize(context, topkTilingData, static_cast<uint32_t>(info.lastAxis), computeInfo.kValue,
                                    computeInfo.isLargest, computeInfo.dataType, computeInfo.isSort,
                                    static_cast<uint32_t>(info.lastAxis)) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        info.tmpUbSize = topkTilingData.get_topkAcApiTmpBufferSize();
        topkTilingData.set_tmpUbSize(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SearchBestTopkNonLastSmallAxisPlan(gert::TilingContext* context, TopkNonLastSmallAxisTileInfo& info,
                                                   const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                                   bool useMergeSort, TopkNonLastSmallAxisCandidate& best)
{
    TopkNonLastSmallAxisTileInfo selectedInfo = info;
    auto estimateUb = [kValue = static_cast<uint32_t>(computeInfo.kValue), useMergeSort](
                          TopkNonLastSmallAxisTileInfo& candidateInfo, uint32_t innerChunk, uint64_t& peakUb,
                          TopkNonLastSmallAxisCandidate& candidate) -> bool {
        return EstimateTopkNonLastSmallAxisUb(candidateInfo, kValue, innerChunk, useMergeSort, peakUb, candidate);
    };

    if (!SearchTopkNonLastSmallAxisPlan(info, computeInfo.ubSizePlatForm, estimateUb, best, &selectedInfo)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SearchTopkNonLastSmallAxisPlan", "false",
                                              "The value of SearchTopkNonLastSmallAxisPlan must be true.");
        return ge::GRAPH_FAILED;
    }
    info = selectedInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace topkV2
} // namespace optiling
