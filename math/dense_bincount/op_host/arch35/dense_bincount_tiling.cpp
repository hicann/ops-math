/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "util/const_util.h"
#include <algorithm>
#include <limits>
#include "../../op_kernel/arch35/dense_bincount_tiling_data.h"
#include "../../op_kernel/arch35/dense_bincount_tiling_key.h"
namespace optiling {
constexpr int64_t IDX_INPUT = 0;
constexpr int64_t IDX_SIZE = 1;
constexpr int64_t IDX_WEIGHTS = 2;
constexpr int64_t ATTR_BINARY = 0;
constexpr int64_t MAX_OUTPUT_ELEMENTS = std::numeric_limits<int64_t>::max() / sizeof(float);
constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr uint64_t DCACHE_SIZE = 128U * 1024U;
constexpr uint64_t UB_ALIGN_BYTES = 32U;
constexpr int64_t PRIVATE_WRITEBACK_FACTOR = 8;
struct DenseBincountCompileInfo {};

static int64_t SafeCeilDiv(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

static int64_t GetRequiredCoreNum(int64_t workload, int64_t maxCoreNum)
{
    if (workload <= 0) {
        return 1;
    }
    int64_t perCoreElements = std::max(SafeCeilDiv(workload, maxCoreNum), PER_CORE_MIN_ELEMENTS);
    return std::min(maxCoreNum, SafeCeilDiv(workload, perCoreElements));
}

static bool FitsInLocalMemory(int64_t elements, uint64_t localMemorySize)
{
    if (elements <= 0) {
        return false;
    }
    const uint64_t histogramBytes = static_cast<uint64_t>(elements) * sizeof(float);
    const uint64_t alignedBytes = ((histogramBytes + UB_ALIGN_BYTES - 1U) / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    return alignedBytes <= localMemorySize;
}

static uint32_t GetPrivateHistElems(int64_t outputElements, int64_t numValues, int64_t usedCoreNum,
                                    uint64_t localMemorySize, bool useRowPrivate, int64_t rowHistElems)
{
    if (outputElements <= 0 || numValues <= 0) {
        return 0U;
    }
    if (FitsInLocalMemory(outputElements, localMemorySize)) {
        const int64_t privateWriteElements = outputElements * usedCoreNum;
        if (SafeCeilDiv(privateWriteElements, PRIVATE_WRITEBACK_FACTOR) <= numValues) {
            return static_cast<uint32_t>(outputElements);
        }
    }
    if (useRowPrivate && FitsInLocalMemory(rowHistElems, localMemorySize)) {
        const int64_t privateWriteElements = rowHistElems * usedCoreNum;
        if (SafeCeilDiv(privateWriteElements, PRIVATE_WRITEBACK_FACTOR) <= numValues) {
            return static_cast<uint32_t>(rowHistElems);
        }
    }
    return 0U;
}

static bool ValidateDenseBincountInputs(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(IDX_INPUT);
    auto sizeShape = context->GetInputShape(IDX_SIZE);
    auto weightsShape = context->GetInputShape(IDX_WEIGHTS);
    auto inputDesc = context->GetInputDesc(IDX_INPUT);
    auto sizeDesc = context->GetInputDesc(IDX_SIZE);
    auto weightsDesc = context->GetInputDesc(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    const auto rank = inputShape->GetStorageShape().GetDimNum();
    const bool validDtype = inputDesc->GetDataType() == ge::DT_INT32 || inputDesc->GetDataType() == ge::DT_INT64;
    const bool validWeights = weightsDesc->GetDataType() == ge::DT_FLOAT;
    if (rank < 1 || rank > 2 || sizeShape->GetStorageShape().GetDimNum() != 1 ||
        sizeShape->GetStorageShape().GetShapeSize() != 1 || !validDtype ||
        sizeDesc->GetDataType() != inputDesc->GetDataType() || !validWeights) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "DenseBincount", "invalid",
                                              "dtype/rank/shape contract failed");
        return false;
    }
    return true;
}

static bool GetDenseBincountSize(gert::TilingContext* context, int64_t& size)
{
    if (Ops::Base::GetConstInt(context, IDX_SIZE, size) && size >= 0) {
        return true;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", "unknown",
                                          "size must be a non-negative constant");
    return false;
}

static bool SetDenseBincountTilingData(gert::TilingContext* context, const gert::Shape& shape,
                                       const gert::Shape& weightsShape, int64_t size, int64_t& weightsNum)
{
    const int64_t rank = shape.GetDimNum();
    const int64_t rows = rank == 1 ? 1 : shape.GetDim(0);
    const int64_t cols = rank == 1 ? shape.GetDim(0) : shape.GetDim(1);
    const int64_t numValues = shape.GetShapeSize();
    weightsNum = weightsShape.GetShapeSize();
    if (rows < 0 || cols < 0 || numValues < 0 || weightsNum < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shape", "unknown",
                                              "runtime tiling requires concrete non-negative dimensions");
        return false;
    }
    if (rows > 0 && size > MAX_OUTPUT_ELEMENTS / rows) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output.numel", "overflow",
                                              "output element count exceeds the int64 FLOAT32 storage limit");
        return false;
    }
    if (weightsNum != 0 && weightsNum != numValues) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "weights.numel", std::to_string(weightsNum),
                                              "weights element count must match input");
        return false;
    }
    auto* tiling = context->GetTilingData<DenseBincountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->numValues = numValues;
    tiling->inputRows = rows;
    tiling->inputCols = cols;
    tiling->size = size;
    return true;
}

static ge::graphStatus ConfigureDenseBincountSchedule(gert::TilingContext* context,
                                                      platform_ascendc::PlatformAscendC& platform, int64_t rank,
                                                      const gert::Shape& shape, bool binary,
                                                      DenseBincountTilingData* tiling)
{
    const int64_t outputElements = rank == 1 ? tiling->size : shape.GetDim(0) * tiling->size;
    const int64_t maxCoreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    if (maxCoreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "AIV core count", std::to_string(maxCoreNum),
                                              "platform AIV core count must be positive");
        return ge::GRAPH_FAILED;
    }
    const int64_t usedCoreNum = GetRequiredCoreNum(std::max(tiling->numValues, outputElements), maxCoreNum);
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= DCACHE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB size", std::to_string(ubSize),
                                              "UB must exceed the SIMT DCache reservation");
        return ge::GRAPH_FAILED;
    }
    const uint64_t localMemorySize = ubSize - DCACHE_SIZE;
    tiling->usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    const bool useRowPrivate = rank == 2 && binary && tiling->inputRows > 1 && tiling->inputRows <= usedCoreNum;
    tiling->privateHistElems = GetPrivateHistElems(outputElements, tiling->numValues, usedCoreNum, localMemorySize,
                                                   useRowPrivate, tiling->size);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS ||
        context->SetDynUBufSize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "failed to configure synchronized SIMT scheduling or dynamic UB");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DenseBincountTilingFunc(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    auto inputShape = context->GetInputShape(IDX_INPUT);
    if (!ValidateDenseBincountInputs(context))
        return ge::GRAPH_FAILED;
    auto weightsShape = context->GetInputShape(IDX_WEIGHTS);
    const auto& shape = inputShape->GetStorageShape();
    int64_t rank = shape.GetDimNum();
    int64_t size = 0;
    if (!GetDenseBincountSize(context, size))
        return ge::GRAPH_FAILED;
    int64_t weightsNum = 0;
    if (!SetDenseBincountTilingData(context, shape, weightsShape->GetStorageShape(), size, weightsNum))
        return ge::GRAPH_FAILED;
    auto* tiling = context->GetTilingData<DenseBincountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    bool binary = false;
    if (const auto* attrs = context->GetAttrs()) {
        if (const bool* v = attrs->GetBool(ATTR_BINARY))
            binary = *v;
    }
    if (ConfigureDenseBincountSchedule(context, platform, rank, shape, binary, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = platform.GetLibApiWorkSpaceSize();
    int32_t hasWeights = weightsNum > 0 ? 1 : 0;
    int32_t is1D = rank == 1 ? 1 : 0;
    uint32_t mode = static_cast<uint32_t>((is1D << 2) | ((binary ? 1 : 0) << 1) | hasWeights);
    context->SetTilingKey(GET_TPL_TILING_KEY(mode));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DenseBincountParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(DenseBincount)
    .Tiling(DenseBincountTilingFunc)
    .TilingParse<DenseBincountCompileInfo>(DenseBincountParse);
} // namespace optiling
