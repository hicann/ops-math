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
 * \file equal_tiling.cpp
 * \brief Host tiling for contiguous and broadcast Equal paths.
 */

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/equal_tiling_data.h"
#include "../op_kernel/equal_tiling_key.h"
#include <algorithm>
#include <array>
#include <limits>

namespace optiling {
namespace {

struct EqualCompileInfo {};

constexpr uint64_t REPEAT_BYTES = 256;
constexpr uint64_t DEFAULT_BLOCK_ALIGN_ELEMENTS = 512;
constexpr uint64_t FP16_BLOCK_ALIGN_ELEMENTS = REPEAT_BYTES / sizeof(uint16_t);
constexpr uint64_t INT8_BLOCK_ALIGN_ELEMENTS = REPEAT_BYTES / sizeof(uint8_t);
constexpr uint64_t MIN_BYTES_PER_CORE = 4 * 1024;
constexpr uint64_t INT8_MIN_BYTES_PER_CORE = 2 * 1024;
constexpr uint64_t GENERAL_BROADCAST_MIN_BYTES_PER_CORE = 2 * 1024;
constexpr uint64_t CONTIGUOUS_QUEUE_BUFFER_NUM = 2;
constexpr uint64_t BROADCAST_QUEUE_BUFFER_NUM = 1;
constexpr uint64_t INPUT_QUEUE_NUM = 2;
constexpr uint64_t OUTPUT_QUEUE_NUM = 1;
constexpr uint64_t BOOL_BYTES = sizeof(uint8_t);
constexpr uint64_t DATA_BLOCK_BYTES = 32;
constexpr uint64_t SANDWICH_EXTRA_DATA_BLOCKS = 2;
constexpr uint64_t SELECT_UB_RESERVED_BYTES = 8 * 1024;
constexpr uint64_t DEFAULT_UB_RESERVED_BYTES = 2 * SELECT_UB_RESERVED_BYTES;
constexpr uint64_t FP16_UB_RESERVED_BYTES = 2 * SELECT_UB_RESERVED_BYTES;
constexpr uint64_t FP16_MAX_TILE_ELEMENTS = (30 * 1024) / sizeof(uint16_t);
constexpr uint64_t EXTENDED_TYPE_UB_RESERVED_BYTES = 4 * SELECT_UB_RESERVED_BYTES;
constexpr uint64_t INT64_MAX_TILE_ELEMENTS = 3072;
constexpr uint64_t INT64_TAIL_REUSE_MAX_TILE_ELEMENTS = 4608;
constexpr uint32_t SANDWICH_AXIS_COUNT = 3;
constexpr uint32_t SANDWICH_MIDDLE_AXIS_OFFSET = 1;
constexpr uint32_t SANDWICH_TAIL_AXIS_OFFSET = 2;

struct BroadcastInfo {
    uint64_t totalLength = 1;
    uint64_t x1Length = 1;
    uint64_t x2Length = 1;
    uint32_t rank = 0;
    uint32_t mode = EQUAL_BROADCAST_CONTIGUOUS;
    uint32_t fastBroadcastInput = 0;
    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> outShape{};
    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> x1Stride{};
    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> x2Stride{};
    uint64_t fastOuter = 0;
    uint64_t fastMiddle = 0;
    uint64_t fastTail = 0;
    uint64_t fastPlaneLength = 0;
    uint64_t fastSourceLength = 0;
    uint64_t fastTmpBytes = 0;
};

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedAlignUp(uint64_t value, uint64_t alignment, uint64_t& result)
{
    if (alignment == 0) {
        return false;
    }
    return CheckedMul(CeilDiv(value, alignment), alignment, result);
}

bool ReadAlignedShape(const gert::Shape& shape, uint32_t rank, std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM>& dims,
                      uint64_t& length)
{
    if (shape.GetDimNum() > rank || rank > EQUAL_MAX_BROADCAST_DIM) {
        return false;
    }
    dims.fill(1);
    length = 1;
    const uint32_t offset = rank - static_cast<uint32_t>(shape.GetDimNum());
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        const int64_t dim = shape.GetDim(index);
        if (dim <= 0) {
            return false;
        }
        dims[offset + index] = static_cast<uint64_t>(dim);
        if (!CheckedMul(length, static_cast<uint64_t>(dim), length)) {
            return false;
        }
    }
    return true;
}

uint32_t GetBroadcastState(uint64_t x1Dim, uint64_t x2Dim)
{
    if (x1Dim == x2Dim) {
        return 0;
    }
    return x1Dim == 1 ? 1 : 2;
}

void CalcStride(const std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM>& inputShape,
                const std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM>& outShape, uint32_t rank,
                std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM>& stride)
{
    stride.fill(0);
    uint64_t runningStride = 1;
    for (int32_t index = static_cast<int32_t>(rank) - 1; index >= 0; --index) {
        const uint32_t dimIndex = static_cast<uint32_t>(index);
        stride[dimIndex] = inputShape[dimIndex] == 1 && outShape[dimIndex] != 1 ? 0 : runningStride;
        runningStride *= inputShape[dimIndex];
    }
}

bool GetBroadcastInfo(gert::TilingContext* context, BroadcastInfo& info)
{
    const auto* x1Shape = context->GetInputShape(0);
    const auto* x2Shape = context->GetInputShape(1);
    const auto* yShape = context->GetOutputShape(0);
    if (x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr) {
        return false;
    }

    const auto& x1StorageShape = x1Shape->GetStorageShape();
    const auto& x2StorageShape = x2Shape->GetStorageShape();
    const auto& yStorageShape = yShape->GetStorageShape();
    const uint32_t rawRank = static_cast<uint32_t>(std::max(x1StorageShape.GetDimNum(), x2StorageShape.GetDimNum()));
    if (rawRank > EQUAL_MAX_BROADCAST_DIM) {
        return false;
    }

    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> x1Dims{};
    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> x2Dims{};
    if (!ReadAlignedShape(x1StorageShape, rawRank, x1Dims, info.x1Length) ||
        !ReadAlignedShape(x2StorageShape, rawRank, x2Dims, info.x2Length)) {
        return false;
    }

    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> rawOut{};
    info.totalLength = 1;
    for (uint32_t index = 0; index < rawRank; ++index) {
        if (x1Dims[index] != x2Dims[index] && x1Dims[index] != 1 && x2Dims[index] != 1) {
            return false;
        }
        rawOut[index] = std::max(x1Dims[index], x2Dims[index]);
        if (!CheckedMul(info.totalLength, rawOut[index], info.totalLength)) {
            return false;
        }
    }

    if (yStorageShape.GetDimNum() != rawRank) {
        return false;
    }
    for (uint32_t index = 0; index < rawRank; ++index) {
        if (yStorageShape.GetDim(index) <= 0 || static_cast<uint64_t>(yStorageShape.GetDim(index)) != rawOut[index]) {
            return false;
        }
    }

    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> compressedX1{};
    std::array<uint64_t, EQUAL_MAX_BROADCAST_DIM> compressedX2{};
    uint32_t lastState = std::numeric_limits<uint32_t>::max();
    for (uint32_t index = 0; index < rawRank; ++index) {
        if (rawRank > 1 && rawOut[index] == 1) {
            continue;
        }
        const uint32_t state = GetBroadcastState(x1Dims[index], x2Dims[index]);
        if (info.rank > 0 && state == lastState) {
            const uint32_t dst = info.rank - 1;
            info.outShape[dst] *= rawOut[index];
            compressedX1[dst] *= x1Dims[index];
            compressedX2[dst] *= x2Dims[index];
        } else {
            const uint32_t dst = info.rank++;
            info.outShape[dst] = rawOut[index];
            compressedX1[dst] = x1Dims[index];
            compressedX2[dst] = x2Dims[index];
            lastState = state;
        }
    }
    if (info.rank == 0) {
        info.rank = 1;
        info.outShape[0] = 1;
        compressedX1[0] = 1;
        compressedX2[0] = 1;
    }
    CalcStride(compressedX1, info.outShape, info.rank, info.x1Stride);
    CalcStride(compressedX2, info.outShape, info.rank, info.x2Stride);

    if (info.x1Length == info.totalLength && info.x2Length == info.totalLength) {
        info.mode = EQUAL_BROADCAST_CONTIGUOUS;
    } else if (info.x1Length == 1) {
        info.mode = EQUAL_BROADCAST_X1_SCALAR;
    } else if (info.x2Length == 1) {
        info.mode = EQUAL_BROADCAST_X2_SCALAR;
    } else {
        info.mode = EQUAL_BROADCAST_GENERAL;
    }
    return true;
}

uint64_t GetUbBytesPerElement(ge::DataType dtype, uint32_t typeBytes, uint64_t queueBufferNum)
{
    const uint64_t queueBytes = queueBufferNum * INPUT_QUEUE_NUM * typeBytes +
                                queueBufferNum * OUTPUT_QUEUE_NUM * BOOL_BYTES;
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return queueBytes + sizeof(uint16_t);
        case ge::DT_BF16:
            return queueBytes + INPUT_QUEUE_NUM * sizeof(float);
        case ge::DT_INT16:
            return queueBytes + INPUT_QUEUE_NUM * sizeof(float);
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return queueBytes + 2 * sizeof(uint16_t);
        case ge::DT_INT64:
            return queueBytes + 4 * sizeof(uint32_t);
        case ge::DT_FLOAT16:
            return queueBytes + sizeof(uint16_t);
        default:
            return queueBytes + sizeof(uint16_t);
    }
}

void TrySetSandwichMode(ge::DataType dtype, uint32_t typeBytes, uint64_t tileLength,
                        const platform_ascendc::PlatformAscendC& platform, BroadcastInfo& info)
{
    if (info.mode != EQUAL_BROADCAST_GENERAL || info.rank < SANDWICH_AXIS_COUNT || dtype == ge::DT_INT8 ||
        dtype == ge::DT_UINT8 || dtype == ge::DT_INT64 || dtype == ge::DT_BOOL) {
        return;
    }
    const uint32_t firstFastAxis = info.rank - SANDWICH_AXIS_COUNT;
    const bool x1Broadcast = info.x1Stride[firstFastAxis] == 0 &&
                             info.x1Stride[firstFastAxis + SANDWICH_MIDDLE_AXIS_OFFSET] == 1 &&
                             info.x1Stride[firstFastAxis + SANDWICH_TAIL_AXIS_OFFSET] == 0 &&
                             info.x2Length == info.totalLength;
    const bool x2Broadcast = info.x2Stride[firstFastAxis] == 0 &&
                             info.x2Stride[firstFastAxis + SANDWICH_MIDDLE_AXIS_OFFSET] == 1 &&
                             info.x2Stride[firstFastAxis + SANDWICH_TAIL_AXIS_OFFSET] == 0 &&
                             info.x1Length == info.totalLength;
    if (x1Broadcast == x2Broadcast) {
        return;
    }
    const auto& broadcastStride = x1Broadcast ? info.x1Stride : info.x2Stride;
    for (uint32_t index = 0; index < firstFastAxis; ++index) {
        if (broadcastStride[index] != 0) {
            return;
        }
    }

    uint64_t planeLength = 0;
    const uint64_t middle = info.outShape[firstFastAxis + SANDWICH_MIDDLE_AXIS_OFFSET];
    const uint64_t tail = info.outShape[firstFastAxis + SANDWICH_TAIL_AXIS_OFFSET];
    if (!CheckedMul(middle, tail, planeLength) || planeLength > tileLength) {
        return;
    }
    const uint64_t cachedPlaneCount = std::max<uint64_t>(1, tileLength / planeLength);
    uint64_t repeatedMiddle = 0;
    if (!CheckedMul(cachedPlaneCount, middle, repeatedMiddle)) {
        return;
    }
    const ge::Shape srcShape({static_cast<int64_t>(repeatedMiddle), 1});
    const ge::Shape dstShape({static_cast<int64_t>(repeatedMiddle), static_cast<int64_t>(tail)});
    uint32_t maxTmpBytes = 0;
    uint32_t minTmpBytes = 0;
    AscendC::GetBroadCastMaxMinTmpSize(platform, srcShape, dstShape, typeBytes, false, maxTmpBytes, minTmpBytes);
    uint64_t sourceElementBytes = 0;
    uint64_t sourceBytes = 0;
    uint64_t extraUbBytes = 0;
    uint64_t fixedExtraBytes = 0;
    uint64_t totalReservedBytes = 0;
    if (!CheckedMul(repeatedMiddle, typeBytes, sourceElementBytes) ||
        !CheckedAlignUp(sourceElementBytes, DATA_BLOCK_BYTES, sourceBytes) ||
        !CheckedMul(SANDWICH_EXTRA_DATA_BLOCKS, DATA_BLOCK_BYTES, fixedExtraBytes) ||
        !CheckedAdd(static_cast<uint64_t>(minTmpBytes), sourceBytes, extraUbBytes) ||
        !CheckedAdd(extraUbBytes, fixedExtraBytes, extraUbBytes) ||
        !CheckedAdd(extraUbBytes, SELECT_UB_RESERVED_BYTES, totalReservedBytes)) {
        return;
    }
    const uint64_t reservedBytes = dtype == ge::DT_FLOAT16 ? FP16_UB_RESERVED_BYTES :
                                                             (dtype == ge::DT_INT16 ? EXTENDED_TYPE_UB_RESERVED_BYTES :
                                                                                      DEFAULT_UB_RESERVED_BYTES);
    if (minTmpBytes == 0 || totalReservedBytes > reservedBytes) {
        return;
    }
    info.mode = EQUAL_BROADCAST_SANDWICH;
    info.fastBroadcastInput = x1Broadcast ? 1 : 2;
    info.fastOuter = info.totalLength / planeLength;
    info.fastMiddle = middle;
    info.fastTail = tail;
    info.fastPlaneLength = planeLength;
    info.fastSourceLength = middle;
    info.fastTmpBytes = minTmpBytes;
}

void TrySetTailReuseMode(BroadcastInfo& info)
{
    // Repeatedly fetching the same contiguous tail from GM makes row-broadcast
    // cases needlessly slower, particularly for the INT64 comparison path.
    // Cache one tail tile and reuse it for consecutive output rows that map to
    // the same source row.  Restrict this mode to the exact layout it handles.
    if (info.mode != EQUAL_BROADCAST_GENERAL || info.rank < 2) {
        return;
    }
    const uint32_t tailAxis = info.rank - 1;
    const uint32_t rowAxis = info.rank - 2;
    const bool x1Broadcast = info.x1Length != info.totalLength && info.x2Length == info.totalLength &&
                             info.x1Stride[tailAxis] == 1 && info.x1Stride[rowAxis] == 0;
    const bool x2Broadcast = info.x2Length != info.totalLength && info.x1Length == info.totalLength &&
                             info.x2Stride[tailAxis] == 1 && info.x2Stride[rowAxis] == 0;
    if (x1Broadcast == x2Broadcast) {
        return;
    }
    info.mode = EQUAL_BROADCAST_TAIL_REUSE;
    info.fastBroadcastInput = x1Broadcast ? 1 : 2;
    info.fastTail = info.outShape[tailAxis];
    info.fastPlaneLength = info.fastTail;
    info.fastSourceLength = info.fastTail;
    info.fastOuter = info.totalLength / info.fastTail;
}

ge::graphStatus EqualTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<EqualTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* otherDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, otherDesc);
    OP_CHECK_IF(inputDesc->GetDataType() != otherDesc->GetDataType(),
                OP_LOGE(context, "Equal AICore inputs must have the same dtype"), return ge::GRAPH_FAILED);

    BroadcastInfo info{};
    OP_CHECK_IF(!GetBroadcastInfo(context, info), OP_LOGE(context, "Invalid Equal broadcast shapes"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.totalLength > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "Equal element count exceeds int64_t range"), return ge::GRAPH_FAILED);

    uint32_t typeBytes = 0;
    const bool typeLengthValid = ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), typeBytes);
    OP_CHECK_IF(!typeLengthValid || typeBytes == 0, OP_LOGE(context, "Failed to obtain Equal input dtype length"),
                return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        availableCores = platform.GetCoreNum();
    }
    OP_CHECK_IF(availableCores == 0 || ubSize == 0,
                OP_LOGE(context, "Invalid platform resources: coreNum=%u, ubSize=%llu", availableCores,
                        static_cast<unsigned long long>(ubSize)),
                return ge::GRAPH_FAILED);

    const ge::DataType dtype = inputDesc->GetDataType();
    const bool isByteInteger = dtype == ge::DT_INT8 || dtype == ge::DT_UINT8;
    const uint64_t minBytesPerCore = info.mode == EQUAL_BROADCAST_GENERAL ?
                                         GENERAL_BROADCAST_MIN_BYTES_PER_CORE :
                                         (isByteInteger ? INT8_MIN_BYTES_PER_CORE : MIN_BYTES_PER_CORE);
    uint64_t totalBytes = 0;
    OP_CHECK_IF(!CheckedMul(info.totalLength, typeBytes, totalBytes),
                OP_LOGE(context, "Equal input byte size exceeds uint64_t range"), return ge::GRAPH_FAILED);
    uint64_t desiredCores = std::max<uint64_t>(
        1, std::min<uint64_t>(CeilDiv(totalBytes, minBytesPerCore), availableCores));
    const uint64_t blockAlignment = dtype == ge::DT_FLOAT16 ?
                                        FP16_BLOCK_ALIGN_ELEMENTS :
                                        (isByteInteger ? INT8_BLOCK_ALIGN_ELEMENTS : DEFAULT_BLOCK_ALIGN_ELEMENTS);
    uint64_t blockLength = 0;
    OP_CHECK_IF(!CheckedAlignUp(CeilDiv(info.totalLength, desiredCores), blockAlignment, blockLength),
                OP_LOGE(context, "Equal block length exceeds uint64_t range"), return ge::GRAPH_FAILED);
    uint32_t blockNum = static_cast<uint32_t>(CeilDiv(info.totalLength, blockLength));
    uint64_t tailBlockLength = info.totalLength - (static_cast<uint64_t>(blockNum) - 1) * blockLength;
    // General broadcast spends substantially more scalar work per output
    // element than the contiguous/vectorized paths.  Keep its final core even
    // when the tail is smaller than half a regular block; merging that tail
    // increases the critical-path core by almost 2x for small alternating
    // broadcasts such as [A,1,B,1,C] vs [1,D,1,E,1].
    if (info.mode != EQUAL_BROADCAST_GENERAL && blockNum > 1 && tailBlockLength < blockLength / 2) {
        --blockNum;
        tailBlockLength += blockLength;
    }

    const uint64_t alignElements = dtype == ge::DT_INT64 ? REPEAT_BYTES / sizeof(uint16_t) :
                                                           std::max<uint64_t>(1, REPEAT_BYTES / typeBytes);
    const uint64_t queueBufferNum = info.mode == EQUAL_BROADCAST_CONTIGUOUS ? CONTIGUOUS_QUEUE_BUFFER_NUM :
                                                                              BROADCAST_QUEUE_BUFFER_NUM;
    const uint64_t bytesPerElement = GetUbBytesPerElement(dtype, typeBytes, queueBufferNum);
    const bool isExtendedType = dtype == ge::DT_INT16 || dtype == ge::DT_INT64 || dtype == ge::DT_BOOL;
    const uint64_t reservedUbBytes = dtype == ge::DT_FLOAT16 ?
                                         FP16_UB_RESERVED_BYTES :
                                         (isExtendedType ? EXTENDED_TYPE_UB_RESERVED_BYTES : DEFAULT_UB_RESERVED_BYTES);
    const uint64_t usableUbSize = ubSize > reservedUbBytes ? ubSize - reservedUbBytes : 0;
    uint64_t tileLength = usableUbSize / bytesPerElement / alignElements * alignElements;
    OP_CHECK_IF(tileLength == 0, OP_LOGE(context, "UB is too small for Equal"), return ge::GRAPH_FAILED);
    if (dtype == ge::DT_FLOAT16) {
        const uint64_t tileCount = CeilDiv(blockLength, tileLength);
        uint64_t alignedTileLength = 0;
        OP_CHECK_IF(!CheckedAlignUp(CeilDiv(blockLength, tileCount), alignElements, alignedTileLength),
                    OP_LOGE(context, "Equal tile length exceeds uint64_t range"), return ge::GRAPH_FAILED);
        tileLength = std::min(alignedTileLength, FP16_MAX_TILE_ELEMENTS);
    } else if (dtype == ge::DT_INT64) {
        tileLength = std::min(tileLength, INT64_MAX_TILE_ELEMENTS);
    }

    TrySetSandwichMode(dtype, typeBytes, tileLength, platform, info);
    // The cached tail must remain unchanged while it is reused. The INT64 and
    // BOOL compute paths preserve the cached first operand; other paths may
    // use an input tensor as result scratch storage.
    if (dtype == ge::DT_INT64 || dtype == ge::DT_BOOL) {
        TrySetTailReuseMode(info);
    }
    if (info.mode == EQUAL_BROADCAST_SANDWICH) {
        const uint64_t cachedPlaneCount = std::max<uint64_t>(1, tileLength / info.fastPlaneLength);
        uint64_t cachedElements = 0;
        OP_CHECK_IF(!CheckedMul(cachedPlaneCount, info.fastPlaneLength, cachedElements) ||
                        !CheckedAlignUp(cachedElements, alignElements, tileLength),
                    OP_LOGE(context, "Equal sandwich tile length exceeds uint64_t range"), return ge::GRAPH_FAILED);
        blockNum = static_cast<uint32_t>(std::min<uint64_t>(availableCores, info.fastOuter));
        blockLength = CeilDiv(info.totalLength, blockNum);
        tailBlockLength = blockLength;
    } else if (info.mode == EQUAL_BROADCAST_TAIL_REUSE) {
        if (dtype == ge::DT_INT64) {
            const uint64_t maxTileLength = usableUbSize / bytesPerElement / alignElements * alignElements;
            tileLength = std::min(maxTileLength, INT64_TAIL_REUSE_MAX_TILE_ELEMENTS);
        }
        blockNum = static_cast<uint32_t>(std::min<uint64_t>(availableCores, info.fastOuter));
        blockLength = CeilDiv(info.totalLength, blockNum);
        tailBlockLength = blockLength;
    }

    tiling->totalLength = static_cast<int64_t>(info.totalLength);
    tiling->blockLength = static_cast<int64_t>(blockLength);
    tiling->tailBlockLength = static_cast<int64_t>(tailBlockLength);
    tiling->tileLength = static_cast<int64_t>(tileLength);
    tiling->blockNum = blockNum;
    tiling->broadcastMode = info.mode;
    tiling->rank = info.rank;
    tiling->fastBroadcastInput = info.fastBroadcastInput;
    tiling->x1Length = static_cast<int64_t>(info.x1Length);
    tiling->x2Length = static_cast<int64_t>(info.x2Length);
    for (uint32_t index = 0; index < EQUAL_MAX_BROADCAST_DIM; ++index) {
        tiling->outShape[index] = static_cast<int64_t>(info.outShape[index]);
        tiling->x1Stride[index] = static_cast<int64_t>(info.x1Stride[index]);
        tiling->x2Stride[index] = static_cast<int64_t>(info.x2Stride[index]);
    }
    tiling->fastOuter = static_cast<int64_t>(info.fastOuter);
    tiling->fastMiddle = static_cast<int64_t>(info.fastMiddle);
    tiling->fastTail = static_cast<int64_t>(info.fastTail);
    tiling->fastPlaneLength = static_cast<int64_t>(info.fastPlaneLength);
    tiling->fastSourceLength = static_cast<int64_t>(info.fastSourceLength);
    tiling->fastTmpBytes = info.fastTmpBytes;

    context->SetBlockDim(blockNum);
    const uint64_t scheduleMode = info.mode == EQUAL_BROADCAST_CONTIGUOUS ?
                                      static_cast<uint64_t>(ELEMENTWISE_TPL_SCH_MODE_0) :
                                      static_cast<uint64_t>(ELEMENTWISE_TPL_SCH_MODE_1);
    context->SetTilingKey(GET_TPL_TILING_KEY(scheduleMode));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EqualTilingParse([[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

} // namespace

IMPL_OP_OPTILING(Equal).Tiling(EqualTilingFunc).TilingParse<EqualCompileInfo>(EqualTilingParse);
} // namespace optiling
