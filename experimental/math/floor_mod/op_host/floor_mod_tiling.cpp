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
 * \file floor_mod_tiling.cpp
 * \brief Shape-topology and cost driven FloorMod tiling.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"

#include "../op_kernel/floor_mod_tiling_data.h"
#include "../op_kernel/floor_mod_tiling_key.h"

using namespace ge;

namespace FloorModNs {

constexpr uint32_t FLOOR_MOD_FP32_BLOCK_ELEMS = 8U;
constexpr uint32_t FLOOR_MOD_GATHER_MASK_ELEMS = 64U;
constexpr uint32_t FLOOR_MOD_MAX_PATTERN_ELEMS = 2040U; // repeat stride is uint8 DataBlocks
constexpr uint32_t FLOOR_MOD_MAX_REPEAT_ROWS = 255U;
constexpr uint64_t FLOOR_MOD_ROW_TARGET_CORE_ELEMS = 2048U;
constexpr uint64_t FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS = 8192U;
constexpr uint64_t FLOOR_MOD_SCALAR_FULL_WAVE_MIN_BYTES = 32U * FLOOR_MOD_GATHER_MASK_ELEMS * sizeof(float);
constexpr uint64_t FLOOR_MOD_MIN_CORE_AMORTIZATION_ELEMS = 4U * FLOOR_MOD_GATHER_MASK_ELEMS;
constexpr uint64_t FLOOR_MOD_SMALL_DENSE_MAX_ELEMENTS = FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS;
// Native double storage kernel processes arbitrarily large tensors in a
// per-core loop.  Keep each UB tile bounded while allowing totalElements to
// exceed one tile.
// FP64 storage uses two uint64 buffers, three FP32 work buffers, and three
// 32-bit scratch slabs for the vectorized bit conversions. The fast path
// covers ordinary magnitudes; exceptional elements are corrected from their
// exact IEEE-754 significands. Keep the tile bounded so the extra scratch
// stays below the 910B UB guard.
constexpr uint32_t FLOOR_MOD_FP64_STORAGE_TILE = 2048U;
// FP64 storage conversion has a fixed per-block launch/event cost.  For
// small tensors, keeping roughly this amount of work per core avoids the
// single-core tail seen when the large-tensor tile limit is used as the
// core-count heuristic.  The value is deliberately a class-wide target,
// rather than a case-specific shape threshold.
constexpr uint32_t FLOOR_MOD_FP64_TARGET_CORE_ELEMS = 512U;
constexpr uint64_t FLOOR_MOD_UB_GUARD_BYTES = 1024U;
constexpr uint64_t FLOOR_MOD_UB_BANK_GUARD_BYTES = 256U;
constexpr uint64_t FLOOR_MOD_HIGH_LEVEL_API_GUARD_BYTES = 8192U;
constexpr uint64_t FLOOR_MOD_DMA_COMMAND_COST = 96U;
constexpr uint64_t FLOOR_MOD_DMA_INTERNAL_BLOCK_COST = 4U;

static uint64_t CeilDiv(uint64_t x, uint64_t y) { return y == 0U ? 0U : (x + y - 1U) / y; }

struct UbLayout {
    uint32_t blockBytes = 0U;

    uint64_t Align(uint64_t bytes) const { return blockBytes == 0U ? 0U : CeilDiv(bytes, blockBytes) * blockBytes; }

    uint32_t AlignElements(uint64_t elements, uint32_t dtypeBytes) const
    {
        if (blockBytes == 0U || dtypeBytes == 0U) {
            return 0U;
        }
        const uint32_t elementsPerBlock = blockBytes / dtypeBytes;
        const uint32_t alignment = std::max<uint32_t>(FLOOR_MOD_FP32_BLOCK_ELEMS, elementsPerBlock);
        return static_cast<uint32_t>(CeilDiv(elements, alignment) * alignment);
    }
};

static bool IsValidLayout(const UbLayout& layout, uint32_t dtypeBytes)
{
    return layout.blockBytes != 0U && dtypeBytes != 0U;
}

static uint32_t ElementsPerBlock(const UbLayout& layout, uint32_t dtypeBytes)
{
    if (dtypeBytes == 0U) {
        return 0U;
    }
    return layout.blockBytes / std::max(dtypeBytes, 1U);
}

static uint64_t MaxElementsForDma(uint32_t dtypeBytes)
{
    if (dtypeBytes == 0U) {
        return 0U;
    }
    return std::numeric_limits<uint32_t>::max() / std::max(dtypeBytes, 1U);
}

static uint64_t SafeRemainder(uint64_t value, uint64_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return value % std::max<uint64_t>(divisor, 1U);
}

static uint64_t ScalarBroadcastUbBytes(uint32_t tRowElems, uint32_t fp32RowElems, uint32_t floorTmpBytes,
                                       uint32_t dtypeBytes, const UbLayout& ubLayout, bool isFp32, bool isBf16,
                                       uint32_t denseBuffers = 1U)
{
    uint64_t bytes = denseBuffers * ubLayout.Align(static_cast<uint64_t>(tRowElems) * dtypeBytes);
    if (!isFp32) {
        bytes += ubLayout.Align(static_cast<uint64_t>(fp32RowElems) * sizeof(float));
    }
    if (isBf16) {
        bytes += ubLayout.blockBytes;
    }
    bytes += FLOOR_MOD_GATHER_MASK_ELEMS * sizeof(float);
    bytes += 2U * ubLayout.Align(static_cast<uint64_t>(fp32RowElems) * sizeof(float));
    bytes += ubLayout.Align(std::max(floorTmpBytes, ubLayout.blockBytes));
    return bytes;
}

static uint64_t UnderfilledCorePenalty(uint64_t totalElements, uint64_t activeCores)
{
    const uint64_t amortizationElements = activeCores * FLOOR_MOD_MIN_CORE_AMORTIZATION_ELEMS;
    if (totalElements >= amortizationElements) {
        return 0U;
    }
    return CeilDiv(amortizationElements - totalElements, FLOOR_MOD_GATHER_MASK_ELEMS);
}

static bool SafeMul(uint64_t a, uint64_t b, uint64_t& out)
{
    if (a != 0U && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

static bool ProductDims(const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& dims, uint32_t rank, uint64_t& product)
{
    product = 1U;
    for (uint32_t i = 0U; i < rank; ++i) {
        if (!SafeMul(product, dims[i], product)) {
            return false;
        }
    }
    return true;
}

static void CalcBroadcastStrides(const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& dims, uint32_t rank,
                                 std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& strides)
{
    strides.fill(0U);
    uint64_t acc = 1U;
    for (uint32_t i = rank; i-- > 0U;) {
        if (dims[i] != 1U) {
            strides[i] = acc;
        }
        acc *= dims[i];
    }
}

struct ReuseDecomp {
    bool valid = false;
    bool posContiguous = true;
    uint64_t pos = 1U;
    uint64_t e1 = 1U;
    uint64_t d = 1U;
    uint64_t e2 = 1U;
    std::array<uint32_t, FLOOR_MOD_MAX_DIGITS> posDims{};
    uint32_t posDigits = 0U;
};

// Dim classes from the resident seed's point of view:
// F: both operands full, G: seed broadcasts, S: streamed operand broadcasts.
static ReuseDecomp DecomposeReuse(const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& seed,
                                  const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& dense,
                                  const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& out, uint32_t rank)
{
    ReuseDecomp result;
    std::array<char, FLOOR_MOD_MAX_BROADCAST_DIM> type{};
    for (uint32_t i = 0U; i < rank; ++i) {
        if (seed[i] == out[i] && dense[i] == out[i]) {
            type[i] = 'F';
        } else if (seed[i] == 1U) {
            type[i] = 'G';
        } else {
            type[i] = 'S';
        }
    }

    int32_t right = static_cast<int32_t>(rank) - 1;
    while (right >= 0 && type[static_cast<uint32_t>(right)] == 'G') {
        result.e2 *= out[static_cast<uint32_t>(right--)];
    }
    while (right >= 0 && type[static_cast<uint32_t>(right)] == 'F') {
        result.d *= out[static_cast<uint32_t>(right--)];
    }
    while (right >= 0 && type[static_cast<uint32_t>(right)] == 'G') {
        result.e1 *= out[static_cast<uint32_t>(right--)];
    }
    for (int32_t i = 0; i <= right; ++i) {
        result.pos *= out[static_cast<uint32_t>(i)];
        result.posDims[result.posDigits++] = static_cast<uint32_t>(i);
        result.posContiguous = result.posContiguous && type[static_cast<uint32_t>(i)] == 'F';
    }
    for (int32_t i = right + 1; i < static_cast<int32_t>(rank); ++i) {
        if (type[static_cast<uint32_t>(i)] == 'S') {
            return result;
        }
    }
    result.valid = true;
    return result;
}

struct CrossDecomp {
    bool valid = false;
    uint64_t outer = 1U;
    uint64_t a = 1U;
    uint64_t m = 1U;
    uint64_t b = 1U;
    uint64_t d = 1U;
};

// Crossed topology: [common outer][seed-only A][common M][dense-only B][common D].
static CrossDecomp DecomposeCrossed(const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& seed,
                                    const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& dense,
                                    const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& out, uint32_t rank)
{
    CrossDecomp result;
    std::array<char, FLOOR_MOD_MAX_BROADCAST_DIM> type{};
    for (uint32_t i = 0U; i < rank; ++i) {
        if (seed[i] == out[i] && dense[i] == out[i]) {
            type[i] = 'F';
        } else if (seed[i] == out[i] && dense[i] == 1U) {
            type[i] = 'S';
        } else if (seed[i] == 1U && dense[i] == out[i]) {
            type[i] = 'G';
        } else {
            return result;
        }
    }

    int32_t i = static_cast<int32_t>(rank) - 1;
    while (i >= 0 && type[static_cast<uint32_t>(i)] == 'F') {
        result.d *= out[static_cast<uint32_t>(i--)];
    }
    while (i >= 0 && type[static_cast<uint32_t>(i)] == 'G') {
        result.b *= out[static_cast<uint32_t>(i--)];
    }
    while (i >= 0 && type[static_cast<uint32_t>(i)] == 'F') {
        result.m *= out[static_cast<uint32_t>(i--)];
    }
    while (i >= 0 && type[static_cast<uint32_t>(i)] == 'S') {
        result.a *= out[static_cast<uint32_t>(i--)];
    }
    for (; i >= 0; --i) {
        if (type[static_cast<uint32_t>(i)] != 'F') {
            return result;
        }
        result.outer *= out[static_cast<uint32_t>(i)];
    }
    result.valid = result.a > 1U && result.b > 1U;
    return result;
}

static ge::graphStatus BuildBroadcastDimensions(gert::TilingContext* context, const gert::Shape& sx1,
                                                const gert::Shape& sx2, uint32_t rank,
                                                std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x1,
                                                std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x2,
                                                std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& out,
                                                uint64_t& outputElements)
{
    const uint32_t off1 = rank - static_cast<uint32_t>(sx1.GetDimNum());
    const uint32_t off2 = rank - static_cast<uint32_t>(sx2.GetDimNum());
    outputElements = 1U;
    for (uint32_t i = 0U; i < rank; ++i) {
        const int64_t d1 = i < off1 ? 1 : sx1.GetDim(static_cast<size_t>(i - off1));
        const int64_t d2 = i < off2 ? 1 : sx2.GetDim(static_cast<size_t>(i - off2));
        OP_CHECK_IF(d1 < 0 || d2 < 0, OP_LOGE(context, "FloorMod input shape has a negative dim."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(d1 != d2 && d1 != 1 && d2 != 1, OP_LOGE(context, "FloorMod input shapes are not broadcastable."),
                    return ge::GRAPH_FAILED);
        x1[i] = static_cast<uint64_t>(d1);
        x2[i] = static_cast<uint64_t>(d2);
        out[i] = std::max(x1[i], x2[i]);
        if (out[i] == 0U) {
            outputElements = 0U;
        } else if (outputElements != 0U) {
            OP_CHECK_IF(outputElements > std::numeric_limits<uint64_t>::max() / out[i],
                        OP_LOGE(context, "FloorMod broadcast output is too large."), return ge::GRAPH_FAILED);
            outputElements *= out[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetBroadcastShapes(gert::TilingContext* context,
                                          std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x1,
                                          std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x2,
                                          std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& out, uint32_t& rank)
{
    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    const gert::StorageShape* x2Shape = context->GetInputShape(1);
    const gert::StorageShape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // The logical layout is retained as origin shape even when an OP-plugin
    // transformation flattens the contiguous storage view (notably FP64).
    // Broadcast indexing must follow the logical layout; storage is used only
    // to prove that the backing buffers contain the expected number of items.
    const gert::Shape& sx1 = x1Shape->GetOriginShape();
    const gert::Shape& sx2 = x2Shape->GetOriginShape();
    const gert::Shape& x1Storage = x1Shape->GetStorageShape();
    const gert::Shape& x2Storage = x2Shape->GetStorageShape();
    const gert::Shape& yStorage = yShape->GetStorageShape();
    rank = static_cast<uint32_t>(std::max(sx1.GetDimNum(), sx2.GetDimNum()));
    OP_CHECK_IF(rank > FLOOR_MOD_MAX_BROADCAST_DIM, OP_LOGE(context, "FloorMod broadcast rank must not exceed 8."),
                return ge::GRAPH_FAILED);
    x1.fill(1U);
    x2.fill(1U);
    out.fill(1U);
    uint64_t outputElements = 1U;
    OP_CHECK_IF(BuildBroadcastDimensions(context, sx1, sx2, rank, x1, x2, out, outputElements) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to resolve broadcast dimensions."), return ge::GRAPH_FAILED);
    uint64_t x1Elements = 0U;
    uint64_t x2Elements = 0U;
    OP_CHECK_IF(!ProductDims(x1, rank, x1Elements) || !ProductDims(x2, rank, x2Elements),
                OP_LOGE(context, "FloorMod input shape product overflow."), return ge::GRAPH_FAILED);
    const int64_t x1StorageElements = x1Storage.GetShapeSize();
    const int64_t x2StorageElements = x2Storage.GetShapeSize();
    OP_CHECK_IF(x1StorageElements < 0 || static_cast<uint64_t>(x1StorageElements) != x1Elements ||
                    x2StorageElements < 0 || static_cast<uint64_t>(x2StorageElements) != x2Elements,
                OP_LOGE(context, "FloorMod input storage element count mismatch."), return ge::GRAPH_FAILED);
    // The Torch OP-plugin may flatten the AI Core output storage shape. The
    // kernel writes a contiguous broadcast result, so only its element count
    // needs to agree with the input-derived logical layout.
    const int64_t storageElements = yStorage.GetShapeSize();
    OP_CHECK_IF(storageElements < 0 || static_cast<uint64_t>(storageElements) != outputElements,
                OP_LOGE(context, "FloorMod output element count mismatch."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void QueryFloorTmp(uint64_t count, uint32_t& maxBytes, uint32_t& minBytes)
{
    std::vector<int64_t> shape = {static_cast<int64_t>(count)};
    AscendC::GetFloorMaxMinTmpSize(ge::Shape(shape), sizeof(float), false, maxBytes, minBytes);
}

static uint32_t QueryBroadcastTmp(const platform_ascendc::PlatformAscendC& platform, uint32_t dTile, uint32_t e2Tile,
                                  uint32_t dtypeBytes = sizeof(float))
{
    uint32_t maxBytes = 0U;
    uint32_t minBytes = 0U;
    const ge::Shape srcShape({static_cast<int64_t>(dTile), 1});
    const ge::Shape dstShape({static_cast<int64_t>(dTile), static_cast<int64_t>(e2Tile)});
    AscendC::GetBroadCastMaxMinTmpSize(platform, srcShape, dstShape, dtypeBytes, false, maxBytes, minBytes);
    return minBytes;
}

static uint32_t QueryTrailingBroadcastTmp(const platform_ascendc::PlatformAscendC& platform, uint32_t rows,
                                          uint32_t width, uint32_t dtypeBytes = sizeof(float))
{
    uint32_t maxBytes = 0U;
    uint32_t minBytes = 0U;
    const ge::Shape srcShape({1, static_cast<int64_t>(width)});
    const ge::Shape dstShape({static_cast<int64_t>(rows), static_cast<int64_t>(width)});
    AscendC::GetBroadCastMaxMinTmpSize(platform, srcShape, dstShape, dtypeBytes, false, maxBytes, minBytes);
    return minBytes;
}

struct StandardCandidate {
    bool valid = false;
    bool rowOwned = false;
    bool packed = false;
    bool materialized = false;
    bool compactPos = false;
    bool denseTail = false;
    bool compactRows = false;
    bool compactRepeat = false;
    bool scalarDoubleBuffered = false;
    uint32_t dTile = 1U;
    uint32_t e2Tile = 1U;
    uint32_t dCount = 1U;
    uint32_t e2Count = 1U;
    uint32_t rowPartitions = 1U;
    uint32_t batchRows = 1U;
    uint32_t posTile = 1U;
    uint32_t patternElems = 1U;
    uint32_t tRowElems = 1U;
    uint32_t fp32RowElems = FLOOR_MOD_FP32_BLOCK_ELEMS;
    uint32_t floorTmpBytes = 0U;
    uint32_t broadcastTmpBytes = 0U;
    uint32_t ubUsedBytes = 0U;
    uint64_t totalTasks = 1U;
    uint64_t score = std::numeric_limits<uint64_t>::max();
};

static bool IsScalarCandidateInputValid(uint64_t totalElements, uint32_t tile, uint32_t dtypeBytes,
                                        uint32_t availableCores, const UbLayout& ubLayout)
{
    return IsValidLayout(ubLayout, dtypeBytes) && totalElements != 0U && tile != 0U && availableCores != 0U &&
           tile <= FLOOR_MOD_MAX_REPEAT_ROWS * FLOOR_MOD_GATHER_MASK_ELEMS + FLOOR_MOD_GATHER_MASK_ELEMS - 1U &&
           CeilDiv(totalElements, tile) <= std::numeric_limits<uint32_t>::max();
}

static void PopulateScalarCandidate(StandardCandidate& c, uint64_t totalElements, uint32_t tile, uint32_t dtypeBytes,
                                    const UbLayout& ubLayout)
{
    c.dTile = 1U;
    c.e2Tile = tile;
    c.dCount = 1U;
    c.e2Count = static_cast<uint32_t>(CeilDiv(totalElements, tile));
    c.totalTasks = c.e2Count;
    c.patternElems = tile;
    c.tRowElems = ubLayout.AlignElements(tile, dtypeBytes);
    c.fp32RowElems = ubLayout.AlignElements(tile, sizeof(float));
    c.packed = true;
}

static bool SelectScalarFloor(const StandardCandidate& c, uint32_t dtypeBytes, bool isFp32, bool isBf16,
                              uint64_t ubLimit, const UbLayout& ubLayout, bool doubleBuffered, uint32_t& floorChosen,
                              uint64_t& used)
{
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    QueryFloorTmp(c.fp32RowElems, floorMax, floorMin);
    floorChosen = floorMax;
    used = ScalarBroadcastUbBytes(c.tRowElems, c.fp32RowElems, floorChosen, dtypeBytes, ubLayout, isFp32, isBf16,
                                  doubleBuffered ? 2U : 1U);
    if (used > ubLimit) {
        floorChosen = floorMin;
        used = ScalarBroadcastUbBytes(c.tRowElems, c.fp32RowElems, floorChosen, dtypeBytes, ubLayout, isFp32, isBf16,
                                      doubleBuffered ? 2U : 1U);
    }
    return used <= ubLimit && used <= std::numeric_limits<uint32_t>::max();
}

static uint64_t ScoreScalarCandidate(const StandardCandidate& c, uint64_t totalElements, uint32_t tile,
                                     uint32_t activeCores, bool doubleBuffered)
{
    const uint64_t baseTasks = c.totalTasks / activeCores;
    const uint64_t extraTasks = c.totalTasks % activeCores;
    const uint64_t criticalTasks = baseTasks + (extraTasks == 0U ? 0U : 1U);
    const uint64_t criticalElements = std::min<uint64_t>(totalElements, criticalTasks * tile);
    const uint64_t idealElements = CeilDiv(totalElements, activeCores);
    const uint64_t imbalanceElements = criticalElements > idealElements ? criticalElements - idealElements : 0U;
    const uint64_t tailWaste = static_cast<uint64_t>(c.e2Count) * tile - totalElements;
    const uint64_t transferUnits = CeilDiv(criticalElements, FLOOR_MOD_GATHER_MASK_ELEMS);
    const uint64_t vectorUnits = CeilDiv(criticalElements, FLOOR_MOD_FP32_BLOCK_ELEMS);
    const uint64_t commandUnits = criticalTasks * 8U;
    const uint64_t pipelineUnits = doubleBuffered ? (transferUnits * 12U + vectorUnits * 8U + commandUnits) :
                                                    (transferUnits * 24U + vectorUnits * 16U + commandUnits +
                                                     CeilDiv(totalElements, FLOOR_MOD_GATHER_MASK_ELEMS));
    return 96U + pipelineUnits + 2U * CeilDiv(imbalanceElements, FLOOR_MOD_GATHER_MASK_ELEMS) +
           CeilDiv(tailWaste, static_cast<uint64_t>(activeCores) * FLOOR_MOD_GATHER_MASK_ELEMS);
}

static StandardCandidate BuildScalarBroadcastCandidate(uint64_t totalElements, uint32_t tile, uint32_t dtypeBytes,
                                                       bool isFp32, bool isBf16, uint32_t availableCores,
                                                       uint64_t ubLimit, const UbLayout& ubLayout, bool doubleBuffered)
{
    StandardCandidate c;
    if (!IsScalarCandidateInputValid(totalElements, tile, dtypeBytes, availableCores, ubLayout)) {
        return c;
    }
    PopulateScalarCandidate(c, totalElements, tile, dtypeBytes, ubLayout);
    const uint32_t activeCores = static_cast<uint32_t>(std::min<uint64_t>(availableCores, c.totalTasks));
    uint32_t floorChosen = 0U;
    uint64_t used = 0U;
    if (!SelectScalarFloor(c, dtypeBytes, isFp32, isBf16, ubLimit, ubLayout, doubleBuffered, floorChosen, used)) {
        return c;
    }
    c.floorTmpBytes = floorChosen;
    c.ubUsedBytes = static_cast<uint32_t>(used);
    c.scalarDoubleBuffered = doubleBuffered;
    c.valid = true;
    c.score = ScoreScalarCandidate(c, totalElements, tile, activeCores, doubleBuffered);
    return c;
}

static uint32_t FindMaxScalarTile(uint64_t totalElements, uint32_t dtypeBytes, bool isFp32, bool isBf16,
                                  uint32_t availableCores, uint64_t ubLimit, const UbLayout& ubLayout,
                                  bool doubleBuffered, uint32_t repeatLimit)
{
    uint32_t best = 0U;
    uint32_t low = 1U;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(totalElements, repeatLimit));
    while (low <= high) {
        const uint32_t mid = low + (high - low) / 2U;
        const auto candidate = BuildScalarBroadcastCandidate(totalElements, mid, dtypeBytes, isFp32, isBf16,
                                                             availableCores, ubLimit, ubLayout, doubleBuffered);
        if (candidate.valid && candidate.totalTasks > availableCores) {
            best = mid;
            low = mid + 1U;
        } else {
            high = mid - 1U;
        }
    }
    return best;
}

static void ConsiderScalarCandidate(StandardCandidate& best, uint64_t totalElements, uint32_t tile, uint32_t dtypeBytes,
                                    bool isFp32, bool isBf16, uint32_t availableCores, uint64_t ubLimit,
                                    const UbLayout& ubLayout)
{
    for (bool doubleBuffered : {true, false}) {
        if (doubleBuffered && dtypeBytes > sizeof(float)) {
            continue;
        }
        const auto candidate = BuildScalarBroadcastCandidate(totalElements, tile, dtypeBytes, isFp32, isBf16,
                                                             availableCores, ubLimit, ubLayout, doubleBuffered);
        if (candidate.valid && candidate.totalTasks > availableCores && (!best.valid || candidate.score < best.score)) {
            best = candidate;
        }
    }
}

static void ConsiderScalarWaveTiles(StandardCandidate& best, uint64_t totalElements, uint32_t maxLegalTile,
                                    uint32_t dtypeBytes, bool isFp32, bool isBf16, uint32_t availableCores,
                                    uint64_t ubLimit, const UbLayout& ubLayout)
{
    if (maxLegalTile == 0U) {
        return;
    }
    const auto consider = [&](uint32_t tile) {
        ConsiderScalarCandidate(best, totalElements, tile, dtypeBytes, isFp32, isBf16, availableCores, ubLimit,
                                ubLayout);
    };
    consider(maxLegalTile);
    const uint64_t minTasks = CeilDiv(totalElements, maxLegalTile);
    const uint64_t minWaves = CeilDiv(minTasks, availableCores);
    for (uint64_t waves = minWaves; waves <= minWaves + 2U; ++waves) {
        const uint64_t targetTasks = waves * availableCores;
        const uint32_t balanced = static_cast<uint32_t>(
            std::min<uint64_t>(maxLegalTile, ubLayout.AlignElements(CeilDiv(totalElements, targetTasks), dtypeBytes)));
        consider(balanced);
        const uint32_t elementsPerBlock = ElementsPerBlock(ubLayout, dtypeBytes);
        if (elementsPerBlock != 0U && balanced > elementsPerBlock) {
            consider(balanced - elementsPerBlock);
        }
    }
}

static StandardCandidate FindMultiWaveScalarBroadcastCandidate(uint64_t totalElements, uint32_t dtypeBytes, bool isFp32,
                                                               bool isBf16, uint32_t availableCores, uint64_t ubLimit,
                                                               const UbLayout& ubLayout)
{
    StandardCandidate best;
    if (availableCores == 0U || totalElements <= availableCores) {
        return best;
    }

    const uint32_t oneWaveTile = static_cast<uint32_t>(
        std::min<uint64_t>(totalElements, ubLayout.AlignElements(CeilDiv(totalElements, availableCores), dtypeBytes)));
    const auto oneWave = BuildScalarBroadcastCandidate(totalElements, oneWaveTile, dtypeBytes, isFp32, isBf16,
                                                       availableCores, ubLimit, ubLayout, true);
    if (oneWave.valid && oneWave.totalTasks <= availableCores) {
        return best;
    }

    const uint32_t repeatLimit = FLOOR_MOD_MAX_REPEAT_ROWS * FLOOR_MOD_GATHER_MASK_ELEMS + FLOOR_MOD_GATHER_MASK_ELEMS -
                                 1U;
    const uint32_t maxLegalTile[2] = {FindMaxScalarTile(totalElements, dtypeBytes, isFp32, isBf16, availableCores,
                                                        ubLimit, ubLayout, true, repeatLimit),
                                      FindMaxScalarTile(totalElements, dtypeBytes, isFp32, isBf16, availableCores,
                                                        ubLimit, ubLayout, false, repeatLimit)};
    if (maxLegalTile[0] == 0U && maxLegalTile[1] == 0U) {
        return best;
    }

    for (uint32_t mode = 0U; mode < 2U; ++mode) {
        ConsiderScalarWaveTiles(best, totalElements, maxLegalTile[mode], dtypeBytes, isFp32, isBf16, availableCores,
                                ubLimit, ubLayout);
    }
    return best;
}

struct CompactRowBatch {
    uint32_t batch = 0U;
    uint32_t tElems = 0U;
    uint32_t fpElems = 0U;
    uint32_t floorTmp = 0U;
    uint32_t broadcastTmp = 0U;
    uint64_t used = 0U;
};

static bool BuildCompactRowBatch(CompactRowBatch& candidate, const ReuseDecomp& dec, uint32_t batch,
                                 uint32_t dtypeBytes, bool isFp32, uint64_t ubLimit,
                                 const platform_ascendc::PlatformAscendC& platform, const UbLayout& ubLayout,
                                 bool repeatSeed)
{
    const uint32_t d = static_cast<uint32_t>(dec.d);
    const uint32_t tRowElems = ubLayout.AlignElements(d, dtypeBytes);
    const uint32_t fpRowElems = ubLayout.AlignElements(d, sizeof(float));
    const uint32_t seedTElems = ubLayout.AlignElements(d, dtypeBytes);
    const uint32_t seedFpElems = ubLayout.AlignElements(d, sizeof(float));
    const uint64_t logicalElements = static_cast<uint64_t>(batch) * d;
    const uint64_t tElements = repeatSeed ? static_cast<uint64_t>(batch) * tRowElems :
                                            ubLayout.AlignElements(logicalElements, dtypeBytes);
    const uint64_t fpElements = repeatSeed ? static_cast<uint64_t>(batch) * fpRowElems :
                                             ubLayout.AlignElements(logicalElements, sizeof(float));
    if (logicalElements > std::numeric_limits<uint32_t>::max() || tElements > std::numeric_limits<uint32_t>::max() ||
        fpElements > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint64_t tBytes = ubLayout.Align(tElements * dtypeBytes);
    const uint64_t fpBytes = ubLayout.Align(fpElements * sizeof(float));
    uint64_t fixed = 2U * tBytes + ubLayout.Align(static_cast<uint64_t>(seedTElems) * dtypeBytes) + fpBytes +
                     FLOOR_MOD_HIGH_LEVEL_API_GUARD_BYTES;
    fixed += isFp32 ? 0U : ubLayout.Align(static_cast<uint64_t>(seedFpElems) * sizeof(float)) + fpBytes;
    fixed += repeatSeed ? 0U : ubLayout.Align(static_cast<uint64_t>(batch) * seedFpElems * sizeof(float));
    if (fixed > ubLimit) {
        return false;
    }
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    QueryFloorTmp(static_cast<uint32_t>(fpElements), floorMax, floorMin);
    const uint32_t broadcastTmp = repeatSeed ? 0U : QueryTrailingBroadcastTmp(platform, batch, seedFpElems);
    const uint32_t floorChosen = fixed + ubLayout.Align(std::max(floorMax, broadcastTmp)) <= ubLimit ? floorMax :
                                                                                                       floorMin;
    const uint64_t used = fixed + ubLayout.Align(std::max(floorChosen, broadcastTmp));
    candidate = {batch, static_cast<uint32_t>(tElements), static_cast<uint32_t>(fpElements), floorChosen, broadcastTmp,
                 used};
    return used <= ubLimit;
}

static CompactRowBatch FindCompactRowBatch(const ReuseDecomp& dec, uint32_t rowPartitions, uint32_t dtypeBytes,
                                           bool isFp32, uint64_t ubLimit,
                                           const platform_ascendc::PlatformAscendC& platform, const UbLayout& ubLayout,
                                           bool repeatSeed)
{
    CompactRowBatch best;
    const uint32_t d = static_cast<uint32_t>(dec.d);
    const uint64_t maxRowsPerPart = CeilDiv(dec.e1, rowPartitions);
    const uint64_t maxElementsByDma = MaxElementsForDma(dtypeBytes);
    uint32_t low = 1U;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(
        std::min<uint64_t>(maxRowsPerPart, std::numeric_limits<uint16_t>::max()), d == 0U ? 0U : maxElementsByDma / d));
    while (low <= high) {
        const uint32_t batch = low + (high - low) / 2U;
        CompactRowBatch candidate;
        if (BuildCompactRowBatch(candidate, dec, batch, dtypeBytes, isFp32, ubLimit, platform, ubLayout, repeatSeed)) {
            best = candidate;
            low = batch + 1U;
        } else {
            high = batch - 1U;
        }
    }
    return best;
}

static uint64_t ScoreCompactRowCandidate(const ReuseDecomp& dec, uint32_t rowPartitions, uint32_t batchRows,
                                         uint32_t fpRowElems, bool isFp32, bool repeatSeed, uint32_t availableCores)
{
    const uint64_t criticalRows = CeilDiv(dec.e1, rowPartitions);
    const uint64_t batches = CeilDiv(criticalRows, batchRows);
    const uint64_t criticalElements = criticalRows * fpRowElems;
    uint64_t totalRowElements = 0U;
    if (!SafeMul(dec.e1, dec.d, totalRowElements)) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t targetPartitions = std::min<uint64_t>(
        availableCores, std::max<uint64_t>(1U, CeilDiv(totalRowElements, FLOOR_MOD_ROW_TARGET_CORE_ELEMS)));
    const uint64_t partitionDeviation = rowPartitions > targetPartitions ? rowPartitions - targetPartitions :
                                                                           targetPartitions - rowPartitions;
    const uint64_t seedCommands = isFp32 ? 1U : 2U;
    const uint64_t batchCommands = repeatSeed ? (isFp32 ? 5U : 8U) : (isFp32 ? 5U : 7U);
    const uint64_t dmaIssue = (1U + 2U * batches) * (FLOOR_MOD_DMA_COMMAND_COST / 3U);
    const uint64_t repeatLaneElems = CeilDiv(dec.d, FLOOR_MOD_GATHER_MASK_ELEMS) * FLOOR_MOD_GATHER_MASK_ELEMS;
    const uint64_t repeatLaneWaste = repeatSeed && repeatLaneElems > fpRowElems ?
                                         3U * CeilDiv(criticalRows * (repeatLaneElems - fpRowElems),
                                                      FLOOR_MOD_GATHER_MASK_ELEMS) :
                                         0U;
    return seedCommands + batches * (32U + batchCommands) + dmaIssue + rowPartitions + CeilDiv(criticalElements, 64U) +
           partitionDeviation + repeatLaneWaste;
}

// A resident [D] vector reused across a contiguous [E1,D] matrix should use
// one GM2UB and one UB2GM transfer per batch. The materialized layout keeps
// dense data compact, broadcasts an aligned [1,Dpad] seed slab, then gathers
// it to [batch,D] for contiguous arithmetic. The repeat layout keeps aligned
// row pitches and reuses the seed with zero repeat stride. Neither layout uses
// an UB2UB DataCopy, and selection is derived from shape and resource costs.
static StandardCandidate BuildCompactRowCandidate(const ReuseDecomp& dec, uint32_t rowPartitions, uint32_t dtypeBytes,
                                                  bool isFp32, uint32_t availableCores, uint64_t ubLimit,
                                                  const platform_ascendc::PlatformAscendC& platform,
                                                  const UbLayout& ubLayout, bool repeatSeed)
{
    StandardCandidate c;
    if (!IsValidLayout(ubLayout, dtypeBytes) || dec.pos != 1U || dec.e2 != 1U || dec.e1 <= 1U || rowPartitions == 0U ||
        dtypeBytes > sizeof(float) || availableCores == 0U || dec.d > std::numeric_limits<uint32_t>::max() ||
        rowPartitions > std::min<uint64_t>(dec.e1, availableCores)) {
        return c;
    }

    const uint32_t d = static_cast<uint32_t>(dec.d);
    const uint32_t fpRowElems = ubLayout.AlignElements(d, sizeof(float));
    const uint32_t elementsPerBlock = ElementsPerBlock(ubLayout, dtypeBytes);
    if (elementsPerBlock == 0U ||
        ubLayout.AlignElements(d, dtypeBytes) / elementsPerBlock > std::numeric_limits<uint8_t>::max() ||
        fpRowElems / FLOOR_MOD_FP32_BLOCK_ELEMS > std::numeric_limits<uint8_t>::max()) {
        return c;
    }
    const auto best = FindCompactRowBatch(dec, rowPartitions, dtypeBytes, isFp32, ubLimit, platform, ubLayout,
                                          repeatSeed);
    if (best.batch == 0U) {
        return c;
    }

    c.valid = true;
    c.rowOwned = true;
    c.compactRows = true;
    c.compactRepeat = repeatSeed;
    c.dTile = d;
    c.dCount = 1U;
    c.e2Tile = 1U;
    c.e2Count = 1U;
    c.rowPartitions = rowPartitions;
    c.batchRows = best.batch;
    c.patternElems = d;
    c.tRowElems = best.tElems;
    c.fp32RowElems = best.fpElems;
    c.floorTmpBytes = best.floorTmp;
    c.broadcastTmpBytes = best.broadcastTmp;
    c.ubUsedBytes = static_cast<uint32_t>(best.used);
    c.totalTasks = rowPartitions;

    c.score = ScoreCompactRowCandidate(dec, rowPartitions, best.batch, fpRowElems, isFp32, repeatSeed, availableCores);
    if (c.score == std::numeric_limits<uint64_t>::max()) {
        return StandardCandidate{};
    }
    return c;
}

struct DenseTailResources {
    uint32_t broadcastTmp = 0U;
    uint32_t floorTmp = 0U;
    uint64_t used = 0U;
};

static DenseTailResources QueryDenseTailResources(uint32_t dTile, uint32_t e2, uint32_t dtypeBytes, bool isFp32,
                                                  bool doubleBuffered, uint64_t ubLimit,
                                                  const platform_ascendc::PlatformAscendC& platform,
                                                  const UbLayout& ubLayout, uint32_t fpRowElems)
{
    DenseTailResources result;
    const uint32_t tRowElems = ubLayout.AlignElements(static_cast<uint64_t>(dTile) * e2, dtypeBytes);
    const uint64_t tBytes = ubLayout.Align(static_cast<uint64_t>(tRowElems) * dtypeBytes);
    const uint64_t fpBytes = ubLayout.Align(static_cast<uint64_t>(fpRowElems) * sizeof(float));
    const uint64_t seedBytes = ubLayout.Align(static_cast<uint64_t>(dTile) * dtypeBytes);
    const uint32_t queueBuffers = doubleBuffered ? 6U : 3U;
    const uint64_t fixed = queueBuffers * tBytes + seedBytes + (isFp32 ? fpBytes : 3U * fpBytes) +
                           FLOOR_MOD_HIGH_LEVEL_API_GUARD_BYTES;
    if (fixed > ubLimit) {
        return result;
    }
    result.broadcastTmp = QueryBroadcastTmp(platform, dTile, e2, dtypeBytes);
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    QueryFloorTmp(fpRowElems, floorMax, floorMin);
    result.floorTmp = fixed + ubLayout.Align(std::max(floorMax, result.broadcastTmp)) <= ubLimit ? floorMax : floorMin;
    result.used = fixed + ubLayout.Align(std::max(result.floorTmp, result.broadcastTmp));
    if (result.used > ubLimit || result.used > std::numeric_limits<uint32_t>::max()) {
        return DenseTailResources{};
    }
    return result;
}

static uint64_t ScoreDenseTailCandidate(const ReuseDecomp& dec, uint32_t dTile, uint32_t availableCores, bool isFp32,
                                        bool doubleBuffered, uint64_t denseTailElements)
{
    const uint64_t usefulCores = std::max<uint64_t>(1U, CeilDiv(denseTailElements, FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS));
    const uint64_t active = std::min<uint64_t>(std::min<uint64_t>(availableCores, dec.d), usefulCores);
    const uint64_t criticalRows = CeilDiv(dec.d, active);
    const uint64_t criticalTiles = CeilDiv(criticalRows, dTile);
    const uint64_t vectorCommands = 5U + (isFp32 ? 0U : 3U);
    const uint64_t dmaIssueCost = doubleBuffered ? FLOOR_MOD_DMA_COMMAND_COST * 2U / 3U : FLOOR_MOD_DMA_COMMAND_COST;
    const uint64_t commandCost = dmaIssueCost + 48U + vectorCommands;
    return criticalTiles * commandCost + active + UnderfilledCorePenalty(dec.d * dec.e2, active) +
           CeilDiv(criticalRows * dec.e2, 256U) +
           (doubleBuffered && criticalTiles == 1U ? FLOOR_MOD_DMA_COMMAND_COST : 0U);
}

static bool IsDenseTailInputValid(const ReuseDecomp& dec, uint32_t dTile, uint32_t dtypeBytes, uint32_t availableCores)
{
    return dec.pos == 1U && dec.e1 == 1U && dec.e2 > 1U && dTile != 0U && dtypeBytes <= sizeof(float) &&
           dec.d <= std::numeric_limits<uint32_t>::max() && dec.e2 <= std::numeric_limits<uint32_t>::max() &&
           availableCores != 0U;
}

// Dense-tail batching is the canonical topology [..., D, 1] x [..., D, E2].
// Unlike tile-owned reuse, each core owns a contiguous interval of D rows.
// That turns all three GM transfers into compact spans and lets a dedicated
// kernel overlap the next two GM2UB commands with the current vector chain.
static StandardCandidate BuildDenseTailCandidate(const ReuseDecomp& dec, uint32_t dTile, uint32_t dtypeBytes,
                                                 bool isFp32, uint32_t availableCores, uint64_t ubLimit,
                                                 const platform_ascendc::PlatformAscendC& platform,
                                                 const UbLayout& ubLayout, bool doubleBuffered)
{
    StandardCandidate c;
    if (!IsDenseTailInputValid(dec, dTile, dtypeBytes, availableCores)) {
        return c;
    }
    uint64_t denseTailElements = 0U;
    if (!SafeMul(dec.d, dec.e2, denseTailElements)) {
        return c;
    }
    // BroadCast covers at most one 64-lane FP32 segment per row efficiently.
    // For a wider tail, a small workload spends more time in per-row Brcb
    // setup while row ownership also caps useful parallelism. The tile-owned
    // reuse path exposes independent D tiles and is measurably faster there.
    // Once the output fills a device wave, contiguous dense-tail DMA and
    // pipeline overlap amortize the extra vector segment again.
    const uint64_t deviceWaveElements = static_cast<uint64_t>(availableCores) * FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS;
    if (dec.e2 > FLOOR_MOD_GATHER_MASK_ELEMS && denseTailElements < deviceWaveElements) {
        return c;
    }
    const uint64_t pattern = static_cast<uint64_t>(dTile) * dec.e2;
    if (pattern == 0U || pattern > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(dTile) * dtypeBytes > std::numeric_limits<uint32_t>::max()) {
        return c;
    }

    c.denseTail = true;
    // Dense-tail uses the schedule bit as its buffering policy. Other reuse
    // controllers continue to use it for row ownership.
    c.rowOwned = doubleBuffered;
    c.dTile = dTile;
    c.e2Tile = static_cast<uint32_t>(dec.e2);
    c.dCount = static_cast<uint32_t>(CeilDiv(dec.d, dTile));
    c.e2Count = 1U;
    c.patternElems = static_cast<uint32_t>(pattern);
    c.tRowElems = ubLayout.AlignElements(pattern, dtypeBytes);
    c.fp32RowElems = ubLayout.AlignElements(pattern, sizeof(float));
    // Dense-tail expands [dTile, 1] to [dTile, e2].  The v220 implementation
    // needs a Brcb/GatherMask temporary buffer for an unaligned wide tail.
    // Account for it explicitly instead of relying on the implicit LCM stack.
    const auto resources = QueryDenseTailResources(dTile, static_cast<uint32_t>(dec.e2), dtypeBytes, isFp32,
                                                   doubleBuffered, ubLimit, platform, ubLayout, c.fp32RowElems);
    if (resources.used == 0U) {
        return c;
    }

    // Dense-tail tasks each load and expand an independent seed slab.  Small
    // outputs cannot amortize that setup on every vector core, even when D
    // exposes enough rows.  Bound the active cores by the same shape-derived
    // work quantum used for expanded paths; compact packed candidates remain
    // available when batching several D rows is cheaper than dense-tail setup.
    const uint64_t usefulCores = std::max<uint64_t>(1U, CeilDiv(denseTailElements, FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS));
    c.floorTmpBytes = resources.floorTmp;
    c.broadcastTmpBytes = resources.broadcastTmp;
    c.ubUsedBytes = static_cast<uint32_t>(resources.used);
    c.totalTasks = std::min<uint64_t>(std::min<uint64_t>(availableCores, dec.d), usefulCores);
    c.score = ScoreDenseTailCandidate(dec, dTile, availableCores, isFp32, doubleBuffered, denseTailElements);
    c.valid = true;
    return c;
}

struct StandardBuildOptions {
    bool expand;
    bool packed;
    bool materialized;
    bool rowOwned;
};

static bool IsStandardCandidateConfigValid(const ReuseDecomp& dec, const StandardBuildOptions& options, uint32_t dTile,
                                           uint32_t e2Tile)
{
    if (dTile == 0U || e2Tile == 0U || (options.expand && e2Tile < dec.e2 && dTile != 1U)) {
        return false;
    }
    if ((options.packed || options.materialized) && !options.expand) {
        return false;
    }
    return !options.packed || !options.materialized;
}

static bool ConfigureStandardSchedule(StandardCandidate& c, const ReuseDecomp& dec, const StandardBuildOptions& options,
                                      uint32_t availableCores, uint32_t requestedPartitions, uint64_t& maxRowsPerPart)
{
    const uint64_t baseTasks = options.rowOwned ? dec.pos : dec.pos * c.dCount * c.e2Count;
    const uint64_t maxParallelParts = baseTasks < availableCores ? std::max<uint64_t>(1U, availableCores / baseTasks) :
                                                                   1U;
    const uint64_t workPerBase = dec.e1 * (options.rowOwned ? dec.d : c.patternElems);
    const uint64_t targetCoreElems = options.expand ? FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS :
                                                      FLOOR_MOD_ROW_TARGET_CORE_ELEMS;
    const uint64_t maxLegalParts = std::min<uint64_t>(dec.e1, maxParallelParts);
    if (requestedPartitions > maxLegalParts) {
        return false;
    }
    const uint64_t usefulParts = std::max<uint64_t>(1U, CeilDiv(workPerBase, targetCoreElems));
    c.rowPartitions = requestedPartitions == 0U ?
                          static_cast<uint32_t>(std::min<uint64_t>(maxLegalParts, usefulParts)) :
                          requestedPartitions;
    c.totalTasks = baseTasks * c.rowPartitions;
    maxRowsPerPart = CeilDiv(dec.e1, c.rowPartitions);
    return true;
}

static bool ConfigureStandardRowLayout(StandardCandidate& c, const StandardBuildOptions& options, uint32_t dtypeBytes,
                                       const UbLayout& ubLayout)
{
    if (options.packed || !options.expand) {
        c.tRowElems = ubLayout.AlignElements(c.patternElems, dtypeBytes);
        c.fp32RowElems = c.tRowElems;
        return true;
    }
    const uint64_t commonStride = ubLayout.AlignElements(c.e2Tile, dtypeBytes);
    const uint64_t physical = static_cast<uint64_t>(c.dTile) * commonStride;
    if (physical > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    c.tRowElems = static_cast<uint32_t>(physical);
    c.fp32RowElems = static_cast<uint32_t>(physical);
    return true;
}

static bool ConfigureStandardCandidate(StandardCandidate& c, const ReuseDecomp& dec,
                                       const StandardBuildOptions& options, uint32_t dTile, uint32_t e2Tile,
                                       uint32_t dtypeBytes, uint32_t availableCores, const UbLayout& ubLayout,
                                       uint32_t packedBroadcastTmpBytes, uint32_t requestedPartitions,
                                       uint64_t& maxRowsPerPart)
{
    if (!IsStandardCandidateConfigValid(dec, options, dTile, e2Tile)) {
        return false;
    }
    const uint64_t pattern = static_cast<uint64_t>(dTile) * e2Tile;
    if (pattern == 0U || pattern > std::numeric_limits<uint32_t>::max() ||
        (options.expand && dTile > std::numeric_limits<uint16_t>::max())) {
        return false;
    }
    c.dTile = dTile;
    c.e2Tile = e2Tile;
    c.rowOwned = options.rowOwned;
    c.packed = options.packed;
    c.materialized = options.materialized;
    c.broadcastTmpBytes = options.packed ? packedBroadcastTmpBytes : 0U;
    c.patternElems = static_cast<uint32_t>(pattern);
    c.dCount = static_cast<uint32_t>(CeilDiv(dec.d, dTile));
    c.e2Count = static_cast<uint32_t>(CeilDiv(dec.e2, e2Tile));
    return ConfigureStandardSchedule(c, dec, options, availableCores, requestedPartitions, maxRowsPerPart) &&
           ConfigureStandardRowLayout(c, options, dtypeBytes, ubLayout);
}

static uint64_t StandardBatchFixedBytes(const StandardCandidate& c, const StandardBuildOptions& options, uint32_t batch,
                                        uint32_t dtypeBytes, bool isFp32, const UbLayout& ubLayout)
{
    const uint64_t rowTBytes = static_cast<uint64_t>(batch) * c.tRowElems * dtypeBytes;
    const uint64_t rowFpBytes = static_cast<uint64_t>(batch) * c.fp32RowElems * sizeof(float);
    const uint32_t seedTElems = ubLayout.AlignElements(c.dTile, dtypeBytes);
    const uint32_t seedFpElems = ubLayout.AlignElements(c.dTile, sizeof(float));
    uint64_t fixed = ubLayout.Align(static_cast<uint64_t>(seedTElems) * dtypeBytes);
    if (!isFp32) {
        fixed += ubLayout.Align(static_cast<uint64_t>(seedFpElems) * sizeof(float));
    }
    if (options.packed || options.materialized) {
        fixed += ubLayout.Align(static_cast<uint64_t>(c.fp32RowElems) * sizeof(float));
        fixed += FLOOR_MOD_HIGH_LEVEL_API_GUARD_BYTES;
    } else if (options.expand) {
        const uint64_t seedBlocks = CeilDiv(c.dTile, FLOOR_MOD_FP32_BLOCK_ELEMS) * FLOOR_MOD_FP32_BLOCK_ELEMS;
        fixed += ubLayout.Align(seedBlocks * ubLayout.blockBytes);
    }
    fixed += ubLayout.Align(2U * rowTBytes);
    fixed += isFp32 ? 0U : ubLayout.Align(rowFpBytes);
    return fixed + ubLayout.Align(rowFpBytes);
}

static bool SelectStandardBatch(StandardCandidate& c, const StandardBuildOptions& options, uint32_t dtypeBytes,
                                bool isFp32, uint64_t ubLimit, const UbLayout& ubLayout, uint64_t maxRowsPerPart)
{
    uint32_t low = 1U;
    uint32_t high = options.materialized ?
                        1U :
                        static_cast<uint32_t>(std::min<uint64_t>(FLOOR_MOD_MAX_REPEAT_ROWS, maxRowsPerPart));
    uint32_t bestBatch = 0U;
    uint32_t bestFloor = 0U;
    uint64_t bestUsed = 0U;
    while (low <= high) {
        const uint32_t batch = low + (high - low) / 2U;
        const uint64_t fixed = StandardBatchFixedBytes(c, options, batch, dtypeBytes, isFp32, ubLayout);
        if (fixed > ubLimit) {
            high = batch - 1U;
            continue;
        }
        uint32_t floorMax = 0U;
        uint32_t floorMin = 0U;
        QueryFloorTmp(static_cast<uint64_t>(batch) * c.fp32RowElems, floorMax, floorMin);
        const uint32_t floorChosen = fixed + ubLayout.Align(std::max(floorMax, c.broadcastTmpBytes)) <= ubLimit ?
                                         floorMax :
                                         floorMin;
        const uint64_t used = fixed + ubLayout.Align(std::max(floorChosen, c.broadcastTmpBytes));
        if (used <= ubLimit) {
            bestBatch = batch;
            bestFloor = floorChosen;
            bestUsed = used;
            low = batch + 1U;
        } else {
            high = batch - 1U;
        }
    }
    if (bestBatch == 0U) {
        return false;
    }
    const uint32_t repeatStrideBlocks = options.expand && !options.packed ?
                                            static_cast<uint32_t>(CeilDiv(c.e2Tile, FLOOR_MOD_FP32_BLOCK_ELEMS)) :
                                            c.fp32RowElems / FLOOR_MOD_FP32_BLOCK_ELEMS;
    if (!options.packed && !options.materialized && repeatStrideBlocks > FLOOR_MOD_MAX_REPEAT_ROWS &&
        (options.expand || bestBatch > 1U)) {
        return false;
    }
    c.batchRows = bestBatch;
    c.floorTmpBytes = bestFloor;
    c.ubUsedBytes = static_cast<uint32_t>(bestUsed);
    c.valid = true;
    return true;
}

static uint64_t StandardVectorCommands(const StandardCandidate& c, const StandardBuildOptions& options,
                                       uint64_t maxRowsPerPart)
{
    const uint64_t batches = CeilDiv(maxRowsPerPart, c.batchRows);
    const uint64_t innerTiles = options.rowOwned ? c.dCount : 1U;
    if (options.packed || options.materialized || (!options.expand && c.batchRows == 1U)) {
        return innerTiles * batches * 4U;
    }
    const uint64_t segments = options.expand ? CeilDiv(c.e2Tile, 64U) : CeilDiv(c.patternElems, 64U);
    if (options.expand) {
        const uint64_t repeatGroups = CeilDiv(c.dTile, FLOOR_MOD_MAX_REPEAT_ROWS);
        const uint64_t seedExpand = CeilDiv(c.dTile, 2040U);
        return innerTiles * (maxRowsPerPart * 3U * repeatGroups * segments + batches + seedExpand);
    }
    return innerTiles * batches * (3U * segments + 1U);
}

static bool ScoreStandardCandidate(StandardCandidate& c, const ReuseDecomp& dec, const StandardBuildOptions& options,
                                   uint32_t dtypeBytes, uint32_t availableCores, const UbLayout& ubLayout,
                                   uint64_t maxRowsPerPart)
{
    const uint64_t active = std::max<uint64_t>(1U, std::min<uint64_t>(availableCores, c.totalTasks));
    const uint64_t waves = CeilDiv(c.totalTasks, active);
    const uint64_t innerTiles = options.rowOwned ? c.dCount : 1U;
    const uint64_t batches = CeilDiv(maxRowsPerPart, c.batchRows);
    const uint64_t vectorCommands = StandardVectorCommands(c, options, maxRowsPerPart);
    constexpr uint64_t TILE_SETUP_COST = 48U;
    const uint64_t alignedE2Bytes = ubLayout.Align(static_cast<uint64_t>(c.e2Tile) * dtypeBytes);
    const uint64_t dmaPhysicalBytes = options.expand && !options.packed ?
                                          alignedE2Bytes * c.dTile :
                                          ubLayout.Align(static_cast<uint64_t>(c.patternElems) * dtypeBytes);
    const uint64_t paddingBytes = dmaPhysicalBytes - static_cast<uint64_t>(c.patternElems) * dtypeBytes;
    const uint64_t dmaCommands = innerTiles * (options.expand && !options.packed ? maxRowsPerPart : batches);
    const uint64_t issueCost = innerTiles * TILE_SETUP_COST + vectorCommands +
                               (FLOOR_MOD_DMA_COMMAND_COST / 2U) * dmaCommands;
    uint64_t outerElements = 0U;
    uint64_t innerElements = 0U;
    uint64_t logicalElements = 0U;
    if (!SafeMul(dec.pos, dec.e1, outerElements) || !SafeMul(dec.d, dec.e2, innerElements) ||
        !SafeMul(outerElements, innerElements, logicalElements)) {
        return false;
    }
    const uint64_t taskPenalty = c.totalTasks <= availableCores ? waves : c.totalTasks;
    const uint64_t underfillPenalty = options.expand ? UnderfilledCorePenalty(logicalElements, active) : 0U;
    c.score = waves * (96U + issueCost) + taskPenalty + underfillPenalty + CeilDiv(logicalElements, active * 256U) +
              CeilDiv(waves * paddingBytes, ubLayout.blockBytes);
    return true;
}

static StandardCandidate BuildStandardCandidate(const ReuseDecomp& dec, bool expand, bool packed, bool materialized,
                                                bool rowOwned, uint32_t dTile, uint32_t e2Tile, uint32_t dtypeBytes,
                                                bool isFp32, uint32_t availableCores, uint64_t ubLimit,
                                                const UbLayout& ubLayout, uint32_t packedBroadcastTmpBytes = 0U,
                                                uint32_t requestedPartitions = 0U)
{
    StandardCandidate c;
    const StandardBuildOptions options{expand, packed, materialized, rowOwned};
    uint64_t maxRowsPerPart = 0U;
    if (!ConfigureStandardCandidate(c, dec, options, dTile, e2Tile, dtypeBytes, availableCores, ubLayout,
                                    packedBroadcastTmpBytes, requestedPartitions, maxRowsPerPart)) {
        return c;
    }
    if (!SelectStandardBatch(c, options, dtypeBytes, isFp32, ubLimit, ubLayout, maxRowsPerPart)) {
        return c;
    }
    return ScoreStandardCandidate(c, dec, options, dtypeBytes, availableCores, ubLayout, maxRowsPerPart) ?
               c :
               StandardCandidate{};
}

static bool IsCompactPosInputValid(const ReuseDecomp& dec, uint32_t posTile, uint32_t availableCores)
{
    return dec.posContiguous && dec.pos > 1U && dec.e1 > 1U && dec.e2 == 1U && posTile > 1U && availableCores != 0U &&
           dec.d <= std::numeric_limits<uint32_t>::max() && dec.e1 <= std::numeric_limits<uint32_t>::max() &&
           posTile >= CeilDiv(dec.pos, std::min<uint64_t>(availableCores, dec.pos));
}

struct CompactPosResources {
    uint32_t rowElems = 0U;
    uint32_t floorTmp = 0U;
    uint32_t broadcastTmp = 0U;
    uint64_t used = 0U;
};

static CompactPosResources QueryCompactPosResources(const ReuseDecomp& dec, uint32_t posTile, uint32_t dtypeBytes,
                                                    bool isFp32, uint64_t ubLimit,
                                                    const platform_ascendc::PlatformAscendC& platform,
                                                    const UbLayout& ubLayout)
{
    CompactPosResources result;
    const uint32_t d = static_cast<uint32_t>(dec.d);
    const uint32_t e1 = static_cast<uint32_t>(dec.e1);
    result.rowElems = ubLayout.AlignElements(d, dtypeBytes);
    const uint64_t rows = static_cast<uint64_t>(posTile) * e1;
    const uint64_t paddedSeedElems = static_cast<uint64_t>(posTile) * result.rowElems;
    const uint64_t expandedSeedElems = rows * result.rowElems;
    const uint64_t logicalWorkElems = rows * d;
    const uint64_t rowSpanBlocks = static_cast<uint64_t>(result.rowElems) * sizeof(float) / ubLayout.blockBytes;
    if (rows > std::numeric_limits<uint16_t>::max() || rowSpanBlocks > std::numeric_limits<uint16_t>::max() ||
        paddedSeedElems > std::numeric_limits<uint32_t>::max() ||
        expandedSeedElems > std::numeric_limits<uint32_t>::max() ||
        logicalWorkElems > std::numeric_limits<uint32_t>::max() ||
        paddedSeedElems * dtypeBytes > std::numeric_limits<uint32_t>::max() ||
        expandedSeedElems * sizeof(float) > std::numeric_limits<uint32_t>::max() ||
        logicalWorkElems * std::max<uint32_t>(dtypeBytes, sizeof(float)) > std::numeric_limits<uint32_t>::max()) {
        return CompactPosResources{};
    }
    const uint64_t seedTBytes = ubLayout.Align(paddedSeedElems * dtypeBytes);
    const uint64_t seedFpBytes = ubLayout.Align(paddedSeedElems * sizeof(float));
    const uint64_t expandedSeedFpBytes = ubLayout.Align(expandedSeedElems * sizeof(float));
    const uint64_t workTBytes = ubLayout.Align(logicalWorkElems * dtypeBytes);
    const uint64_t workFpBytes = ubLayout.Align(logicalWorkElems * sizeof(float));
    uint64_t fixed = seedTBytes + expandedSeedFpBytes + workTBytes + workFpBytes + FLOOR_MOD_HIGH_LEVEL_API_GUARD_BYTES;
    if (!isFp32) {
        fixed += seedFpBytes + workFpBytes;
    }
    if (fixed > ubLimit) {
        return CompactPosResources{};
    }
    result.broadcastTmp = QueryTrailingBroadcastTmp(platform, e1, result.rowElems);
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    QueryFloorTmp(logicalWorkElems, floorMax, floorMin);
    result.floorTmp = fixed + ubLayout.Align(std::max(floorMax, result.broadcastTmp)) <= ubLimit ? floorMax : floorMin;
    result.used = fixed + ubLayout.Align(std::max(result.floorTmp, result.broadcastTmp));
    if (result.used > ubLimit || result.used > std::numeric_limits<uint32_t>::max()) {
        return CompactPosResources{};
    }
    return result;
}

static uint64_t ScoreCompactPosCandidate(const ReuseDecomp& dec, uint32_t dtypeBytes, bool isFp32,
                                         uint32_t availableCores, const UbLayout& ubLayout, uint32_t rowElems)
{
    const uint64_t activeCores = std::min<uint64_t>(availableCores, dec.pos);
    const uint64_t maxPositionsPerCore = CeilDiv(dec.pos, activeCores);
    const uint64_t vectorCommands = maxPositionsPerCore + 5U + (isFp32 ? 0U : 3U);
    const uint64_t seedPhysicalBytes = maxPositionsPerCore * ubLayout.Align(dec.d * dtypeBytes);
    const uint64_t paddingBytes = seedPhysicalBytes - maxPositionsPerCore * dec.d * dtypeBytes;
    const uint64_t logicalPerCore = CeilDiv(dec.pos * dec.e1 * dec.d, activeCores);
    const uint64_t vectorElements = maxPositionsPerCore * dec.e1 * rowElems +
                                    logicalPerCore * (5U + (isFp32 ? 0U : 2U)) +
                                    (isFp32 ? 0U : maxPositionsPerCore * rowElems);
    return 96U + vectorCommands + 3U * (FLOOR_MOD_DMA_COMMAND_COST / 2U) + activeCores + CeilDiv(vectorElements, 256U) +
           CeilDiv(paddingBytes, ubLayout.blockBytes);
}

static StandardCandidate BuildCompactPosCandidate(const ReuseDecomp& dec, uint32_t posTile, uint32_t dtypeBytes,
                                                  bool isFp32, uint32_t availableCores, uint64_t ubLimit,
                                                  const platform_ascendc::PlatformAscendC& platform,
                                                  const UbLayout& ubLayout)
{
    StandardCandidate c;
    if (!IsCompactPosInputValid(dec, posTile, availableCores)) {
        return c;
    }
    const uint32_t d = static_cast<uint32_t>(dec.d);
    const auto resources = QueryCompactPosResources(dec, posTile, dtypeBytes, isFp32, ubLimit, platform, ubLayout);
    if (resources.used == 0U) {
        return c;
    }

    c.valid = true;
    c.compactPos = true;
    c.posTile = posTile;
    c.dTile = d;
    c.dCount = 1U;
    c.e2Tile = 1U;
    c.e2Count = 1U;
    c.rowPartitions = 1U;
    c.batchRows = 1U;
    c.patternElems = d;
    c.tRowElems = resources.rowElems;
    c.fp32RowElems = resources.rowElems;
    c.floorTmpBytes = resources.floorTmp;
    c.broadcastTmpBytes = resources.broadcastTmp;
    c.ubUsedBytes = static_cast<uint32_t>(resources.used);
    c.totalTasks = std::min<uint64_t>(availableCores, dec.pos);

    c.score = ScoreCompactPosCandidate(dec, dtypeBytes, isFp32, availableCores, ubLayout, c.tRowElems);
    return c;
}

static bool SameStandardPlan(const StandardCandidate& lhs, const StandardCandidate& rhs)
{
    return lhs.rowOwned == rhs.rowOwned && lhs.packed == rhs.packed && lhs.materialized == rhs.materialized &&
           lhs.compactPos == rhs.compactPos && lhs.denseTail == rhs.denseTail && lhs.compactRows == rhs.compactRows &&
           lhs.compactRepeat == rhs.compactRepeat && lhs.posTile == rhs.posTile && lhs.dTile == rhs.dTile &&
           lhs.e2Tile == rhs.e2Tile && lhs.rowPartitions == rhs.rowPartitions && lhs.batchRows == rhs.batchRows;
}

static void AddStandardCandidate(std::vector<StandardCandidate>& candidates, const StandardCandidate& candidate)
{
    if (!candidate.valid) {
        return;
    }
    const auto duplicate = std::find_if(candidates.begin(), candidates.end(), [&](const StandardCandidate& item) {
        return SameStandardPlan(item, candidate);
    });
    if (duplicate == candidates.end()) {
        candidates.push_back(candidate);
    } else if (candidate.score < duplicate->score) {
        *duplicate = candidate;
    }
}

struct StandardSearchContext {
    const ReuseDecomp& dec;
    uint32_t dtypeBytes;
    bool isFp32;
    uint32_t availableCores;
    uint64_t ubLimit;
    const platform_ascendc::PlatformAscendC& platform;
    const UbLayout& ubLayout;
    std::vector<StandardCandidate>& candidates;
};

static void CollectCompactRowCandidates(const StandardSearchContext& ctx)
{
    const auto addCompact = [&](uint32_t parts) {
        if (parts == 0U || parts > std::min<uint64_t>(ctx.dec.e1, ctx.availableCores)) {
            return;
        }
        AddStandardCandidate(ctx.candidates,
                             BuildCompactRowCandidate(ctx.dec, parts, ctx.dtypeBytes, ctx.isFp32, ctx.availableCores,
                                                      ctx.ubLimit, ctx.platform, ctx.ubLayout, false));
        AddStandardCandidate(ctx.candidates,
                             BuildCompactRowCandidate(ctx.dec, parts, ctx.dtypeBytes, ctx.isFp32, ctx.availableCores,
                                                      ctx.ubLimit, ctx.platform, ctx.ubLayout, true));
    };
    if (ctx.dec.pos == 1U && ctx.dec.e2 == 1U && ctx.dtypeBytes <= sizeof(float)) {
        const uint32_t maxParts = static_cast<uint32_t>(std::min<uint64_t>(ctx.dec.e1, ctx.availableCores));
        const uint32_t useful = static_cast<uint32_t>(std::min<uint64_t>(
            maxParts, std::max<uint64_t>(1U, CeilDiv(ctx.dec.e1 * ctx.dec.d, FLOOR_MOD_ROW_TARGET_CORE_ELEMS))));
        addCompact(useful);
        addCompact(useful > 1U ? useful - 1U : 1U);
        addCompact(useful + 1U);
        addCompact(maxParts);
        addCompact(maxParts > 1U ? maxParts - 1U : 1U);
        addCompact(static_cast<uint32_t>(CeilDiv(maxParts, 2U)));
        addCompact(static_cast<uint32_t>(CeilDiv(maxParts, 4U)));
    }
}

static void AddNoExpandDTileCandidates(const StandardSearchContext& ctx, uint32_t dTile)
{
    for (bool rowOwned : {false, true}) {
        const uint64_t baseTasks = rowOwned ? ctx.dec.pos : ctx.dec.pos * CeilDiv(ctx.dec.d, dTile);
        const uint32_t maxParts = static_cast<uint32_t>(
            std::min<uint64_t>(ctx.dec.e1, baseTasks < ctx.availableCores ? ctx.availableCores / baseTasks : 1U));
        for (uint32_t rowParts : {0U, maxParts, static_cast<uint32_t>(CeilDiv(maxParts, 2U))}) {
            AddStandardCandidate(
                ctx.candidates,
                BuildStandardCandidate(ctx.dec, false, false, false, rowOwned, dTile, 1U, ctx.dtypeBytes, ctx.isFp32,
                                       ctx.availableCores, ctx.ubLimit, ctx.ubLayout, 0U, rowParts));
        }
    }
}

static void CollectNoExpandDTileCandidates(const StandardSearchContext& ctx)
{
    const uint32_t minTiles = static_cast<uint32_t>(CeilDiv(ctx.dec.d, FLOOR_MOD_MAX_PATTERN_ELEMS));
    const uint32_t maxBalancedTiles = static_cast<uint32_t>(std::min<uint64_t>(ctx.dec.d, minTiles + 8U));
    for (uint32_t tiles = minTiles; tiles <= maxBalancedTiles; ++tiles) {
        const uint32_t dTile = ctx.ubLayout.AlignElements(CeilDiv(ctx.dec.d, tiles), ctx.dtypeBytes);
        if (dTile <= ctx.dec.d && dTile <= FLOOR_MOD_MAX_PATTERN_ELEMS) {
            AddNoExpandDTileCandidates(ctx, dTile);
        }
    }
    const uint32_t maxTile = static_cast<uint32_t>(std::min<uint64_t>(ctx.dec.d, FLOOR_MOD_MAX_PATTERN_ELEMS));
    for (uint32_t tile = maxTile; tile > 0U; tile /= 2U) {
        AddNoExpandDTileCandidates(ctx, tile);
    }
    uint32_t wideTile = static_cast<uint32_t>(std::min<uint64_t>(ctx.dec.d, std::numeric_limits<uint32_t>::max()));
    while (wideTile > FLOOR_MOD_MAX_PATTERN_ELEMS) {
        AddNoExpandDTileCandidates(ctx, wideTile);
        wideTile /= 2U;
    }
}

static void AddCompactPosCandidate(const StandardSearchContext& ctx, uint32_t posTile)
{
    AddStandardCandidate(ctx.candidates,
                         BuildCompactPosCandidate(ctx.dec, posTile, ctx.dtypeBytes, ctx.isFp32, ctx.availableCores,
                                                  ctx.ubLimit, ctx.platform, ctx.ubLayout));
}

static uint32_t FindMaxLegalCompactPos(const StandardSearchContext& ctx, uint32_t low, uint32_t high)
{
    uint32_t result = 0U;
    while (low <= high) {
        const uint32_t mid = low + (high - low) / 2U;
        const auto candidate = BuildCompactPosCandidate(ctx.dec, mid, ctx.dtypeBytes, ctx.isFp32, ctx.availableCores,
                                                        ctx.ubLimit, ctx.platform, ctx.ubLayout);
        if (candidate.valid) {
            result = mid;
            low = mid + 1U;
        } else {
            high = mid - 1U;
        }
    }
    return result;
}

static void CollectCompactPosCandidates(const StandardSearchContext& ctx)
{
    if (!ctx.dec.posContiguous || ctx.dec.pos <= 1U || ctx.dec.e1 <= 1U ||
        ctx.dec.d > std::numeric_limits<uint32_t>::max() || ctx.dec.e1 > std::numeric_limits<uint32_t>::max()) {
        return;
    }
    const uint32_t minCoreOwnedPos = static_cast<uint32_t>(
        CeilDiv(ctx.dec.pos, std::min<uint64_t>(ctx.availableCores, ctx.dec.pos)));
    const uint32_t maxPos = static_cast<uint32_t>(
        std::min<uint64_t>(ctx.dec.pos, std::numeric_limits<uint32_t>::max()));
    const uint32_t maxLegalPos = FindMaxLegalCompactPos(ctx, std::max(2U, minCoreOwnedPos), maxPos);
    if (maxLegalPos <= 1U) {
        return;
    }
    AddCompactPosCandidate(ctx, maxLegalPos);
    for (uint64_t waves = 1U; waves <= 3U; ++waves) {
        const uint32_t balanced = static_cast<uint32_t>(
            std::min<uint64_t>(maxLegalPos, std::max<uint64_t>(2U, ctx.dec.pos / (waves * ctx.availableCores))));
        AddCompactPosCandidate(ctx, balanced);
        if (balanced < maxLegalPos) {
            AddCompactPosCandidate(ctx, balanced + 1U);
        }
    }
    for (uint32_t tile = maxLegalPos / 2U; tile > 1U; tile /= 2U) {
        AddCompactPosCandidate(ctx, tile);
    }
}

static void CollectNoExpandCandidates(const StandardSearchContext& ctx)
{
    CollectCompactRowCandidates(ctx);
    CollectNoExpandDTileCandidates(ctx);
    CollectCompactPosCandidates(ctx);
}

static void CollectDenseTailCandidates(const StandardSearchContext& ctx)
{
    if (ctx.dec.pos != 1U || ctx.dec.e1 != 1U || ctx.dec.e2 > std::numeric_limits<uint32_t>::max() ||
        ctx.dec.d > std::numeric_limits<uint32_t>::max()) {
        return;
    }
    for (bool buffered : {false, true}) {
        const auto build = [&](uint32_t dTile) {
            return BuildDenseTailCandidate(ctx.dec, dTile, ctx.dtypeBytes, ctx.isFp32, ctx.availableCores, ctx.ubLimit,
                                           ctx.platform, ctx.ubLayout, buffered);
        };
        uint32_t low = 1U;
        uint32_t high = static_cast<uint32_t>(
            std::min<uint64_t>(ctx.dec.d, std::numeric_limits<uint32_t>::max() / ctx.dec.e2));
        uint32_t maxLegalD = 0U;
        while (low <= high) {
            const uint32_t mid = low + (high - low) / 2U;
            if (build(mid).valid) {
                maxLegalD = mid;
                low = mid + 1U;
            } else {
                high = mid - 1U;
            }
        }
        if (maxLegalD > 0U) {
            AddStandardCandidate(ctx.candidates, build(maxLegalD));
            for (uint32_t tile = maxLegalD / 2U; tile > 0U; tile /= 2U) {
                AddStandardCandidate(ctx.candidates, build(tile));
            }
        }
    }
}

static void CollectRepeatedExpandCandidates(const StandardSearchContext& ctx)
{
    if (ctx.dec.e2 > FLOOR_MOD_MAX_PATTERN_ELEMS) {
        return;
    }
    const uint32_t maxD = static_cast<uint32_t>(std::min<uint64_t>(
        ctx.dec.d,
        std::min<uint64_t>(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint32_t>::max() / ctx.dec.e2)));
    for (uint32_t tile = maxD; tile > 0U; tile /= 2U) {
        AddStandardCandidate(
            ctx.candidates,
            BuildStandardCandidate(ctx.dec, true, false, false, false, tile, static_cast<uint32_t>(ctx.dec.e2),
                                   ctx.dtypeBytes, ctx.isFp32, ctx.availableCores, ctx.ubLimit, ctx.ubLayout));
    }
}

static void CollectPaddedE2Candidates(const StandardSearchContext& ctx)
{
    const uint32_t maxE2 = static_cast<uint32_t>(std::min<uint64_t>(ctx.dec.e2, FLOOR_MOD_MAX_PATTERN_ELEMS));
    for (uint32_t e2Tile = maxE2; e2Tile > 0U; e2Tile /= 2U) {
        AddStandardCandidate(ctx.candidates,
                             BuildStandardCandidate(ctx.dec, true, false, false, false, 1U, e2Tile, ctx.dtypeBytes,
                                                    ctx.isFp32, ctx.availableCores, ctx.ubLimit, ctx.ubLayout));
    }
}

static void CollectMaterializedCandidates(const StandardSearchContext& ctx)
{
    if (ctx.dec.d <= FLOOR_MOD_MAX_REPEAT_ROWS || ctx.dec.e2 > FLOOR_MOD_MAX_PATTERN_ELEMS ||
        ctx.dec.e2 > std::numeric_limits<uint32_t>::max()) {
        return;
    }
    const auto build = [&](uint32_t dTile) {
        return BuildStandardCandidate(ctx.dec, true, false, true, false, dTile, static_cast<uint32_t>(ctx.dec.e2),
                                      ctx.dtypeBytes, ctx.isFp32, ctx.availableCores, ctx.ubLimit, ctx.ubLayout);
    };
    const uint32_t dLimit = static_cast<uint32_t>(std::min<uint64_t>(
        ctx.dec.d,
        std::min<uint64_t>(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint32_t>::max() / ctx.dec.e2)));
    uint32_t low = FLOOR_MOD_MAX_REPEAT_ROWS + 1U;
    uint32_t high = dLimit;
    uint32_t maxLegalD = 0U;
    while (low <= high) {
        const uint32_t mid = low + (high - low) / 2U;
        if (build(mid).valid) {
            maxLegalD = mid;
            low = mid + 1U;
        } else {
            high = mid - 1U;
        }
    }
    if (maxLegalD <= FLOOR_MOD_MAX_REPEAT_ROWS) {
        return;
    }
    AddStandardCandidate(ctx.candidates, build(maxLegalD));
    const uint64_t minWaves = CeilDiv(CeilDiv(ctx.dec.d, maxLegalD), ctx.availableCores);
    for (uint64_t waves = std::max<uint64_t>(1U, minWaves); waves <= minWaves + 2U; ++waves) {
        const uint32_t balanced = static_cast<uint32_t>(
            std::min<uint64_t>(maxLegalD, CeilDiv(ctx.dec.d, waves * ctx.availableCores)));
        if (balanced > FLOOR_MOD_MAX_REPEAT_ROWS) {
            AddStandardCandidate(ctx.candidates, build(balanced));
        }
    }
    for (uint32_t tile = maxLegalD / 2U; tile > FLOOR_MOD_MAX_REPEAT_ROWS; tile /= 2U) {
        AddStandardCandidate(ctx.candidates, build(tile));
    }
}

static StandardCandidate BuildPackedSearchCandidate(const StandardSearchContext& ctx, uint32_t dTile, uint32_t e2Tile)
{
    const uint32_t tmp = QueryBroadcastTmp(ctx.platform, dTile, e2Tile, ctx.dtypeBytes);
    return BuildStandardCandidate(ctx.dec, true, true, false, false, dTile, e2Tile, ctx.dtypeBytes, ctx.isFp32,
                                  ctx.availableCores, ctx.ubLimit, ctx.ubLayout, tmp);
}

static void AddWaveBalancedPackedCandidates(const StandardSearchContext& ctx, uint64_t extent, uint32_t maxTile,
                                            bool splitD)
{
    if (maxTile == 0U) {
        return;
    }
    const auto add = [&](uint32_t tile) {
        const uint32_t dTile = splitD ? tile : 1U;
        const uint32_t e2Tile = splitD ? static_cast<uint32_t>(ctx.dec.e2) : tile;
        AddStandardCandidate(ctx.candidates, BuildPackedSearchCandidate(ctx, dTile, e2Tile));
    };
    add(maxTile);
    const uint64_t minWaves = CeilDiv(CeilDiv(extent, maxTile), ctx.availableCores);
    for (uint64_t waves = std::max<uint64_t>(1U, minWaves); waves <= minWaves + 2U; ++waves) {
        add(static_cast<uint32_t>(std::min<uint64_t>(maxTile, CeilDiv(extent, waves * ctx.availableCores))));
    }
    for (uint32_t tile = maxTile / 2U; tile > 0U; tile /= 2U) {
        add(tile);
    }
}

static uint32_t FindMaxLegalPackedTile(const StandardSearchContext& ctx, uint32_t high, bool splitD)
{
    uint32_t low = 1U;
    uint32_t result = 0U;
    while (low <= high) {
        const uint32_t mid = low + (high - low) / 2U;
        const uint32_t dTile = splitD ? mid : 1U;
        const uint32_t e2Tile = splitD ? static_cast<uint32_t>(ctx.dec.e2) : mid;
        if (BuildPackedSearchCandidate(ctx, dTile, e2Tile).valid) {
            result = mid;
            low = mid + 1U;
        } else {
            high = mid - 1U;
        }
    }
    return result;
}

static void CollectPackedCandidates(const StandardSearchContext& ctx)
{
    if (ctx.dtypeBytes > sizeof(float) || ctx.dec.e2 > std::numeric_limits<uint32_t>::max()) {
        return;
    }
    const uint32_t dLimit = static_cast<uint32_t>(std::min<uint64_t>(
        ctx.dec.d,
        std::min<uint64_t>(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint32_t>::max() / ctx.dec.e2)));
    const uint32_t maxLegalD = FindMaxLegalPackedTile(ctx, dLimit, true);
    AddWaveBalancedPackedCandidates(ctx, ctx.dec.d, maxLegalD, true);
    const uint32_t maxLegalE2 = FindMaxLegalPackedTile(ctx, static_cast<uint32_t>(ctx.dec.e2), false);
    AddWaveBalancedPackedCandidates(ctx, ctx.dec.e2, maxLegalE2, false);
}

static void CollectExpandCandidates(const StandardSearchContext& ctx)
{
    CollectDenseTailCandidates(ctx);
    CollectRepeatedExpandCandidates(ctx);
    CollectMaterializedCandidates(ctx);
    CollectPaddedE2Candidates(ctx);
    CollectPackedCandidates(ctx);
}

static std::vector<StandardCandidate> FindStandardCandidates(const ReuseDecomp& dec, uint32_t dtypeBytes, bool isFp32,
                                                             uint32_t availableCores, uint64_t ubLimit,
                                                             const platform_ascendc::PlatformAscendC& platform,
                                                             const UbLayout& ubLayout)
{
    std::vector<StandardCandidate> candidates;
    StandardSearchContext ctx{dec, dtypeBytes, isFp32, availableCores, ubLimit, platform, ubLayout, candidates};
    if (dec.e2 <= 1U) {
        CollectNoExpandCandidates(ctx);
    } else {
        CollectExpandCandidates(ctx);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.score < rhs.score; });
    return candidates;
}

struct CrossCandidate {
    bool valid = false;
    bool packed = false;
    uint32_t outerTile = 1U;
    uint32_t aTile = 1U;
    uint32_t bTile = 1U;
    uint32_t unitBTAligned = 1U;
    uint32_t unitBAligned = 1U;
    uint32_t outerCount = 1U;
    uint32_t aCnt = 1U;
    uint32_t bCount = 1U;
    uint32_t dAligned = FLOOR_MOD_FP32_BLOCK_ELEMS;
    uint32_t floorTmpBytes = 0U;
    uint32_t broadcastTmpBytes = 0U;
    uint32_t ubUsedBytes = 0U;
    uint64_t totalTasks = 1U;
    uint64_t score = std::numeric_limits<uint64_t>::max();
};

static bool PreferCrossRoute(const CrossDecomp& cross, const CrossCandidate& crossBest,
                             const StandardCandidate& reuseBest, uint32_t availableCores)
{
    if (!crossBest.valid) {
        return false;
    }
    if (!reuseBest.valid) {
        return true;
    }

    // Crossed ownership removes the streamed operand's A-axis reload. It pays
    // off once the plan fills a hardware wave, provided there is no common M
    // span turning the output into many short DMA rows. Candidate validation
    // has already rejected dtype/D combinations that the crossed kernel cannot
    // represent. This boundary uses only topology and hardware capacity so new
    // counterfactual measurements can recalibrate it.
    const bool amortizedCross = cross.m == 1U && crossBest.totalTasks >= availableCores;
    return amortizedCross || crossBest.score < reuseBest.score;
}

static std::vector<uint32_t> MakeTileCandidates(uint64_t extent, uint32_t alignment, uint32_t targetParts = 1U)
{
    std::vector<uint32_t> values;
    const auto addCandidate = [&](uint64_t raw) {
        if (raw == 0U || raw > extent || raw > std::numeric_limits<uint32_t>::max()) {
            return;
        }
        uint32_t value = static_cast<uint32_t>(raw);
        if (alignment > 1U && value > 1U) {
            value = value / alignment * alignment;
        }
        if (value > 0U) {
            values.push_back(value);
        }
    };

    // Cover both command-efficient small tiles and UB-filling large tiles.
    // The old fixed list stopped at 128, so a large full tile that missed UB
    // fell straight back to a tiny batch even when an intermediate tile fit.
    constexpr uint64_t MAX_POWER_CANDIDATE = 2048U;
    for (uint64_t power = 1U; power <= std::min<uint64_t>(extent, MAX_POWER_CANDIDATE); power *= 2U) {
        addCandidate(power);
    }
    for (uint32_t divisor = 2U; divisor <= 128U; divisor *= 2U) {
        addCandidate(CeilDiv(extent, divisor));
    }
    if (targetParts > 1U) {
        const uint64_t balanced = CeilDiv(extent, targetParts);
        for (int32_t delta = -1; delta <= 1; ++delta) {
            const int64_t adjusted = static_cast<int64_t>(balanced) + delta;
            addCandidate(adjusted > 0 ? static_cast<uint64_t>(adjusted) : 0U);
        }
    }
    addCandidate(extent);
    // Alignment is useful for interior tiles, but a full-extent tile has no
    // inter-tile boundary and must not be rounded down.  Keeping the exact
    // extent is especially important for crossed B tiles: it turns many
    // strided partial stores into one batched contiguous UB2GM command.
    if (extent <= std::numeric_limits<uint32_t>::max()) {
        values.push_back(static_cast<uint32_t>(extent));
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

struct CrossBufferSizes {
    uint64_t seedElems = 0U;
    uint64_t denseTElems = 0U;
    uint64_t denseFpElems = 0U;
    uint64_t workingElems = 0U;
    uint64_t outputFpElems = 0U;
    uint64_t outputBufferElems = 0U;
};

static bool HasValidCrossExtents(const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile, uint32_t bTile,
                                 uint32_t availableCores)
{
    return outerTile != 0U && aTile != 0U && bTile != 0U && availableCores != 0U && dec.outer != 0U && dec.a != 0U &&
           dec.b != 0U && dec.m != 0U && dec.d != 0U;
}

static bool SupportsPackedCrossTile(const CrossDecomp& dec, uint32_t bTile, bool packed)
{
    if (!packed) {
        return true;
    }
    return dec.d == 1U ? bTile <= FLOOR_MOD_GATHER_MASK_ELEMS : dec.d <= FLOOR_MOD_GATHER_MASK_ELEMS;
}

static bool IsCrossCandidateShapeValid(const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile, uint32_t bTile,
                                       uint32_t dtypeBytes, uint32_t availableCores, const UbLayout& ubLayout,
                                       bool packed)
{
    if (!IsValidLayout(ubLayout, dtypeBytes) || !HasValidCrossExtents(dec, outerTile, aTile, bTile, availableCores) ||
        !SupportsPackedCrossTile(dec, bTile, packed)) {
        return false;
    }
    const bool unitD = dec.d == 1U;
    return (dtypeBytes <= sizeof(float) || unitD) && (unitD || dec.m == 1U || bTile == dec.b);
}

static bool IsCrossPackedAlignmentValid(const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile, uint32_t bTile,
                                        uint32_t dtypeBytes, const UbLayout& ubLayout)
{
    const uint64_t remainderA = SafeRemainder(dec.a, aTile);
    const uint64_t remainderB = SafeRemainder(dec.b, bTile);
    const uint64_t tailA = remainderA == 0U ? aTile : remainderA;
    const uint64_t tailB = remainderB == 0U ? bTile : remainderB;
    if (bTile != dec.b && ((static_cast<uint64_t>(bTile) * dtypeBytes) % ubLayout.blockBytes != 0U ||
                           (tailB * dtypeBytes) % ubLayout.blockBytes != 0U)) {
        return false;
    }
    if (outerTile <= 1U || aTile == dec.a) {
        return true;
    }
    for (uint64_t aSpan : {static_cast<uint64_t>(aTile), tailA}) {
        if ((aSpan * dec.m * dtypeBytes) % ubLayout.blockBytes != 0U) {
            return false;
        }
        for (uint64_t bSpan : {static_cast<uint64_t>(bTile), tailB}) {
            if ((aSpan * dec.m * bSpan * dtypeBytes) % ubLayout.blockBytes != 0U) {
                return false;
            }
        }
    }
    return true;
}

static bool ConfigureCrossLayout(CrossCandidate& c, const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile,
                                 uint32_t bTile, uint32_t dtypeBytes, const UbLayout& ubLayout, bool packed,
                                 uint32_t compactBroadcastTmpBytes)
{
    const uint32_t elementsPerBlock = ElementsPerBlock(ubLayout, dtypeBytes);
    if (elementsPerBlock == 0U) {
        return false;
    }
    const uint32_t commonAlignment = std::max<uint32_t>(elementsPerBlock, FLOOR_MOD_FP32_BLOCK_ELEMS);
    c.packed = packed;
    c.broadcastTmpBytes = packed ? compactBroadcastTmpBytes : 0U;
    c.dAligned = static_cast<uint32_t>(CeilDiv(dec.d, commonAlignment) * commonAlignment);
    c.unitBTAligned = packed ? ubLayout.AlignElements(static_cast<uint64_t>(bTile) * dec.d, dtypeBytes) :
                               static_cast<uint32_t>(CeilDiv(bTile, elementsPerBlock) * elementsPerBlock);
    c.unitBAligned = packed ? ubLayout.AlignElements(static_cast<uint64_t>(bTile) * dec.d, sizeof(float)) :
                              static_cast<uint32_t>(CeilDiv(bTile, commonAlignment) * commonAlignment);
    return !packed || dec.d != 1U || IsCrossPackedAlignmentValid(dec, outerTile, aTile, bTile, dtypeBytes, ubLayout);
}

static bool CrossBufferSizesFit(const CrossCandidate& c, const CrossBufferSizes& sizes, const CrossDecomp& dec,
                                uint32_t dtypeBytes, const UbLayout& ubLayout)
{
    constexpr uint64_t MAX_OFFSET = std::numeric_limits<uint32_t>::max();
    if (sizes.workingElems == 0U || sizes.seedElems > MAX_OFFSET || sizes.denseTElems > MAX_OFFSET ||
        sizes.denseFpElems > MAX_OFFSET) {
        return false;
    }
    if (sizes.workingElems > MAX_OFFSET || sizes.outputFpElems > MAX_OFFSET || sizes.outputBufferElems > MAX_OFFSET) {
        return false;
    }
    const bool unitD = dec.d == 1U;
    if (!unitD) {
        return c.dAligned / FLOOR_MOD_FP32_BLOCK_ELEMS <= FLOOR_MOD_MAX_REPEAT_ROWS;
    }
    return c.unitBAligned / FLOOR_MOD_FP32_BLOCK_ELEMS <= FLOOR_MOD_MAX_REPEAT_ROWS &&
           c.unitBTAligned * dtypeBytes / ubLayout.blockBytes <= FLOOR_MOD_MAX_REPEAT_ROWS;
}

static bool ComputeCrossBufferSizes(CrossBufferSizes& sizes, const CrossCandidate& c, const CrossDecomp& dec,
                                    uint32_t outerTile, uint32_t aTile, uint32_t bTile, uint32_t dtypeBytes,
                                    const UbLayout& ubLayout)
{
    const bool unitD = dec.d == 1U;
    const uint64_t logicalSeed = unitD ? static_cast<uint64_t>(outerTile) * aTile * dec.m :
                                         static_cast<uint64_t>(outerTile) * aTile * dec.m * c.dAligned;
    sizes.seedElems = unitD ? CeilDiv(logicalSeed, FLOOR_MOD_FP32_BLOCK_ELEMS) * FLOOR_MOD_FP32_BLOCK_ELEMS :
                              logicalSeed;
    sizes.denseTElems = (unitD || c.packed) ? static_cast<uint64_t>(outerTile) * dec.m * c.unitBTAligned :
                                              static_cast<uint64_t>(outerTile) * dec.m * bTile * c.dAligned;
    sizes.denseFpElems = unitD ? static_cast<uint64_t>(outerTile) * dec.m * c.unitBAligned : sizes.denseTElems;
    sizes.workingElems = (unitD || c.packed) ? static_cast<uint64_t>(dec.m) * c.unitBAligned :
                                               static_cast<uint64_t>(dec.m) * bTile * c.dAligned;
    sizes.outputFpElems = static_cast<uint64_t>(outerTile) * aTile * sizes.workingElems;
    const uint64_t paddedBroadcast = c.packed ? static_cast<uint64_t>(outerTile) * aTile * dec.m * bTile * c.dAligned :
                                                0U;
    sizes.outputBufferElems = c.packed ? std::max(sizes.outputFpElems, paddedBroadcast) : sizes.outputFpElems;
    const uint64_t compactOutput = static_cast<uint64_t>(outerTile) * aTile * dec.m * bTile * dec.d;
    return compactOutput <= std::numeric_limits<uint32_t>::max() &&
           CrossBufferSizesFit(c, sizes, dec, dtypeBytes, ubLayout);
}

static uint64_t CrossOutputFixedBytes(const CrossCandidate& c, const CrossDecomp& dec, const CrossBufferSizes& sizes,
                                      uint32_t outerTile, uint32_t aTile, uint32_t dtypeBytes, bool isFp32,
                                      const UbLayout& ubLayout)
{
    const bool unitD = dec.d == 1U;
    uint64_t fixed = 0U;
    if (unitD || c.packed) {
        const uint64_t outputTElems = static_cast<uint64_t>(outerTile) * aTile * dec.m * c.unitBTAligned;
        fixed += isFp32 ? 0U : 2U * ubLayout.Align(outputTElems * dtypeBytes) + FLOOR_MOD_UB_BANK_GUARD_BYTES;
    } else {
        fixed += isFp32 ? 0U : 2U * ubLayout.Align(sizes.outputFpElems * dtypeBytes) + FLOOR_MOD_UB_BANK_GUARD_BYTES;
    }
    if (unitD) {
        fixed += ubLayout.Align(sizes.seedElems * ubLayout.blockBytes);
        fixed += 3U * ubLayout.Align(sizes.outputFpElems * sizeof(float));
        return fixed + 3U * FLOOR_MOD_UB_BANK_GUARD_BYTES;
    }
    if (c.packed) {
        fixed += ubLayout.Align(sizes.outputFpElems * sizeof(float));
        fixed += 2U * ubLayout.Align(sizes.outputBufferElems * sizeof(float));
        return fixed + 2U * FLOOR_MOD_UB_BANK_GUARD_BYTES;
    }
    return fixed + 2U * ubLayout.Align(sizes.outputFpElems * sizeof(float)) + FLOOR_MOD_UB_BANK_GUARD_BYTES;
}

static bool ReserveCrossUb(CrossCandidate& c, const CrossDecomp& dec, const CrossBufferSizes& sizes, uint32_t outerTile,
                           uint32_t aTile, uint32_t dtypeBytes, bool isFp32, uint64_t ubLimit, const UbLayout& ubLayout)
{
    uint64_t fixed = 2U * ubLayout.Align(sizes.seedElems * dtypeBytes) +
                     2U * ubLayout.Align(sizes.denseTElems * dtypeBytes) + 2U * FLOOR_MOD_UB_BANK_GUARD_BYTES;
    if (!isFp32) {
        fixed += ubLayout.Align(sizes.seedElems * sizeof(float));
        fixed += ubLayout.Align(sizes.denseFpElems * sizeof(float)) + 2U * FLOOR_MOD_UB_BANK_GUARD_BYTES;
    }
    fixed += CrossOutputFixedBytes(c, dec, sizes, outerTile, aTile, dtypeBytes, isFp32, ubLayout);
    if (fixed > ubLimit) {
        return false;
    }
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    if (dec.d == 1U || c.packed) {
        QueryFloorTmp(sizes.outputFpElems, floorMax, floorMin);
    }
    const uint64_t scratchMax = std::max({floorMax, c.broadcastTmpBytes, ubLayout.blockBytes});
    const uint32_t floorChosen = fixed + ubLayout.Align(scratchMax) <= ubLimit ? floorMax : floorMin;
    const uint64_t used = fixed + ubLayout.Align(std::max({floorChosen, c.broadcastTmpBytes, ubLayout.blockBytes}));
    if (used > ubLimit) {
        return false;
    }
    c.floorTmpBytes = floorChosen;
    c.ubUsedBytes = static_cast<uint32_t>(used);
    return true;
}

static void ConfigureCrossTasks(CrossCandidate& c, const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile,
                                uint32_t bTile)
{
    c.outerTile = outerTile;
    c.aTile = aTile;
    c.bTile = bTile;
    c.outerCount = static_cast<uint32_t>(CeilDiv(dec.outer, outerTile));
    c.aCnt = static_cast<uint32_t>(CeilDiv(dec.a, aTile));
    c.bCount = static_cast<uint32_t>(CeilDiv(dec.b, bTile));
    c.totalTasks = static_cast<uint64_t>(c.outerCount) * c.aCnt * c.bCount;
}

static uint64_t CrossOutputDmaCommands(const CrossCandidate& c, const CrossDecomp& dec, bool unitD)
{
    if (c.packed) {
        if (c.aCnt == 1U && c.bCount == 1U) {
            return c.outerCount;
        }
        return c.bCount == 1U ? dec.outer * c.aCnt : dec.outer * dec.a * dec.m * c.bCount;
    }
    if (unitD) {
        return c.aCnt == 1U ? static_cast<uint64_t>(c.outerCount) * c.bCount : dec.outer * c.aCnt * c.bCount;
    }
    if (c.aCnt == 1U && c.bCount == 1U) {
        return c.outerCount;
    }
    return c.bCount == 1U ? dec.outer * c.aCnt : dec.outer * dec.a * c.bCount;
}

static uint64_t ScoreCrossCandidate(const CrossCandidate& c, const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile,
                                    uint32_t dtypeBytes, uint32_t availableCores, const UbLayout& ubLayout)
{
    const bool unitD = dec.d == 1U;
    const uint64_t active = std::max<uint64_t>(1U, std::min<uint64_t>(availableCores, c.totalTasks));
    const uint64_t waves = CeilDiv(c.totalTasks, active);
    const uint64_t seedLoads = dec.outer * dec.a * dec.m * dec.d * c.bCount;
    const uint64_t denseLoads = dec.outer * dec.m * dec.b * dec.d * c.aCnt;
    const uint64_t seedDma = c.aCnt == 1U ? static_cast<uint64_t>(c.outerCount) * c.bCount :
                                            dec.outer * c.aCnt * c.bCount;
    const uint64_t denseDma = unitD          ? c.totalTasks :
                              c.bCount == 1U ? static_cast<uint64_t>(c.outerCount) * c.aCnt :
                                               dec.outer * c.aCnt * c.bCount;
    const uint64_t outputDma = CrossOutputDmaCommands(c, dec, unitD);
    const uint64_t vectorElems = (unitD || c.packed) ? dec.outer * dec.a * dec.m * c.unitBAligned * c.bCount :
                                                       dec.outer * dec.a * dec.m * dec.b * c.dAligned;
    const uint64_t brcbCommands = unitD ?
                                      c.totalTasks * CeilDiv(static_cast<uint64_t>(outerTile) * aTile * dec.m, 2040U) :
                                      0U;
    const uint64_t paddedBlocks = dec.outer * dec.a * dec.m * dec.b;
    const uint64_t inputBlocks = dec.outer * dec.a * dec.m * c.bCount + dec.outer * dec.m * dec.b * c.aCnt;
    const uint64_t packedBlocks = c.aCnt == 1U && c.bCount == 1U ?
                                      c.outerCount :
                                      (c.bCount == 1U ? dec.outer * c.aCnt : dec.outer * dec.a * dec.m * c.bCount);
    const uint64_t dmaBlocks = inputBlocks + (c.packed ? packedBlocks : paddedBlocks);
    const uint64_t gatherElems = c.packed ? dec.outer * dec.a * dec.m * dec.b * dec.d : 0U;
    return waves * 96U + CeilDiv(c.totalTasks * 64U, active) +
           CeilDiv((seedLoads + denseLoads) * dtypeBytes, active * ubLayout.blockBytes) +
           CeilDiv((seedDma + denseDma) * FLOOR_MOD_DMA_COMMAND_COST, active) + CeilDiv(outputDma * 64U, active) +
           CeilDiv(brcbCommands * 16U, active) + CeilDiv(dmaBlocks * FLOOR_MOD_DMA_INTERNAL_BLOCK_COST, active) +
           CeilDiv(gatherElems, active * 64U) + CeilDiv(vectorElems, active * 64U);
}

static CrossCandidate BuildCrossCandidate(const CrossDecomp& dec, uint32_t outerTile, uint32_t aTile, uint32_t bTile,
                                          uint32_t dtypeBytes, bool isFp32, uint32_t availableCores, uint64_t ubLimit,
                                          const UbLayout& ubLayout, bool packed = false,
                                          uint32_t compactBroadcastTmpBytes = 0U)
{
    CrossCandidate c;
    if (!IsCrossCandidateShapeValid(dec, outerTile, aTile, bTile, dtypeBytes, availableCores, ubLayout, packed)) {
        return c;
    }
    if (!ConfigureCrossLayout(c, dec, outerTile, aTile, bTile, dtypeBytes, ubLayout, packed,
                              compactBroadcastTmpBytes)) {
        return c;
    }
    CrossBufferSizes sizes;
    if (!ComputeCrossBufferSizes(sizes, c, dec, outerTile, aTile, bTile, dtypeBytes, ubLayout)) {
        return c;
    }

    if (!ReserveCrossUb(c, dec, sizes, outerTile, aTile, dtypeBytes, isFp32, ubLimit, ubLayout)) {
        return c;
    }

    ConfigureCrossTasks(c, dec, outerTile, aTile, bTile);
    c.score = ScoreCrossCandidate(c, dec, outerTile, aTile, dtypeBytes, availableCores, ubLayout);
    c.valid = true;
    return c;
}

static bool BuildDenseTile(uint32_t tile, uint32_t dtypeBytes, bool isFp32, uint64_t ubLimit, uint32_t& floorTmpBytes,
                           uint32_t& ubUsedBytes, const UbLayout& ubLayout)
{
    const uint32_t aligned = ubLayout.AlignElements(tile, dtypeBytes);
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    QueryFloorTmp(aligned, floorMax, floorMin);
    // DenseVector owns three T slabs (x1, x2, y), two FP32 conversion slabs
    // for non-FP32 inputs, and two non-overlapping FP32 quotient slabs for

    // Floor. Keep this accounting in lockstep with InitDenseBuffers(); the
    // Floor API rejects input/output aliasing, so the second quotient slab is
    // part of the real UB footprint.
    uint64_t fixed = 3U * ubLayout.Align(static_cast<uint64_t>(aligned) * dtypeBytes) +
                     2U * ubLayout.Align(static_cast<uint64_t>(aligned) * sizeof(float));
    if (!isFp32) {
        fixed += 2U * ubLayout.Align(static_cast<uint64_t>(aligned) * sizeof(float));
    }
    uint32_t chosen = floorMax;
    if (fixed + ubLayout.Align(chosen) > ubLimit) {
        chosen = floorMin;
    }
    const uint64_t used = fixed + ubLayout.Align(chosen);
    if (used > ubLimit) {
        return false;
    }
    floorTmpBytes = chosen;
    ubUsedBytes = static_cast<uint32_t>(used);
    return true;
}

struct DenseCandidate {
    uint32_t tile = 1U;
    uint32_t cores = 1U;
    uint32_t floorTmpBytes = 0U;
    uint32_t ubUsedBytes = 0U;
};

static void AddDenseCandidate(std::vector<DenseCandidate>& candidates, uint32_t tile, uint32_t cores,
                              uint32_t dtypeBytes, bool isFp32, uint64_t ubLimit, const UbLayout& ubLayout)
{
    if (tile == 0U || cores == 0U) {
        return;
    }
    DenseCandidate candidate;
    candidate.tile = tile;
    candidate.cores = cores;
    if (!BuildDenseTile(tile, dtypeBytes, isFp32, ubLimit, candidate.floorTmpBytes, candidate.ubUsedBytes, ubLayout)) {
        return;
    }
    const auto duplicate = std::find_if(candidates.begin(), candidates.end(), [&](const DenseCandidate& item) {
        return item.tile == candidate.tile && item.cores == candidate.cores;
    });
    if (duplicate == candidates.end()) {
        candidates.push_back(candidate);
    }
}

// Candidate zero preserves the deterministic analytical plan. Offline sweep
// mode then exposes nearby and full legal core counts, followed by smaller UB
// tiles. This makes dense selection measurable without encoding shape IDs.
static std::vector<DenseCandidate> FindDenseCandidates(uint64_t totalElements, uint32_t dtypeBytes, bool isFp32,
                                                       uint32_t availableCores, uint64_t ubLimit,
                                                       const UbLayout& ubLayout)
{
    const uint64_t logicalTotal = totalElements == 0U ? 1U : totalElements;
    uint32_t maxTile = static_cast<uint32_t>(std::min<uint64_t>(logicalTotal, FLOOR_MOD_EXPAND_TARGET_CORE_ELEMS));
    uint32_t floorTmpBytes = 0U;
    uint32_t ubUsedBytes = 0U;
    while (maxTile > 0U &&
           !BuildDenseTile(maxTile, dtypeBytes, isFp32, ubLimit, floorTmpBytes, ubUsedBytes, ubLayout)) {
        maxTile /= 2U;
    }
    if (maxTile == 0U) {
        return {};
    }

    const uint32_t maxCores = static_cast<uint32_t>(std::min<uint64_t>(availableCores, logicalTotal));
    const uint32_t desired = static_cast<uint32_t>(
        std::min<uint64_t>(maxCores, std::max<uint64_t>(1U, CeilDiv(logicalTotal, FLOOR_MOD_ROW_TARGET_CORE_ELEMS))));
    std::vector<DenseCandidate> candidates;
    AddDenseCandidate(candidates, maxTile, desired, dtypeBytes, isFp32, ubLimit, ubLayout);

    const auto addCore = [&](uint32_t cores) {
        if (cores > 0U && cores <= maxCores) {
            AddDenseCandidate(candidates, maxTile, cores, dtypeBytes, isFp32, ubLimit, ubLayout);
        }
    };
    addCore(desired > 1U ? desired - 1U : 1U);
    addCore(desired + 1U);
    addCore(static_cast<uint32_t>(CeilDiv(desired, 2U)));
    addCore(1U);
    addCore(maxCores);
    for (uint32_t cores = 1U; cores <= maxCores; ++cores) {
        addCore(cores);
    }

    for (uint32_t tile = maxTile / 2U; tile > 0U; tile /= 2U) {
        AddDenseCandidate(candidates, tile, desired, dtypeBytes, isFp32, ubLimit, ubLayout);
        AddDenseCandidate(candidates, tile, std::min<uint32_t>(maxCores, desired + 1U), dtypeBytes, isFp32, ubLimit,
                          ubLayout);
        AddDenseCandidate(candidates, tile, desired > 1U ? desired - 1U : 1U, dtypeBytes, isFp32, ubLimit, ubLayout);
    }
    return candidates;
}

// ML is allowed to change paths freely because the analytical scores are not
// comparable across different implementations. Within one implementation,
// however, a large analytical regression means the tree is extrapolating a
// schedule or tile choice without enough local counterfactual support.

static uint32_t StandardCandidateLayout(const StandardCandidate& candidate)
{
    if (candidate.denseTail) {
        return FLOOR_MOD_REUSE_DENSE_TAIL_BATCH;
    }
    if (candidate.compactPos) {
        return FLOOR_MOD_REUSE_COMPACT_POS_BROADCAST;
    }
    if (candidate.compactRepeat) {
        return FLOOR_MOD_REUSE_COMPACT_ROW_REPEAT;
    }
    if (candidate.compactRows) {
        return FLOOR_MOD_REUSE_COMPACT_ROW_BATCH;
    }
    if (candidate.materialized) {
        return FLOOR_MOD_REUSE_PADDED_MATERIALIZED;
    }
    return candidate.packed ? FLOOR_MOD_REUSE_PACKED_BROADCAST : FLOOR_MOD_REUSE_PADDED_REPEAT;
}

static void ApplyDenseCandidate(FloorModTilingData& tiling, const DenseCandidate& candidate, uint32_t dtypeBytes,
                                const UbLayout& ubLayout)
{
    tiling.mode = FLOOR_MOD_MODE_DENSE;
    tiling.seedIsX1 = 0U;
    tiling.swapped = 0U;
    tiling.denseTile = candidate.tile;
    tiling.floorTmpBytes = candidate.floorTmpBytes;
    tiling.maxTRowElems = ubLayout.AlignElements(candidate.tile, dtypeBytes);
    tiling.maxFp32RowElems = tiling.maxTRowElems;
    tiling.ubUsedBytes = candidate.ubUsedBytes;
    tiling.coreNum = candidate.cores;
}

static void ApplyCrossCandidate(FloorModTilingData& tiling, const CrossCandidate& candidate,
                                const CrossDecomp& decomposition, bool seedIsX1, uint32_t availableCores)
{
    tiling.mode = FLOOR_MOD_MODE_CROSSED;
    tiling.seedIsX1 = seedIsX1 ? 1U : 0U;
    tiling.swapped = tiling.seedIsX1;
    tiling.crossOuter = static_cast<uint32_t>(decomposition.outer);
    tiling.crossA = static_cast<uint32_t>(decomposition.a);
    tiling.crossM = static_cast<uint32_t>(decomposition.m);
    tiling.crossB = static_cast<uint32_t>(decomposition.b);
    tiling.crossD = static_cast<uint32_t>(decomposition.d);
    tiling.crossDAligned = candidate.dAligned;
    tiling.crossOuterTile = candidate.outerTile;
    tiling.crossATile = candidate.aTile;
    tiling.crossBTile = candidate.bTile;
    tiling.crossUnitBTAligned = candidate.unitBTAligned;
    tiling.crossUnitBAligned = candidate.unitBAligned;
    tiling.crossOuterTileCount = candidate.outerCount;
    tiling.crossATileCount = candidate.aCnt;
    tiling.crossBTileCount = candidate.bCount;
    tiling.crossTotalTasks = candidate.totalTasks;
    tiling.reuseLayout = candidate.packed ? FLOOR_MOD_REUSE_PADDED_COMPACT : FLOOR_MOD_REUSE_PADDED_REPEAT;
    tiling.floorTmpBytes = candidate.floorTmpBytes;
    tiling.broadcastTmpBytes = candidate.broadcastTmpBytes;
    tiling.ubUsedBytes = candidate.ubUsedBytes;
    tiling.coreNum = static_cast<uint32_t>(std::min<uint64_t>(availableCores, candidate.totalTasks));
}

static void ConfigureStandardTilingFields(FloorModTilingData& tiling, const StandardCandidate& candidate,
                                          const ReuseDecomp& decomposition, bool seedIsX1, uint32_t availableCores)
{
    tiling.mode = decomposition.e2 > 1U ? FLOOR_MOD_MODE_EXPAND_REUSE : FLOOR_MOD_MODE_ROW_REUSE;
    tiling.seedIsX1 = seedIsX1 ? 1U : 0U;
    tiling.swapped = tiling.seedIsX1;
    tiling.e1 = static_cast<uint32_t>(decomposition.e1);
    tiling.D = decomposition.d;
    tiling.e2 = static_cast<uint32_t>(decomposition.e2);
    tiling.dTile = candidate.dTile;
    tiling.dTileCount = candidate.dCount;
    tiling.e2Tile = candidate.e2Tile;
    tiling.e2TileCount = candidate.e2Count;
    tiling.reuseSchedule = candidate.rowOwned ? 1U : 0U;
    tiling.reuseLayout = StandardCandidateLayout(candidate);
    tiling.rowPartitions = candidate.rowPartitions;
    tiling.batchRows = candidate.batchRows;
    tiling.posTile = candidate.posTile;
    tiling.totalTasks = candidate.totalTasks;
    tiling.maxPatternElems = candidate.patternElems;
    tiling.maxTRowElems = candidate.tRowElems;
    tiling.maxFp32RowElems = candidate.fp32RowElems;
    tiling.floorTmpBytes = candidate.floorTmpBytes;
    tiling.broadcastTmpBytes = candidate.broadcastTmpBytes;
    tiling.ubUsedBytes = candidate.ubUsedBytes;
    tiling.scalarDoubleBuffer = candidate.scalarDoubleBuffered ? 1U : 0U;
    tiling.posTotal = static_cast<uint32_t>(decomposition.pos);
    tiling.posDigits = decomposition.posDigits;
    tiling.coreNum = static_cast<uint32_t>(std::min<uint64_t>(availableCores, candidate.totalTasks));
}

static void ConfigureStandardPositionStrides(FloorModTilingData& tiling, const ReuseDecomp& decomposition,
                                             bool seedIsX1, const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x1,
                                             const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x2,
                                             const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& out,
                                             uint32_t rank)
{
    const auto& seedDims = seedIsX1 ? x1 : x2;
    const auto& denseDims = seedIsX1 ? x2 : x1;
    std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> seedStrides{};
    std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> denseStrides{};
    CalcBroadcastStrides(seedDims, rank, seedStrides);
    CalcBroadcastStrides(denseDims, rank, denseStrides);
    for (uint32_t i = 0U; i < decomposition.posDigits; ++i) {
        const uint32_t dim = decomposition.posDims[i];
        tiling.posExtent[i] = static_cast<uint32_t>(out[dim]);
        tiling.seedStride[i] = seedStrides[dim];
        tiling.denseStride[i] = denseStrides[dim];
    }
}

static uint32_t StandardKernelPath(const StandardCandidate& candidate)
{
    if (candidate.denseTail) {
        return FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH;
    }
    if (candidate.compactRows || candidate.compactRepeat) {
        return FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH;
    }
    return FLOOR_MOD_TPL_PATH_GENERAL;
}

static uint32_t ApplyStandardCandidate(FloorModTilingData& tiling, const StandardCandidate& candidate,
                                       const ReuseDecomp& decomposition, bool seedIsX1,
                                       const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x1,
                                       const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& x2,
                                       const std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM>& out, uint32_t rank,
                                       uint32_t availableCores)
{
    ConfigureStandardTilingFields(tiling, candidate, decomposition, seedIsX1, availableCores);
    ConfigureStandardPositionStrides(tiling, decomposition, seedIsX1, x1, x2, out, rank);
    return StandardKernelPath(candidate);
}

} // namespace FloorModNs

using namespace FloorModNs;

namespace optiling {

struct FloorModCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};

static void LogFloorModPlan(gert::TilingContext* context, const FloorModTilingData& tiling, uint32_t kernelPath)
{
    OP_LOGD(context, "FloorMod mode=%u kernelPath=%u cores=%u elements=%lu tile=%u tasks=%lu ub=%u.", tiling.mode,
            kernelPath, tiling.coreNum, tiling.totalElements, tiling.denseTile,
            tiling.mode == FLOOR_MOD_MODE_CROSSED ? tiling.crossTotalTasks : tiling.totalTasks, tiling.ubUsedBytes);
}

static bool ResolveFloorModDtype(ge::DataType dtype, uint32_t& dtypeKey, uint32_t& dtypeBytes)
{
    switch (dtype) {
        case ge::DataType::DT_FLOAT16:
            dtypeKey = FLOOR_MOD_TPL_FP16;
            dtypeBytes = sizeof(uint16_t);
            return true;
        case ge::DataType::DT_BF16:
            dtypeKey = FLOOR_MOD_TPL_BF16;
            dtypeBytes = sizeof(uint16_t);
            return true;
        case ge::DataType::DT_FLOAT:
            dtypeKey = FLOOR_MOD_TPL_FP32;
            dtypeBytes = sizeof(float);
            return true;
        case ge::DataType::DT_INT32:
            dtypeKey = FLOOR_MOD_TPL_INT32;
            dtypeBytes = sizeof(int32_t);
            return true;
        case ge::DataType::DT_INT64:
            dtypeKey = FLOOR_MOD_TPL_INT64;
            dtypeBytes = sizeof(int64_t);
            return true;
        case ge::DataType::DT_DOUBLE:
            dtypeKey = FLOOR_MOD_TPL_DOUBLE;
            dtypeBytes = sizeof(double);
            return true;
        default:
            return false;
    }
}

static ge::graphStatus ConfigureFp64Tiling(gert::TilingContext* context, FloorModTilingData* tiling,
                                           uint64_t totalElements, uint32_t availableCores, uint64_t ubLimit,
                                           const UbLayout& ubLayout, uint32_t dtypeKey)
{
    OP_CHECK_IF(totalElements == 0U, OP_LOGE(context, "FP64 storage path requires a non-empty output."),
                return ge::GRAPH_FAILED);
    const uint32_t cores = static_cast<uint32_t>(std::min<uint64_t>(
        availableCores, std::max<uint64_t>(1U, CeilDiv(totalElements, FLOOR_MOD_FP64_TARGET_CORE_ELEMS))));
    const uint32_t tile = static_cast<uint32_t>(
        std::min<uint64_t>(FLOOR_MOD_FP64_STORAGE_TILE, std::max<uint64_t>(1U, CeilDiv(totalElements, cores))));
    const uint32_t alignedTile = ubLayout.AlignElements(tile, sizeof(double));
    uint32_t floorMax = 0U;
    uint32_t floorMin = 0U;
    QueryFloorTmp(alignedTile, floorMax, floorMin);
    const uint64_t scratchBytes = ubLayout.Align(std::max<uint64_t>(
        std::max<uint64_t>(static_cast<uint64_t>(alignedTile) * sizeof(uint32_t), FLOOR_MOD_UB_BANK_GUARD_BYTES),
        ubLayout.blockBytes));
    const uint64_t maskStride = ubLayout.Align(std::max<uint64_t>(
        std::max<uint64_t>((static_cast<uint64_t>(alignedTile) + 7U) / 8U, FLOOR_MOD_UB_BANK_GUARD_BYTES),
        ubLayout.blockBytes));
    const uint64_t fixedBytes = 2U * ubLayout.Align(static_cast<uint64_t>(alignedTile) * sizeof(uint64_t)) +
                                3U * ubLayout.Align(static_cast<uint64_t>(alignedTile) * sizeof(float)) +
                                5U * scratchBytes + 2U * maskStride;
    const uint32_t floorChosen = fixedBytes + ubLayout.Align(floorMax) <= ubLimit ? floorMax : floorMin;
    const uint64_t usedBytes = fixedBytes + ubLayout.Align(std::max(floorChosen, ubLayout.blockBytes));
    OP_CHECK_IF(usedBytes > ubLimit || usedBytes > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "FP64 storage path exceeds UB: need %lu, limit %lu.", usedBytes, ubLimit),
                return ge::GRAPH_FAILED);
    tiling->mode = FLOOR_MOD_MODE_DENSE;
    tiling->coreNum = cores;
    tiling->denseTile = tile;
    tiling->maxTRowElems = alignedTile;
    tiling->maxFp32RowElems = alignedTile;
    tiling->floorTmpBytes = floorChosen;
    tiling->ubUsedBytes = static_cast<uint32_t>(usedBytes);
    context->SetBlockDim(cores);
    context->SetTilingKey(GET_TPL_TILING_KEY(dtypeKey, dtypeKey, dtypeKey, FLOOR_MOD_TPL_PATH_FP64_STORAGE));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0U;
    LogFloorModPlan(context, *tiling, FLOOR_MOD_TPL_PATH_FP64_STORAGE);
    return ge::GRAPH_SUCCESS;
}

class FloorModTiling {
public:
    explicit FloorModTiling(gert::TilingContext* ctx) : context(ctx) {}

    ge::graphStatus Run()
    {
        OP_CHECK_IF(Initialize() != ge::GRAPH_SUCCESS, OP_LOGE(context, "FloorMod initialization failed."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(InitializeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(context, "FloorMod shape initialization failed."),
                    return ge::GRAPH_FAILED);
        if (dtype == ge::DT_DOUBLE && totalElements != 0U) {
            return ConfigureFp64Tiling(context, tiling, totalElements, availableCores, ubLimit, ubLayout, dtypeKey);
        }
        const auto status = x1 == x2 || totalElements == 0U ? SelectDense() : SelectBroadcast();
        OP_CHECK_IF(status != ge::GRAPH_SUCCESS, OP_LOGE(context, "FloorMod has no legal execution plan."),
                    return status);
        tiling->coreNum = std::max(tiling->coreNum, 1U);
        context->SetBlockDim(tiling->coreNum);
        context->SetTilingKey(GET_TPL_TILING_KEY(dtypeKey, dtypeKey, dtypeKey, selectedKernelPath));
        size_t* workspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
        workspace[0] = 0U;
        LogFloorModPlan(context, *tiling, selectedKernelPath);
        return ge::GRAPH_SUCCESS;
    }

private:
    ge::graphStatus Initialize()
    {
        const auto* info = reinterpret_cast<const FloorModCompileInfo*>(context->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context, info);
        ubLayout.blockBytes = Ops::Base::GetUbBlockSize(context);
        auto* x1Desc = context->GetInputDesc(0);
        auto* x2Desc = context->GetInputDesc(1);
        auto* yDesc = context->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
        OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
        OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
        dtype = x1Desc->GetDataType();
        OP_CHECK_IF(dtype != x2Desc->GetDataType() || dtype != yDesc->GetDataType(),
                    OP_LOGE(context, "FloorMod inputs and output must use one dtype."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!ResolveFloorModDtype(dtype, dtypeKey, dtypeBytes),
                    OP_LOGE(context, "FloorMod unsupported dtype %d.", static_cast<int32_t>(dtype)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!IsValidLayout(ubLayout, dtypeBytes), OP_LOGE(context, "FloorMod dtype or UB block size is zero."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(info->totalCoreNum <= 0 || info->ubSize <= static_cast<int64_t>(FLOOR_MOD_UB_GUARD_BYTES),
                    OP_LOGE(context, "FloorMod invalid core count or UB size."), return ge::GRAPH_FAILED);
        availableCores = static_cast<uint32_t>(info->totalCoreNum);
        ubLimit = static_cast<uint64_t>(info->ubSize) - FLOOR_MOD_UB_GUARD_BYTES;
        isFp32 = dtype == ge::DT_FLOAT;
        tiling = context->GetTilingData<FloorModTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        *tiling = {};
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus InitializeShapes()
    {
        OP_CHECK_IF(GetBroadcastShapes(context, x1, x2, out, rank) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "FloorMod invalid broadcast shapes."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!ProductDims(out, rank, totalElements) || !ProductDims(x1, rank, tiling->x1Elements) ||
                        !ProductDims(x2, rank, tiling->x2Elements),
                    OP_LOGE(context, "FloorMod shape product overflow."), return ge::GRAPH_FAILED);
        tiling->dtypeKey = dtypeKey;
        tiling->dtypeBytes = dtypeBytes;
        tiling->rank = rank;
        tiling->totalElements = totalElements;
        std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> x1Strides{};
        std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> x2Strides{};
        CalcBroadcastStrides(x1, rank, x1Strides);
        CalcBroadcastStrides(x2, rank, x2Strides);
        for (uint32_t i = 0U; i < FLOOR_MOD_MAX_BROADCAST_DIM; ++i) {
            tiling->outShape[i] = out[i];
            tiling->x1Stride[i] = x1Strides[i];
            tiling->x2Stride[i] = x2Strides[i];
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SelectDense()
    {
        const auto candidates = FindDenseCandidates(totalElements, dtypeBytes, isFp32, availableCores, ubLimit,
                                                    ubLayout);
        OP_CHECK_IF(candidates.empty(), OP_LOGE(context, "No valid FloorMod dense tile."), return ge::GRAPH_FAILED);
        ApplyDenseCandidate(*tiling, candidates.front(), dtypeBytes, ubLayout);
        if (totalElements > 0U && totalElements < FLOOR_MOD_SMALL_DENSE_MAX_ELEMENTS &&
            tiling->denseTile >= CeilDiv(totalElements, tiling->coreNum)) {
            selectedKernelPath = FLOOR_MOD_TPL_PATH_SMALL_DENSE;
        }
        return ge::GRAPH_SUCCESS;
    }

    bool SelectReuseTopology(ReuseDecomp& reuse) const
    {
        const auto first = DecomposeReuse(x1, x2, out, rank);
        const auto second = DecomposeReuse(x2, x1, out, rank);
        const bool seedIsX1 = first.valid &&
                              (!second.valid || first.pos < second.pos ||
                               (first.pos == second.pos && first.pos * first.e1 < second.pos * second.e1));
        reuse = seedIsX1 ? first : second;
        return seedIsX1;
    }

    StandardCandidate SelectReuse(const ReuseDecomp& reuse, bool globalScalar) const
    {
        if (!reuse.valid || reuse.pos > UINT32_MAX || reuse.e1 > UINT32_MAX || reuse.e2 > UINT32_MAX) {
            return {};
        }
        if (globalScalar) {
            const auto scalar = FindMultiWaveScalarBroadcastCandidate(
                totalElements, dtypeBytes, isFp32, dtype == ge::DT_BF16, availableCores, ubLimit, ubLayout);
            if (scalar.valid) {
                return scalar;
            }
        }
        const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const auto candidates = FindStandardCandidates(reuse, dtypeBytes, isFp32, availableCores, ubLimit, platform,
                                                       ubLayout);
        return candidates.empty() ? StandardCandidate{} : candidates.front();
    }

    CrossCandidate SelectCross(const CrossDecomp& cross) const
    {
        if (!cross.valid || cross.outer > UINT32_MAX || cross.a > UINT32_MAX || cross.m > UINT32_MAX ||
            cross.b > UINT32_MAX || cross.d > UINT32_MAX) {
            return {};
        }
        const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t bAlign = ubLayout.blockBytes / std::gcd<uint64_t>(ubLayout.blockBytes, cross.d * dtypeBytes);
        const auto aTiles = MakeTileCandidates(cross.a, 1U);
        const auto bTiles = MakeTileCandidates(cross.b, bAlign);
        const auto outerTiles = MakeTileCandidates(cross.outer, 1U, availableCores);
        CrossCandidate best;
        for (uint32_t outerTile : outerTiles) {
            for (uint32_t aTile : aTiles) {
                for (uint32_t bTile : bTiles) {
                    const uint32_t tmp = cross.d > 1U ?
                                             QueryTrailingBroadcastTmp(platform, bTile,
                                                                       ubLayout.AlignElements(cross.d, sizeof(float))) :
                                             0U;
                    for (bool packed : {false, true}) {
                        const auto candidate = BuildCrossCandidate(cross, outerTile, aTile, bTile, dtypeBytes, isFp32,
                                                                   availableCores, ubLimit, ubLayout, packed,
                                                                   packed ? tmp : 0U);
                        if (candidate.valid && (!best.valid || candidate.score < best.score)) {
                            best = candidate;
                        }
                    }
                }
            }
        }
        return best;
    }

    bool SupportsScalarTiles() const
    {
        return tiling->D == 1U && tiling->e1 == 1U && tiling->e2TileCount == tiling->totalTasks &&
               tiling->maxTRowElems >= tiling->e2Tile &&
               tiling->e2Tile <=
                   FLOOR_MOD_MAX_REPEAT_ROWS * FLOOR_MOD_GATHER_MASK_ELEMS + FLOOR_MOD_GATHER_MASK_ELEMS - 1U &&
               ScalarBroadcastUbBytes(tiling->maxTRowElems, tiling->maxFp32RowElems, tiling->floorTmpBytes, dtypeBytes,
                                      ubLayout, isFp32, dtype == ge::DT_BF16,
                                      tiling->scalarDoubleBuffer != 0U ? 2U : 1U) <= ubLimit;
    }

    ge::graphStatus SelectBroadcast()
    {
        ReuseDecomp reuse;
        const bool reuseSeedIsX1 = SelectReuseTopology(reuse);
        const bool scalar = (reuseSeedIsX1 ? tiling->x1Elements : tiling->x2Elements) == 1U;
        const auto reuseBest = SelectReuse(reuse, scalar);
        const auto crossX1 = DecomposeCrossed(x1, x2, out, rank);
        const bool crossSeedIsX1 = crossX1.valid;
        const auto cross = crossSeedIsX1 ? crossX1 : DecomposeCrossed(x2, x1, out, rank);
        const auto crossBest = SelectCross(cross);
        const bool useCross = PreferCrossRoute(cross, crossBest, reuseBest, availableCores);
        OP_CHECK_IF(!useCross && !reuseBest.valid, OP_LOGE(context, "No legal batched FloorMod execution plan."),
                    return ge::GRAPH_FAILED);
        if (useCross) {
            const uint64_t matrixBlocks = cross.m * crossBest.unitBAligned / FLOOR_MOD_FP32_BLOCK_ELEMS;
            const bool fma = cross.d == 1U && isFp32 && cross.m <= FLOOR_MOD_MAX_REPEAT_ROWS &&
                             matrixBlocks <= FLOOR_MOD_MAX_REPEAT_ROWS;
            tiling->arithmeticMode = fma ? FLOOR_MOD_ARITH_FMA_NEG_DENOMINATOR : FLOOR_MOD_ARITH_MUL_SUB;
            ApplyCrossCandidate(*tiling, crossBest, cross, crossSeedIsX1, availableCores);
        } else {
            selectedKernelPath = ApplyStandardCandidate(*tiling, reuseBest, reuse, reuseSeedIsX1, x1, x2, out, rank,
                                                        availableCores);
            if (scalar && SupportsScalarTiles()) {
                selectedKernelPath = FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    gert::TilingContext* context;
    FloorModTilingData* tiling = nullptr;
    UbLayout ubLayout{};
    ge::DataType dtype = ge::DT_FLOAT;
    uint32_t dtypeKey = FLOOR_MOD_TPL_FP32;
    uint32_t dtypeBytes = sizeof(float);
    uint32_t availableCores = 0U;
    uint32_t rank = 0U;
    uint32_t selectedKernelPath = FLOOR_MOD_TPL_PATH_GENERAL;
    uint64_t totalElements = 0U;
    uint64_t ubLimit = 0U;
    bool isFp32 = true;
    std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> x1{};
    std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> x2{};
    std::array<uint64_t, FLOOR_MOD_MAX_BROADCAST_DIM> out{};
};

static ge::graphStatus FloorModTilingForGe(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("FloorMod", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    return FloorModTiling(context).Run();
}

static ge::graphStatus TilingPrepare4FloorModTiling(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<FloorModCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = platform.GetCoreNumAiv();
    compileInfo->isRegbase = Ops::Base::IsRegbaseSocVersion(context);
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0,
                OP_LOGE(context, "FloorMod failed to query hardware."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FloorMod).Tiling(FloorModTilingForGe).TilingParse<FloorModCompileInfo>(TilingPrepare4FloorModTiling);

} // namespace optiling
