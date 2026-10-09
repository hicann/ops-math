/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOOR_MOD_ENTRY_H
#define FLOOR_MOD_ENTRY_H

#include "floor_mod_tiling_data.h"
#include "floor_mod_tiling_key.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_operator.h"

namespace FloorModNs {
using namespace AscendC;

constexpr uint32_t FLOOR_MOD_VECTOR_SEGMENT = 64U;
constexpr uint32_t FLOOR_MOD_MAX_VECTOR_REPEATS = 255U;
constexpr uint32_t FLOOR_MOD_UB_BLOCK_BYTES = GetDataBlockSizeInBytes();
// The count-mode int64<->fp32 conversion uses 32 source elements per vector
// repeat and the hardware repeat field is limited to 255. Keep a little
// alignment headroom so every scalar tile can be computed in safe sub-tiles.
constexpr uint32_t FLOOR_MOD_SCALAR_COMPUTE_CHUNK = 252U * 32U;
constexpr uint32_t FLOOR_MOD_FP32_BLOCK_ELEMS = 8U;
constexpr uint32_t FLOOR_MOD_UB_BANK_GUARD_BYTES = 256U;

template <typename T>
__aicore__ inline void CastToFp32(LocalTensor<float> dst, LocalTensor<T> src, uint32_t count)
{
    if constexpr (std::is_same_v<T, int64_t>) {
        Cast(dst, src, AscendC::RoundMode::CAST_ROUND, count);
    } else {
        Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    }
}

template <typename T>
__aicore__ inline void CastFromFp32(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
    if constexpr (std::is_same_v<T, half>) {
        Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        Cast(dst, src, AscendC::RoundMode::CAST_TRUNC, count);
    } else {
        Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
    }
}

__aicore__ inline void ComputeFloorRemainder(LocalTensor<float> quotient, LocalTensor<float> dense,
                                             LocalTensor<float> seed, LocalTensor<uint8_t> floorTmp, uint32_t count,
                                             bool swapped)
{
    if (swapped) {
        Div(quotient, seed, dense, count);
    } else {
        Div(quotient, dense, seed, count);
    }
    Floor(quotient, quotient, floorTmp, count);
    if (swapped) {
        Mul(quotient, quotient, dense, count);
        Sub(dense, seed, quotient, count);
    } else {
        Mul(quotient, quotient, seed, count);
        Sub(dense, dense, quotient, count);
    }
}

template <bool Remainder>
__aicore__ inline void ComputeBroadcastStage(LocalTensor<float> quotient, LocalTensor<float> dense,
                                             LocalTensor<float> seed, uint32_t count, uint8_t repeats,
                                             const BinaryRepeatParams& normal, const BinaryRepeatParams& swapped,
                                             const BinaryRepeatParams& bothRows, bool seedIsDividend)
{
    if constexpr (!Remainder) {
        if (seedIsDividend) {
            Div(quotient, seed, dense, static_cast<uint64_t>(count), repeats, swapped);
        } else {
            Div(quotient, dense, seed, static_cast<uint64_t>(count), repeats, normal);
        }
    } else if (seedIsDividend) {
        Mul(quotient, quotient, dense, static_cast<uint64_t>(count), repeats, bothRows);
        Sub(dense, seed, quotient, static_cast<uint64_t>(count), repeats, swapped);
    } else {
        Mul(quotient, quotient, seed, static_cast<uint64_t>(count), repeats, normal);
        Sub(dense, dense, quotient, static_cast<uint64_t>(count), repeats, bothRows);
    }
}

template <typename T, bool SmallDense = false>
class FloorModVector {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const FloorModTilingData* tiling)
    {
        t_ = tiling;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        if (t_->mode == FLOOR_MOD_MODE_DENSE) {
            InitDenseBuffers();
        } else if (t_->mode == FLOOR_MOD_MODE_CROSSED) {
            InitCrossBuffers();
        } else if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_POS_BROADCAST) {
            InitCompactPosBroadcastBuffers();
        } else {
            InitReuseBuffers();
        }
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= t_->coreNum || t_->totalElements == 0U) {
            return;
        }
        if (t_->mode == FLOOR_MOD_MODE_DENSE) {
            ProcessDense();
        } else if (t_->mode == FLOOR_MOD_MODE_CROSSED) {
            ProcessCrossed();
        } else if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_POS_BROADCAST) {
            ProcessCompactPosBroadcast();
        } else {
            ProcessReuse();
        }
    }

private:
    __aicore__ static inline uint32_t MinU32(uint64_t a, uint32_t b) { return static_cast<uint32_t>(a < b ? a : b); }

    __aicore__ static inline uint32_t MaxU32(uint32_t a, uint32_t b, uint32_t c)
    {
        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }

    __aicore__ static inline uint32_t AlignLocalElements(uint32_t elements)
    {
        constexpr uint32_t blockElements = FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T) > 8U ?
                                               FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T) :
                                               8U;
        return (elements + blockElements - 1U) / blockElements * blockElements;
    }

    __aicore__ inline void GetTaskRange(uint64_t totalTasks, uint64_t& begin, uint64_t& end) const
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t base = totalTasks / t_->coreNum;
        const uint64_t extra = totalTasks % t_->coreNum;
        begin = block * base + (block < extra ? block : extra);
        end = begin + base + (block < extra ? 1U : 0U);
    }

    __aicore__ inline void InitDenseBuffers()
    {
        denseCapacity_ = t_->maxTRowElems;
        if constexpr (SmallDense) {
            // SmallDense is selected only when every active core owns one
            // tile, so raw buffers have no cross-iteration ownership hazard.
            pipe_.InitBuffer(x1In_, denseCapacity_ * sizeof(T));
            pipe_.InitBuffer(x2In_, denseCapacity_ * sizeof(T));
        } else {
            pipe_.InitBuffer(denseX1Queue_, 1U, denseCapacity_ * sizeof(T));
            pipe_.InitBuffer(denseX2Queue_, 1U, denseCapacity_ * sizeof(T));
            pipe_.InitBuffer(denseYQueue_, 1U, denseCapacity_ * sizeof(T));
        }
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(aFp_, denseCapacity_ * sizeof(float));
            pipe_.InitBuffer(bFp_, denseCapacity_ * sizeof(float));
        }
        // Floor does not permit overlapping input/output tensors.
        pipe_.InitBuffer(remFp_, 2U * denseCapacity_ * sizeof(float));
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void InitReuseBuffers()
    {
        rowCapacity_ = t_->maxFp32RowElems;
        reuseSlotElems_ = t_->batchRows * rowCapacity_;
        reuseTSlotElems_ = t_->batchRows * t_->maxTRowElems;
        seedTCapacity_ = AlignLocalElements(t_->dTile);
        seedFpCapacity_ = (t_->dTile + FLOOR_MOD_FP32_BLOCK_ELEMS - 1U) / FLOOR_MOD_FP32_BLOCK_ELEMS *
                          FLOOR_MOD_FP32_BLOCK_ELEMS;
        pipe_.InitBuffer(seedIn_, seedTCapacity_ * sizeof(T));
        pipe_.InitBuffer(denseIn_, 2U * reuseTSlotElems_ * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(seedFp_, seedFpCapacity_ * sizeof(float));
            pipe_.InitBuffer(aFp_, reuseSlotElems_ * sizeof(float));
        }
        if (t_->mode == FLOOR_MOD_MODE_EXPAND_REUSE) {
            seedBroadcastCapacity_ = (t_->reuseLayout == FLOOR_MOD_REUSE_PACKED_BROADCAST ||
                                      t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_MATERIALIZED) ?
                                         t_->maxFp32RowElems :
                                         ((t_->dTile + 7U) / 8U) * 64U;
            pipe_.InitBuffer(expSeedFp_, seedBroadcastCapacity_ * sizeof(float));
        }
        pipe_.InitBuffer(remFp_, reuseSlotElems_ * sizeof(float));
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void InitCompactPosBroadcastBuffers()
    {
        posRowCapacity_ = t_->maxFp32RowElems;
        posSeedCapacity_ = t_->posTile * posRowCapacity_;
        posWorkCapacity_ = t_->posTile * t_->e1 * static_cast<uint32_t>(t_->D);
        posExpandedCapacity_ = t_->posTile * t_->e1 * posRowCapacity_;
        pipe_.InitBuffer(seedIn_, posSeedCapacity_ * sizeof(T));
        pipe_.InitBuffer(denseIn_, posWorkCapacity_ * sizeof(T));
        pipe_.InitBuffer(expSeedFp_, posExpandedCapacity_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(aFp_, posWorkCapacity_ * sizeof(float));
            pipe_.InitBuffer(seedFp_, posSeedCapacity_ * sizeof(float));
        }
        pipe_.InitBuffer(remFp_, posWorkCapacity_ * sizeof(float));
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void InitCrossUnitDBuffers()
    {
        crossSeedCapacity_ = AlignLocalElements(t_->crossOuterTile * t_->crossATile * t_->crossM);
        crossDenseTCapacity_ = t_->crossOuterTile * t_->crossM * t_->crossUnitBTAligned;
        crossDenseCapacity_ = t_->crossOuterTile * t_->crossM * t_->crossUnitBAligned;
        crossRowCapacity_ = t_->crossM * t_->crossUnitBAligned;
        crossTRowCapacity_ = t_->crossM * t_->crossUnitBTAligned;
        crossOutputCapacity_ = t_->crossOuterTile * t_->crossATile * crossRowCapacity_;
        crossOutputTCapacity_ = t_->crossOuterTile * t_->crossATile * crossTRowCapacity_;
        pipe_.InitBuffer(seedIn_, 2U * crossSeedCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(denseIn_, 2U * crossDenseTCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(seedFp_, crossSeedCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
            pipe_.InitBuffer(denseFp_, crossDenseCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
            pipe_.InitBuffer(outT_, 2U * crossOutputTCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        }
        seedBroadcastCapacity_ = ((crossSeedCapacity_ + 7U) / 8U) * 64U;
        pipe_.InitBuffer(expSeedFp_, seedBroadcastCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(expDenseFp_, 2U * crossOutputCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(remFp_, crossOutputCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void InitCrossCompactBuffers()
    {
        crossSeedCapacity_ = t_->crossOuterTile * t_->crossATile * t_->crossM * t_->crossDAligned;
        crossDenseCapacity_ = t_->crossOuterTile * t_->crossM * t_->crossUnitBAligned;
        crossDenseTCapacity_ = crossDenseCapacity_;
        crossRowCapacity_ = t_->crossM * t_->crossUnitBAligned;
        crossCompactCapacity_ = t_->crossOuterTile * t_->crossATile * crossRowCapacity_;
        const uint32_t paddedBroadcastCapacity = t_->crossOuterTile * t_->crossATile * t_->crossM * t_->crossBTile *
                                                 t_->crossDAligned;
        crossOutputCapacity_ = crossCompactCapacity_ > paddedBroadcastCapacity ? crossCompactCapacity_ :
                                                                                 paddedBroadcastCapacity;
        crossOutputTCapacity_ = t_->crossOuterTile * t_->crossATile * t_->crossM * t_->crossUnitBTAligned;
        pipe_.InitBuffer(seedIn_, 2U * crossSeedCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(denseIn_, 2U * crossDenseTCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(seedFp_, crossSeedCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
            pipe_.InitBuffer(denseFp_, crossDenseCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
            pipe_.InitBuffer(outT_, 2U * crossOutputTCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        }
        pipe_.InitBuffer(expSeedFp_, crossCompactCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(expDenseFp_, 2U * crossOutputCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void InitCrossPaddedBuffers()
    {
        crossSeedCapacity_ = t_->crossOuterTile * t_->crossATile * t_->crossM * t_->crossDAligned;
        crossDenseCapacity_ = t_->crossOuterTile * t_->crossM * t_->crossBTile * t_->crossDAligned;
        crossDenseTCapacity_ = crossDenseCapacity_;
        crossRowCapacity_ = t_->crossM * t_->crossBTile * t_->crossDAligned;
        crossOutputCapacity_ = t_->crossOuterTile * t_->crossATile * crossRowCapacity_;
        pipe_.InitBuffer(seedIn_, 2U * crossSeedCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(denseIn_, 2U * crossDenseTCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(seedFp_, crossSeedCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
            pipe_.InitBuffer(denseFp_, crossDenseCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
            pipe_.InitBuffer(outT_, 2U * crossOutputCapacity_ * sizeof(T) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        }
        pipe_.InitBuffer(expDenseFp_, 2U * crossOutputCapacity_ * sizeof(float) + FLOOR_MOD_UB_BANK_GUARD_BYTES);
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void InitCrossBuffers()
    {
        if (t_->crossD == 1U) {
            InitCrossUnitDBuffers();
        } else if (t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_COMPACT) {
            InitCrossCompactBuffers();
        } else {
            InitCrossPaddedBuffers();
        }
    }

    __aicore__ inline void CastUnitRowsToFp32(LocalTensor<float> dst, LocalTensor<T> src, uint32_t rows,
                                              uint32_t validElements)
    {
        constexpr uint32_t castMask = 256U / (sizeof(T) > sizeof(float) ? sizeof(T) : sizeof(float));
        const uint8_t dstRepStride = static_cast<uint8_t>(t_->crossUnitBAligned * sizeof(float) /
                                                          FLOOR_MOD_UB_BLOCK_BYTES);
        const uint8_t srcRepStride = static_cast<uint8_t>(t_->crossUnitBTAligned * sizeof(T) /
                                                          FLOOR_MOD_UB_BLOCK_BYTES);
        const UnaryRepeatParams params(1U, 1U, dstRepStride, srcRepStride);
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += 255U) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(rows - rowBase, 255U));
            for (uint32_t segment = 0U; segment < validElements; segment += castMask) {
                const uint32_t count = MinU32(validElements - segment, castMask);
                if constexpr (std::is_same_v<T, int64_t>) {
                    Cast(dst[rowBase * t_->crossUnitBAligned + segment],
                         src[rowBase * t_->crossUnitBTAligned + segment], AscendC::RoundMode::CAST_ROUND,
                         static_cast<uint64_t>(count), repeats, params);
                } else {
                    Cast(dst[rowBase * t_->crossUnitBAligned + segment],
                         src[rowBase * t_->crossUnitBTAligned + segment], AscendC::RoundMode::CAST_NONE,
                         static_cast<uint64_t>(count), repeats, params);
                }
            }
        }
    }

    __aicore__ inline void CastUnitRowsFromFp32(LocalTensor<T> dst, LocalTensor<float> src, uint32_t rows,
                                                uint32_t validElements)
    {
        constexpr uint32_t castMask = 256U / (sizeof(T) > sizeof(float) ? sizeof(T) : sizeof(float));
        const uint8_t dstRepStride = static_cast<uint8_t>(t_->crossUnitBTAligned * sizeof(T) /
                                                          FLOOR_MOD_UB_BLOCK_BYTES);
        const uint8_t srcRepStride = static_cast<uint8_t>(t_->crossUnitBAligned * sizeof(float) /
                                                          FLOOR_MOD_UB_BLOCK_BYTES);
        const UnaryRepeatParams params(1U, 1U, dstRepStride, srcRepStride);
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += 255U) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(rows - rowBase, 255U));
            for (uint32_t segment = 0U; segment < validElements; segment += castMask) {
                const uint32_t count = MinU32(validElements - segment, castMask);
                if constexpr (std::is_same_v<T, half>) {
                    Cast(dst[rowBase * t_->crossUnitBTAligned + segment],
                         src[rowBase * t_->crossUnitBAligned + segment], AscendC::RoundMode::CAST_NONE,
                         static_cast<uint64_t>(count), repeats, params);
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    Cast(dst[rowBase * t_->crossUnitBTAligned + segment],
                         src[rowBase * t_->crossUnitBAligned + segment], AscendC::RoundMode::CAST_TRUNC,
                         static_cast<uint64_t>(count), repeats, params);
                } else {
                    Cast(dst[rowBase * t_->crossUnitBTAligned + segment],
                         src[rowBase * t_->crossUnitBAligned + segment], AscendC::RoundMode::CAST_RINT,
                         static_cast<uint64_t>(count), repeats, params);
                }
            }
        }
    }

    __aicore__ inline void ComputeCompact(LocalTensor<float> a, LocalTensor<float> b, uint32_t count)
    {
        auto quotient = remFp_.Get<float>();
        auto floored = quotient[denseCapacity_];
        Div(quotient, a, b, count);
        Floor(floored, quotient, floorTmp_.Get<uint8_t>(), count);
        Mul(floored, floored, b, count);
        Sub(a, a, floored, count);
    }

    __aicore__ inline void ComputeCompactTo(LocalTensor<float> out, LocalTensor<float> a, LocalTensor<float> b,
                                            uint32_t count)
    {
        auto quotient = remFp_.Get<float>();
        auto floored = quotient[denseCapacity_];
        Div(quotient, a, b, count);
        Floor(floored, quotient, floorTmp_.Get<uint8_t>(), count);
        Mul(floored, floored, b, count);
        Sub(out, a, floored, count);
    }

    __aicore__ inline void CopyInCompact(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t count)
    {
        DataCopyExtParams params{1U, count * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void CopyOutCompact(GlobalTensor<T> dst, uint64_t offset, LocalTensor<T> src, uint32_t count)
    {
        DataCopyExtParams params{1U, count * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(dst[offset], src, params);
    }

    __aicore__ inline void WaitMte2(const event_t event)
    {
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void WaitVectorForStore(const event_t event)
    {
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
    }

    __aicore__ inline void WaitStore(const event_t event)
    {
        SetFlag<HardEvent::MTE3_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
    }

    __aicore__ inline void GetSmallDenseRange(uint64_t& offset, uint64_t& end) const
    {
        constexpr uint64_t blockElements = FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T);
        const uint64_t perCore = (t_->totalElements + t_->coreNum - 1U) / t_->coreNum;
        const uint64_t alignedPerCore = (perCore + blockElements - 1U) / blockElements * blockElements;
        const uint64_t coreOffset = static_cast<uint64_t>(GetBlockIdx()) * alignedPerCore;
        offset = coreOffset < t_->totalElements ? coreOffset : t_->totalElements;
        const uint64_t coreEnd = offset + alignedPerCore;
        end = coreEnd < t_->totalElements ? coreEnd : t_->totalElements;
    }

    __aicore__ inline void CopyInSmallDense(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t count)
    {
        constexpr uint32_t blockElements = FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T);
        if (offset % blockElements == 0U && count % blockElements == 0U) {
            DataCopy(dst, src[offset], count);
            return;
        }
        CopyInCompact(dst, src, offset, count);
    }

    __aicore__ inline void CopyOutSmallDense(GlobalTensor<T> dst, uint64_t offset, LocalTensor<T> src, uint32_t count)
    {
        constexpr uint32_t blockElements = FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T);
        if (offset % blockElements == 0U && count % blockElements == 0U) {
            DataCopy(dst[offset], src, count);
            return;
        }
        CopyOutCompact(dst, offset, src, count);
    }

    __aicore__ inline void ProcessSmallDenseTile(uint64_t offset, uint32_t count)
    {
        auto x1 = x1In_.Get<T>();
        auto x2 = x2In_.Get<T>();
        CopyInSmallDense(x1, x1Gm_, offset, count);
        CopyInSmallDense(x2, x2Gm_, offset, count);
        const event_t inputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        WaitMte2(inputReady);

        LocalTensor<T> out;
        if constexpr (std::is_same_v<T, float>) {
            ComputeCompact(x1.template ReinterpretCast<float>(), x2.template ReinterpretCast<float>(), count);
            out = x1;
        } else {
            auto a = aFp_.Get<float>();
            auto b = bFp_.Get<float>();
            CastToFp32(a, x1, count);
            CastToFp32(b, x2, count);
            ComputeCompact(a, b, count);
            out = x1;
            CastFromFp32(out, a, count);
        }
        const event_t outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        WaitVectorForStore(outputReady);
        CopyOutSmallDense(yGm_, offset, out, count);
    }

    __aicore__ inline void ProcessQueuedDenseTile(uint64_t offset, uint32_t count, event_t vectorToMte3,
                                                  event_t mte3ToVector)
    {
        auto x1 = denseX1Queue_.AllocTensor<T>();
        auto x2 = denseX2Queue_.AllocTensor<T>();
        CopyInCompact(x1, x1Gm_, offset, count);
        CopyInCompact(x2, x2Gm_, offset, count);
        denseX1Queue_.EnQue(x1);
        denseX2Queue_.EnQue(x2);
        x1 = denseX1Queue_.DeQue<T>();
        x2 = denseX2Queue_.DeQue<T>();
        auto out = denseYQueue_.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            ComputeCompactTo(out.template ReinterpretCast<float>(), x1.template ReinterpretCast<float>(),
                             x2.template ReinterpretCast<float>(), count);
            WaitVectorForStore(vectorToMte3);
        } else {
            auto a = aFp_.Get<float>();
            auto b = bFp_.Get<float>();
            CastToFp32(a, x1, count);
            CastToFp32(b, x2, count);
            ComputeCompact(a, b, count);
            CastFromFp32(out, a, count);
        }
        denseYQueue_.EnQue(out);
        auto ready = denseYQueue_.DeQue<T>();
        CopyOutCompact(yGm_, offset, ready, count);
        WaitStore(mte3ToVector);
        denseYQueue_.FreeTensor(ready);
        denseX1Queue_.FreeTensor(x1);
        denseX2Queue_.FreeTensor(x2);
    }

    __aicore__ inline void ProcessDense()
    {
        uint64_t offset = 0U;
        uint64_t end = 0U;
        if constexpr (SmallDense) {
            GetSmallDenseRange(offset, end);
            ProcessSmallDenseTile(offset, static_cast<uint32_t>(end - offset));
            return;
        }
        const uint64_t core = GetBlockIdx();
        const uint64_t base = t_->totalElements / t_->coreNum;
        const uint64_t extra = t_->totalElements % t_->coreNum;
        offset = base * core + (core < extra ? core : extra);
        end = offset + base + (core < extra ? 1U : 0U);

        const event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        while (offset < end) {
            const uint32_t count = MinU32(end - offset, t_->denseTile);
            ProcessQueuedDenseTile(offset, count, vectorToMte3, mte3ToVector);
            offset += count;
        }
    }

    __aicore__ inline void DecodePos(uint64_t pos, uint64_t& seedOffset, uint64_t& denseOffset) const
    {
        seedOffset = 0U;
        denseOffset = 0U;
        for (uint32_t i = t_->posDigits; i-- > 0U;) {
            const uint64_t coordinate = pos % t_->posExtent[i];
            pos /= t_->posExtent[i];
            seedOffset += coordinate * t_->seedStride[i];
            denseOffset += coordinate * t_->denseStride[i];
        }
    }

    __aicore__ inline void DecodeReuseTask(uint64_t task, uint32_t& part, uint32_t& e2Index, uint32_t& dIndex,
                                           uint64_t& pos) const
    {
        part = static_cast<uint32_t>(task % t_->rowPartitions);
        task /= t_->rowPartitions;

        e2Index = static_cast<uint32_t>(task % t_->e2TileCount);
        task /= t_->e2TileCount;
        dIndex = static_cast<uint32_t>(task % t_->dTileCount);
        pos = task / t_->dTileCount;
    }

    __aicore__ inline LocalTensor<float> PrepareReuseSeed(uint64_t seedOffset, uint32_t dReal, const event_t mte2)
    {
        auto seedT = seedIn_.Get<T>();
        CopyInCompact(seedT, t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_, seedOffset, dReal);
        WaitMte2(mte2);
        LocalTensor<float> seed;
        if constexpr (std::is_same_v<T, float>) {
            seed = seedT.template ReinterpretCast<float>();
        } else {
            seed = seedFp_.Get<float>();
            CastToFp32(seed, seedT, dReal);
            PipeBarrier<PIPE_V>();
        }
        return seed;
    }

    __aicore__ inline LocalTensor<float> BroadcastReuseSeed(LocalTensor<float> seed, uint32_t dReal, uint32_t e2Real)
    {
        auto expanded = expSeedFp_.Get<float>();
        if (t_->reuseLayout == FLOOR_MOD_REUSE_PACKED_BROADCAST) {
            // The public API uses Brcb + vector Copy + GatherMask for an
            // unaligned last dimension, producing a compact [D,E2] slab
            // without scalar GetValue or MTE UB2UB copies.
            const uint32_t srcShape[2] = {dReal, 1U};
            const uint32_t dstShape[2] = {dReal, e2Real};
            if (t_->broadcastTmpBytes == 0U) {
                Broadcast<float, 2, 1>(expanded, seed, dstShape, srcShape);
            } else {
                auto sharedTmp = floorTmp_.Get<uint8_t>();
                Broadcast<float, 2, 1>(expanded, seed, dstShape, srcShape, sharedTmp);
            }
            return expanded;
        }
        if (t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_MATERIALIZED) {
            const uint32_t paddedE2 = t_->maxFp32RowElems / t_->dTile;
            const uint32_t srcShape[2] = {dReal, 1U};
            const uint32_t dstShape[2] = {dReal, paddedE2};
            if (t_->broadcastTmpBytes == 0U) {
                BroadCast<float, 2, 1>(expanded, seed, dstShape, srcShape);
            } else {
                auto sharedTmp = floorTmp_.Get<uint8_t>();
                BroadCast<float, 2, 1>(expanded, seed, dstShape, srcShape, sharedTmp);
            }
            return expanded;
        }
        // Brcb turns eight consecutive seeds into eight FP32 DataBlocks.  The
        // arithmetic below reuses each block with srcBlkStride=0, so a seed is
        // materialized once instead of once per E2 element.
        uint32_t seedBase = 0U;
        while (seedBase < dReal) {
            const uint32_t batch = MinU32(static_cast<uint64_t>(dReal - seedBase), 2040U);
            const uint8_t repeats = static_cast<uint8_t>((batch + 7U) / 8U);
            Brcb(expanded[seedBase * 8U], seed[seedBase], repeats, {1U, 8U});
            seedBase += batch;
        }
        PipeBarrier<PIPE_V>();
        (void)e2Real;
        return expanded;
    }

    __aicore__ inline LocalTensor<float> LoadReuseSeed(uint64_t seedOffset, uint32_t dReal, uint32_t e2Real,
                                                       const event_t mte2)
    {
        auto seed = PrepareReuseSeed(seedOffset, dReal, mte2);
        return t_->mode == FLOOR_MOD_MODE_ROW_REUSE ? seed : BroadcastReuseSeed(seed, dReal, e2Real);
    }

    __aicore__ inline void CopyInPackedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows,
                                            uint32_t pattern, uint64_t gmRowElements)
    {
        const uint32_t blockBytes = pattern * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = t_->maxTRowElems * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(rows), blockBytes,
                                 static_cast<uint32_t>((gmRowElements - pattern) * sizeof(T)),
                                 localBlocks - paddedBlocks, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void CopyOutPackedRows(uint64_t offset, LocalTensor<T> src, uint32_t rows, uint32_t pattern,
                                             uint64_t gmRowElements)
    {
        const uint32_t blockBytes = pattern * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = t_->maxTRowElems * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(rows), blockBytes, localBlocks - paddedBlocks,
                                 static_cast<uint32_t>((gmRowElements - pattern) * sizeof(T)), 0U};
        DataCopyPad(yGm_[offset], src, params);
    }

    __aicore__ inline void CopyInExpandedRows(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows,
                                              uint32_t dReal, uint32_t e2Real, uint64_t gmRowElements)
    {
        const uint32_t e2TStride = t_->maxTRowElems / t_->dTile;
        const uint32_t blockBytes = e2Real * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = e2TStride * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(dReal), blockBytes,
                                 static_cast<uint32_t>((t_->e2 - e2Real) * sizeof(T)), localBlocks - paddedBlocks, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        for (uint32_t row = 0U; row < rows; ++row) {
            DataCopyPad(dst[row * t_->maxTRowElems], src[offset + row * gmRowElements], params, pad);
        }
    }

    __aicore__ inline void CopyOutExpandedRows(uint64_t offset, LocalTensor<T> src, uint32_t rows, uint32_t dReal,
                                               uint32_t e2Real, uint64_t gmRowElements)
    {
        if constexpr (sizeof(T) > sizeof(float)) {
            // GatherMask compacts 32-bit lanes and cannot preserve 64-bit
            // element boundaries.  Keep the padded rows in UB and let one
            // batched UB2GM command strip their aligned gaps instead.
            const uint32_t e2TStride = t_->maxTRowElems / t_->dTile;
            const uint32_t blockBytes = e2Real * static_cast<uint32_t>(sizeof(T));
            const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
            const uint32_t localBlocks = e2TStride * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
            const uint32_t gmStrideBytes = (t_->e2 - e2Real) * static_cast<uint32_t>(sizeof(T));
            for (uint32_t row = 0U; row < rows; ++row) {
                DataCopyExtParams params{static_cast<uint16_t>(dReal), blockBytes, localBlocks - paddedBlocks,
                                         gmStrideBytes, 0U};
                DataCopyPad(yGm_[offset + row * gmRowElements], src[row * t_->maxTRowElems], params);
            }
            return;
        }
        const uint32_t compactElements = dReal * e2Real;
        // CompactExpandedRows has removed the aligned E2 gaps. Each D slab is
        // now one contiguous GM interval, so issue one wide DMA instead of a
        // short-row blockCount transfer. Partial D tiles still retain the GM
        // gap between different E1 rows, hence one command per batched row.
        for (uint32_t row = 0U; row < rows; ++row) {
            DataCopyExtParams params{1U, compactElements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(yGm_[offset + row * gmRowElements], src[row * t_->maxTRowElems], params);
        }
    }

    __aicore__ inline void CompactExpandedRows(LocalTensor<T> data, uint32_t rows, uint32_t dReal, uint32_t e2Real)
    {
        if constexpr (sizeof(T) > sizeof(float)) {
            // The 64-bit path is compacted by batched UB2GM in
            // CopyOutExpandedRows; no UB layout conversion is needed.
            return;
        }
        const uint32_t e2TStride = t_->maxTRowElems / t_->dTile;
        if (dReal <= 1U || e2TStride == e2Real) {
            return;
        }
        const uint16_t srcRowBlocks = static_cast<uint16_t>(e2TStride * static_cast<uint32_t>(sizeof(T)) /
                                                            FLOOR_MOD_UB_BLOCK_BYTES);
        for (uint32_t row = 0U; row < rows; ++row) {
            const uint32_t base = row * t_->maxTRowElems;
            GatherMaskParams params{1U, static_cast<uint16_t>(dReal), srcRowBlocks, 0U};
            uint64_t gathered = 0U;
            GatherMask(data[base], data[base], static_cast<uint8_t>(7U), true, e2Real, params, gathered);
        }
    }

    __aicore__ inline void CopyInReuseRows(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows,
                                           uint32_t pattern, uint64_t gmRowElements)
    {
        const uint32_t blockBytes = pattern * static_cast<uint32_t>(sizeof(T));
        if (pattern == gmRowElements && rowCapacity_ == pattern) {
            DataCopyExtParams params{1U, rows * blockBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
            DataCopyPad(dst, src[offset], params, pad);
            return;
        }
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = rowCapacity_ * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(rows), blockBytes,
                                 static_cast<uint32_t>((gmRowElements - pattern) * sizeof(T)),
                                 localBlocks - paddedBlocks, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void CopyOutReuseRows(uint64_t offset, LocalTensor<T> src, uint32_t rows, uint32_t pattern,
                                            uint64_t gmRowElements)
    {
        const uint32_t blockBytes = pattern * static_cast<uint32_t>(sizeof(T));
        if (pattern == gmRowElements && rowCapacity_ == pattern) {
            DataCopyExtParams params{1U, rows * blockBytes, 0U, 0U, 0U};
            DataCopyPad(yGm_[offset], src, params);
            return;
        }
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = rowCapacity_ * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(rows), blockBytes, localBlocks - paddedBlocks,
                                 static_cast<uint32_t>((gmRowElements - pattern) * sizeof(T)), 0U};
        DataCopyPad(yGm_[offset], src, params);
    }

    __aicore__ inline void ComputeSingleReuseRow(LocalTensor<float> a, LocalTensor<float> rem, LocalTensor<float> seed,
                                                 uint32_t pattern)
    {
        if (t_->swapped != 0U) {
            Div(rem, seed, a, pattern);
        } else {
            Div(rem, a, seed, pattern);
        }
        Floor(rem, rem, floorTmp_.Get<uint8_t>(), pattern);
        if (t_->swapped != 0U) {
            Mul(rem, rem, a, pattern);
            Sub(a, seed, rem, pattern);
        } else {
            Mul(rem, rem, seed, pattern);
            Sub(a, a, rem, pattern);
        }
    }

    __aicore__ inline void ComputeRepeatedReuseRows(LocalTensor<float> a, LocalTensor<float> rem,
                                                    LocalTensor<float> seed, uint32_t rows, uint32_t pattern)
    {
        Duplicate(rem, 0.0f, rows * rowCapacity_);
        const uint8_t rowBlocks = static_cast<uint8_t>(rowCapacity_ / 8U);
        const BinaryRepeatParams normal(1U, 1U, 1U, rowBlocks, rowBlocks, 0U);
        const BinaryRepeatParams swappedDiv(1U, 1U, 1U, rowBlocks, 0U, rowBlocks);
        const BinaryRepeatParams bothRows(1U, 1U, 1U, rowBlocks, rowBlocks, rowBlocks);
        for (uint32_t segment = 0U; segment < pattern; segment += FLOOR_MOD_VECTOR_SEGMENT) {
            const uint32_t count = MinU32(pattern - segment, FLOOR_MOD_VECTOR_SEGMENT);
            if (t_->swapped != 0U) {
                Div(rem[segment], seed[segment], a[segment], static_cast<uint64_t>(count), rows, swappedDiv);
            } else {
                Div(rem[segment], a[segment], seed[segment], static_cast<uint64_t>(count), rows, normal);
            }
        }
        Floor(rem, rem, floorTmp_.Get<uint8_t>(), rows * rowCapacity_);
        for (uint32_t segment = 0U; segment < pattern; segment += FLOOR_MOD_VECTOR_SEGMENT) {
            const uint32_t count = MinU32(pattern - segment, FLOOR_MOD_VECTOR_SEGMENT);
            if (t_->swapped != 0U) {
                Mul(rem[segment], rem[segment], a[segment], static_cast<uint64_t>(count), rows, bothRows);
                Sub(a[segment], seed[segment], rem[segment], static_cast<uint64_t>(count), rows, swappedDiv);
            } else {
                Mul(rem[segment], rem[segment], seed[segment], static_cast<uint64_t>(count), rows, normal);
                Sub(a[segment], a[segment], rem[segment], static_cast<uint64_t>(count), rows, bothRows);
            }
        }
    }

    __aicore__ inline void ComputeReuseRows(LocalTensor<T> input, LocalTensor<T> output, LocalTensor<float> seed,
                                            uint32_t rows, uint32_t pattern)
    {
        LocalTensor<float> a;
        if constexpr (std::is_same_v<T, float>) {
            a = input.template ReinterpretCast<float>();
        } else {
            a = aFp_.Get<float>();
            CastToFp32(a, input, rows == 1U ? pattern : rows * rowCapacity_);
        }
        auto rem = remFp_.Get<float>();

        if (rows == 1U) {
            ComputeSingleReuseRow(a, rem, seed, pattern);
        } else {
            ComputeRepeatedReuseRows(a, rem, seed, rows, pattern);
        }
        if constexpr (!std::is_same_v<T, float>) {
            CastFromFp32(output, a, rows == 1U ? pattern : rows * rowCapacity_);
        }
    }

    template <bool Remainder>
    __aicore__ inline void ComputeExpandedStage(LocalTensor<float> quotient, LocalTensor<float> dense,
                                                LocalTensor<float> seed, uint32_t rows, uint32_t dReal, uint32_t e2Real)
    {
        const uint32_t stride = t_->maxFp32RowElems / t_->dTile;
        const uint8_t blocks = static_cast<uint8_t>(stride / FLOOR_MOD_FP32_BLOCK_ELEMS);
        const BinaryRepeatParams normal(1U, 1U, 0U, blocks, blocks, 1U);
        const BinaryRepeatParams swapped(1U, 0U, 1U, blocks, 1U, blocks);
        const BinaryRepeatParams bothRows(1U, 1U, 1U, blocks, blocks, blocks);
        for (uint32_t row = 0U; row < rows; ++row) {
            for (uint32_t base = 0U; base < dReal; base += FLOOR_MOD_MAX_VECTOR_REPEATS) {
                const uint8_t repeats = static_cast<uint8_t>(MinU32(dReal - base, FLOOR_MOD_MAX_VECTOR_REPEATS));
                const uint32_t denseBase = row * rowCapacity_ + base * stride;
                for (uint32_t segment = 0U; segment < e2Real; segment += FLOOR_MOD_VECTOR_SEGMENT) {
                    ComputeBroadcastStage<Remainder>(quotient[denseBase + segment], dense[denseBase + segment],
                                                     seed[base * FLOOR_MOD_FP32_BLOCK_ELEMS],
                                                     MinU32(e2Real - segment, FLOOR_MOD_VECTOR_SEGMENT), repeats,
                                                     normal, swapped, bothRows, t_->swapped != 0U);
                }
            }
        }
    }

    __aicore__ inline void ComputeExpandedRows(LocalTensor<T> input, LocalTensor<T> output, LocalTensor<float> seed,
                                               uint32_t rows, uint32_t dReal, uint32_t e2Real)
    {
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = input.template ReinterpretCast<float>();
        } else {
            dense = aFp_.Get<float>();
            CastToFp32(dense, input, rows * rowCapacity_);
        }
        auto quotient = remFp_.Get<float>();
        Duplicate(quotient, 0.0f, rows * rowCapacity_);
        ComputeExpandedStage<false>(quotient, dense, seed, rows, dReal, e2Real);
        Floor(quotient, quotient, floorTmp_.Get<uint8_t>(), rows * rowCapacity_);
        ComputeExpandedStage<true>(quotient, dense, seed, rows, dReal, e2Real);
        if constexpr (!std::is_same_v<T, float>) {
            CastFromFp32(output, dense, rows * rowCapacity_);
        }
    }

    __aicore__ inline void ComputePackedRows(LocalTensor<T> input, LocalTensor<T> output, LocalTensor<float> seed,
                                             uint32_t rows, uint32_t pattern)
    {
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = input.template ReinterpretCast<float>();
        } else {
            dense = aFp_.Get<float>();
            CastToFp32(dense, input, rows * rowCapacity_);
            PipeBarrier<PIPE_V>();
        }
        auto quotient = remFp_.Get<float>();
        for (uint32_t row = 0U; row < rows; ++row) {
            const uint32_t base = row * rowCapacity_;
            if (t_->swapped != 0U) {
                Div(quotient[base], seed, dense[base], pattern);
            } else {
                Div(quotient[base], dense[base], seed, pattern);
            }
            Floor(quotient[base], quotient[base], floorTmp_.Get<uint8_t>(), pattern);
            if (t_->swapped != 0U) {
                Mul(quotient[base], quotient[base], dense[base], pattern);
                Sub(dense[base], seed, quotient[base], pattern);
            } else {
                Mul(quotient[base], quotient[base], seed, pattern);
                Sub(dense[base], dense[base], quotient[base], pattern);
            }
        }
        if constexpr (!std::is_same_v<T, float>) {
            CastFromFp32(output, dense, rows * rowCapacity_);
        }
    }

    __aicore__ inline void ComputeMaterializedRows(LocalTensor<T> input, LocalTensor<T> output, LocalTensor<float> seed,
                                                   uint32_t rows)
    {
        const uint32_t count = rows * rowCapacity_;
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = input.template ReinterpretCast<float>();
        } else {
            dense = aFp_.Get<float>();
            CastToFp32(dense, input, count);
        }
        auto quotient = remFp_.Get<float>();
        ComputeFloorRemainder(quotient, dense, seed, floorTmp_.Get<uint8_t>(), count, t_->swapped != 0U);
        if constexpr (!std::is_same_v<T, float>) {
            CastFromFp32(output, dense, count);
        }
    }

    __aicore__ inline void LoadCompactPosInput(uint64_t posBase, uint32_t posReal, event_t inputReady)
    {
        auto seedT = seedIn_.Get<T>();
        auto denseT = denseIn_.Get<T>();
        GlobalTensor<T> seedGm = t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_;
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;
        const uint32_t blockBytes = static_cast<uint32_t>(t_->D) * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = posRowCapacity_ * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams seedParams{static_cast<uint16_t>(posReal), blockBytes, 0U, localBlocks - paddedBlocks, 0U};
        const uint32_t logicalCount = posReal * t_->e1 * static_cast<uint32_t>(t_->D);
        DataCopyExtParams denseParams{1U, logicalCount * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(seedT, seedGm[posBase * t_->D], seedParams, pad);
        DataCopyPad(denseT, denseGm[posBase * t_->e1 * t_->D], denseParams, pad);
        SetFlag<HardEvent::MTE2_V>(inputReady);
    }

    __aicore__ inline LocalTensor<float> ExpandCompactPosSeed(LocalTensor<T> seedT, uint32_t posReal)
    {
        LocalTensor<float> seed;
        if constexpr (std::is_same_v<T, float>) {
            seed = seedT.template ReinterpretCast<float>();
        } else {
            seed = seedFp_.Get<float>();
            CastToFp32(seed, seedT, posReal * posRowCapacity_);
            PipeBarrier<PIPE_V>();
        }
        auto expanded = expSeedFp_.Get<float>();
        const uint32_t srcShape[2] = {1U, posRowCapacity_};
        const uint32_t dstShape[2] = {t_->e1, posRowCapacity_};
        for (uint32_t pos = 0U; pos < posReal; ++pos) {
            const uint32_t dstOffset = pos * t_->e1 * posRowCapacity_;
            if (t_->broadcastTmpBytes == 0U) {
                Broadcast<float, 2, 0>(expanded[dstOffset], seed[pos * posRowCapacity_], dstShape, srcShape);
            } else {
                auto broadcastTmp = floorTmp_.Get<uint8_t>();
                Broadcast<float, 2, 0>(expanded[dstOffset], seed[pos * posRowCapacity_], dstShape, srcShape,
                                       broadcastTmp);
            }
        }
        const uint32_t rows = posReal * t_->e1;
        const uint16_t rowSpanBlocks = static_cast<uint16_t>(posRowCapacity_ / FLOOR_MOD_FP32_BLOCK_ELEMS);
        GatherMaskParams params{1U, static_cast<uint16_t>(rows), rowSpanBlocks, 0U};
        uint64_t gathered = 0U;
        GatherMask(expanded, expanded, static_cast<uint8_t>(7U), true, static_cast<uint32_t>(t_->D), params, gathered);
        PipeBarrier<PIPE_V>();
        return expanded;
    }

    __aicore__ inline void ComputeCompactPos(LocalTensor<float> seed, LocalTensor<T> denseT, uint32_t logicalCount)
    {
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = denseT.template ReinterpretCast<float>();
        } else {
            dense = aFp_.Get<float>();
            CastToFp32(dense, denseT, logicalCount);
            PipeBarrier<PIPE_V>();
        }

        auto quotient = remFp_.Get<float>();
        if (t_->swapped != 0U) {
            Div(quotient, seed, dense, logicalCount);
        } else {
            Div(quotient, dense, seed, logicalCount);
        }
        Floor(quotient, quotient, floorTmp_.Get<uint8_t>(), logicalCount);
        if (t_->swapped != 0U) {
            Mul(quotient, quotient, dense, logicalCount);
            Sub(dense, seed, quotient, logicalCount);
        } else {
            Mul(quotient, quotient, seed, logicalCount);
            Sub(dense, dense, quotient, logicalCount);
        }
        if constexpr (!std::is_same_v<T, float>) {
            CastFromFp32(denseT, dense, logicalCount);
        }
    }

    __aicore__ inline void ProcessCompactPosBroadcast()
    {
        const event_t inputReady = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        const event_t outputReady = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

        const uint64_t block = GetBlockIdx();
        const uint64_t base = t_->posTotal / t_->coreNum;
        const uint64_t extra = t_->posTotal % t_->coreNum;
        const uint64_t posBase = block * base + (block < extra ? block : extra);
        const uint32_t posReal = static_cast<uint32_t>(base + (block < extra ? 1U : 0U));
        auto seedT = seedIn_.Get<T>();
        auto denseT = denseIn_.Get<T>();

        LoadCompactPosInput(posBase, posReal, inputReady);
        WaitFlag<HardEvent::MTE2_V>(inputReady);
        auto expandedSeed = ExpandCompactPosSeed(seedT, posReal);
        const uint32_t logicalCount = posReal * t_->e1 * static_cast<uint32_t>(t_->D);
        ComputeCompactPos(expandedSeed, denseT, logicalCount);
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);
        DataCopyExtParams params{1U, logicalCount * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(yGm_[posBase * t_->e1 * t_->D], denseT, params);

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outputReady);
    }

    __aicore__ inline void ProcessReuseBatch(uint64_t pos, uint64_t row, uint32_t rows, uint64_t dBase, uint64_t e2Base,
                                             uint32_t dReal, uint32_t e2Real, uint32_t pattern, bool expanded,
                                             bool packed, bool materialized, uint64_t densePosOffset, event_t mte2,
                                             event_t vMte3, const event_t slotFree[2], LocalTensor<float> seed,
                                             uint32_t& iteration)
    {
        const uint64_t gmRowElements = t_->D * t_->e2;
        const uint32_t slot = iteration & 1U;
        WaitFlag<HardEvent::MTE3_MTE2>(slotFree[slot]);
        auto input = denseIn_.Get<T>()[slot * reuseTSlotElems_];
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;
        const uint64_t inOffset = densePosOffset + row * gmRowElements + dBase * t_->e2 + e2Base;
        const uint64_t outOffset = (pos * t_->e1 + row) * gmRowElements + dBase * t_->e2 + e2Base;
        if (packed) {
            CopyInPackedRows(input, denseGm, inOffset, rows, pattern, gmRowElements);
        } else if (expanded) {
            CopyInExpandedRows(input, denseGm, inOffset, rows, dReal, e2Real, gmRowElements);
        } else {
            CopyInReuseRows(input, denseGm, inOffset, rows, pattern, gmRowElements);
        }
        WaitMte2(mte2);
        LocalTensor<T> output = input;
        if (packed) {
            ComputePackedRows(input, output, seed, rows, pattern);
        } else if (materialized) {
            ComputeMaterializedRows(input, output, seed, rows);
        } else if (expanded) {
            ComputeExpandedRows(input, output, seed, rows, dReal, e2Real);
        } else {
            ComputeReuseRows(input, output, seed, rows, pattern);
        }
        if (expanded && !packed) {
            CompactExpandedRows(output, rows, dReal, e2Real);
        }
        WaitVectorForStore(vMte3);
        if (packed) {
            CopyOutPackedRows(outOffset, output, rows, pattern, gmRowElements);
        } else if (expanded) {
            CopyOutExpandedRows(outOffset, output, rows, dReal, e2Real, gmRowElements);
        } else {
            CopyOutReuseRows(outOffset, output, rows, pattern, gmRowElements);
        }
        SetFlag<HardEvent::MTE3_MTE2>(slotFree[slot]);
        ++iteration;
    }

    __aicore__ inline void ProcessReuseRows(uint64_t pos, uint32_t dIndex, uint32_t dEnd, uint32_t e2Index,
                                            uint64_t seedPosOffset, uint64_t densePosOffset, uint64_t rowBegin,
                                            uint64_t rowEnd, bool expanded, bool packed, bool materialized,
                                            event_t mte2, event_t vMte3, event_t seedFree, const event_t slotFree[2],
                                            bool& seedUsed, uint32_t& iteration)
    {
        const uint64_t gmRowElements = t_->D * t_->e2;
        for (; dIndex < dEnd; ++dIndex) {
            const uint64_t dBase = static_cast<uint64_t>(dIndex) * t_->dTile;
            const uint64_t e2Base = static_cast<uint64_t>(e2Index) * t_->e2Tile;
            const uint32_t dReal = MinU32(t_->D - dBase, t_->dTile);
            const uint32_t e2Real = MinU32(static_cast<uint64_t>(t_->e2) - e2Base, t_->e2Tile);
            const uint32_t pattern = dReal * e2Real;
            if (seedUsed) {
                WaitFlag<HardEvent::V_MTE2>(seedFree);
            }
            auto seed = LoadReuseSeed(seedPosOffset + dBase, dReal, e2Real, mte2);
            uint64_t row = rowBegin;
            while (row < rowEnd) {
                const uint32_t rows = MinU32(rowEnd - row, t_->batchRows);
                ProcessReuseBatch(pos, row, rows, dBase, e2Base, dReal, e2Real, pattern, expanded, packed, materialized,
                                  densePosOffset, mte2, vMte3, slotFree, seed, iteration);
                row += rows;
            }
            SetFlag<HardEvent::V_MTE2>(seedFree);
            seedUsed = true;
        }
    }

    __aicore__ inline void ProcessReuseTasks(uint64_t taskBegin, uint64_t taskEnd, event_t mte2, event_t vMte3,
                                             event_t seedFree, const event_t slotFree[2], bool& seedUsed,
                                             uint32_t& iteration)
    {
        const bool expanded = t_->mode == FLOOR_MOD_MODE_EXPAND_REUSE;
        const bool packed = expanded && t_->reuseLayout == FLOOR_MOD_REUSE_PACKED_BROADCAST;
        const bool materialized = expanded && t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_MATERIALIZED;
        const bool rowOwned = !expanded && t_->reuseSchedule != 0U;
        for (uint64_t task = taskBegin; task < taskEnd; ++task) {
            uint32_t part = 0U;
            uint32_t e2Index = 0U;
            uint32_t dIndex = 0U;
            uint64_t pos = 0U;
            if (rowOwned) {
                part = static_cast<uint32_t>(task % t_->rowPartitions);
                pos = task / t_->rowPartitions;
            } else {
                DecodeReuseTask(task, part, e2Index, dIndex, pos);
            }
            const uint64_t rowsBase = t_->e1 / t_->rowPartitions;
            const uint64_t rowsExtra = t_->e1 % t_->rowPartitions;
            const uint64_t rowBegin = rowsBase * part + (part < rowsExtra ? part : rowsExtra);
            const uint64_t rowEnd = rowBegin + rowsBase + (part < rowsExtra ? 1U : 0U);
            uint64_t seedPosOffset = 0U;
            uint64_t densePosOffset = 0U;
            DecodePos(pos, seedPosOffset, densePosOffset);
            const uint32_t dEnd = rowOwned ? t_->dTileCount : dIndex + 1U;
            ProcessReuseRows(pos, dIndex, dEnd, e2Index, seedPosOffset, densePosOffset, rowBegin, rowEnd, expanded,
                             packed, materialized, mte2, vMte3, seedFree, slotFree, seedUsed, iteration);
        }
    }

    __aicore__ inline void ProcessReuse()
    {
        const event_t mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        const event_t vMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        const event_t seedFree = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        const event_t slotFree[2] = {static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>()),
                                     static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>())};
        bool seedUsed = false;
        uint32_t iteration = 0U;
        SetFlag<HardEvent::MTE3_MTE2>(slotFree[0]);
        SetFlag<HardEvent::MTE3_MTE2>(slotFree[1]);
        uint64_t taskBegin = 0U;
        uint64_t taskEnd = 0U;
        GetTaskRange(t_->totalTasks, taskBegin, taskEnd);
        ProcessReuseTasks(taskBegin, taskEnd, mte2, vMte3, seedFree, slotFree, seedUsed, iteration);
        if (seedUsed) {
            WaitFlag<HardEvent::V_MTE2>(seedFree);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(slotFree[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(slotFree[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(seedFree);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(slotFree[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(slotFree[1]);
    }

    __aicore__ inline void DecodeCrossTask(uint64_t task, uint32_t& bIndex, uint32_t& aIndex,
                                           uint32_t& outerIndex) const
    {
        bIndex = static_cast<uint32_t>(task % t_->crossBTileCount);
        task /= t_->crossBTileCount;
        aIndex = static_cast<uint32_t>(task % t_->crossATileCount);
        outerIndex = static_cast<uint32_t>(task / t_->crossATileCount);
    }

    struct CrossTileInfo {
        uint32_t outerBase;
        uint32_t outerReal;
        uint32_t aBase;
        uint32_t bBase;
        uint32_t aReal;
        uint32_t bActual;
    };

    __aicore__ inline CrossTileInfo DecodeCrossTileInfo(uint64_t task) const
    {
        uint32_t bIndex = 0U;
        uint32_t aIndex = 0U;
        uint32_t outerIndex = 0U;
        CrossTileInfo info{};
        DecodeCrossTile(task, bIndex, aIndex, outerIndex, info.outerBase, info.outerReal, info.aBase, info.bBase,
                        info.aReal, info.bActual);
        return info;
    }

    __aicore__ inline void DecodeCrossTile(uint64_t task, uint32_t& bIndex, uint32_t& aIndex, uint32_t& outerIndex,
                                           uint32_t& outerBase, uint32_t& outerReal, uint32_t& aBase, uint32_t& bBase,
                                           uint32_t& aReal, uint32_t& bActual) const
    {
        DecodeCrossTask(task, bIndex, aIndex, outerIndex);
        outerBase = outerIndex * t_->crossOuterTile;
        outerReal = MinU32(t_->crossOuter - outerBase, t_->crossOuterTile);
        aBase = aIndex * t_->crossATile;
        bBase = bIndex * t_->crossBTile;
        aReal = MinU32(t_->crossA - aBase, t_->crossATile);
        bActual = MinU32(t_->crossB - bBase, t_->crossBTile);
    }

    __aicore__ inline void CopyInCrossRows(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows,
                                           uint32_t localRowElements)
    {
        const uint32_t blockBytes = t_->crossD * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = localRowElements * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyPadExtParams<T> pad{true, 0U, static_cast<uint8_t>(localRowElements - t_->crossD), static_cast<T>(0)};
        while (rows > 0U) {
            const uint32_t batch = rows > 4095U ? 4095U : rows;
            DataCopyExtParams params{static_cast<uint16_t>(batch), blockBytes, 0U, localBlocks - paddedBlocks, 0U};
            DataCopyPad(dst, src[offset], params, pad);
            dst = dst[batch * localRowElements];
            offset += static_cast<uint64_t>(batch) * t_->crossD;
            rows -= batch;
        }
    }

    __aicore__ inline void CopyInCrossCompactDense(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset,
                                                   uint32_t rows, uint32_t bActual)
    {
        const uint32_t slabElements = bActual * t_->crossD;
        const uint32_t blockBytes = slabElements * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = t_->crossUnitBTAligned * static_cast<uint32_t>(sizeof(T)) /
                                     FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t gmStrideBytes = (t_->crossB - bActual) * t_->crossD * static_cast<uint32_t>(sizeof(T));
        DataCopyPadExtParams<T> pad{true, 0U, 0U, static_cast<T>(0)};
        while (rows > 0U) {
            const uint32_t batch = rows > 4095U ? 4095U : rows;
            DataCopyExtParams params{static_cast<uint16_t>(batch), blockBytes, gmStrideBytes,
                                     localBlocks - paddedBlocks, 0U};
            DataCopyPad(dst, src[offset], params, pad);
            dst = dst[batch * t_->crossUnitBTAligned];
            offset += static_cast<uint64_t>(batch) * t_->crossB * t_->crossD;
            rows -= batch;
        }
    }

    __aicore__ inline void CopyOutCrossRows(uint64_t offset, LocalTensor<T> src, uint32_t rows)
    {
        const uint32_t blockBytes = t_->crossD * static_cast<uint32_t>(sizeof(T));
        const uint32_t alignedBlockBytes = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES *
                                           FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localRowBytes = t_->crossDAligned * static_cast<uint32_t>(sizeof(T));
        const uint32_t srcStrideBlocks = (localRowBytes - alignedBlockBytes) / FLOOR_MOD_UB_BLOCK_BYTES;
        while (rows > 0U) {
            const uint32_t batch = rows > 4095U ? 4095U : rows;
            DataCopyExtParams params{static_cast<uint16_t>(batch), blockBytes, srcStrideBlocks, 0U, 0U};
            DataCopyPad(yGm_[offset], src, params);
            src = src[batch * t_->crossDAligned];
            offset += static_cast<uint64_t>(batch) * t_->crossD;
            rows -= batch;
        }
    }

    __aicore__ inline void CopyOutCrossPadded(uint64_t offset, LocalTensor<T> src, uint32_t bActual)
    {
        if (bActual == t_->crossB) {
            CopyOutCrossRows(offset, src, t_->crossM * bActual);
            return;
        }
        for (uint32_t m = 0U; m < t_->crossM; ++m) {
            CopyOutCrossRows(offset + static_cast<uint64_t>(m) * t_->crossB * t_->crossD,
                             src[m * bActual * t_->crossDAligned], bActual);
        }
    }

    __aicore__ inline void CopyInCrossUnitD(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows,
                                            uint32_t bActual)
    {
        const uint32_t blockBytes = bActual * static_cast<uint32_t>(sizeof(T));
        const uint32_t localRowBytes = t_->crossUnitBTAligned * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localBlocks = localRowBytes / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t gmStrideBytes = (t_->crossB - bActual) * static_cast<uint32_t>(sizeof(T));
        // DataCopyPad padding is limited to one DataBlock. Keep the
        // tile-sized UB row pitch with dstStride instead of explicit padding.
        DataCopyPadExtParams<T> pad{true, 0U, 0U, static_cast<T>(0)};
        while (rows > 0U) {
            const uint32_t batch = rows > 4095U ? 4095U : rows;
            DataCopyExtParams params{static_cast<uint16_t>(batch), blockBytes, gmStrideBytes,
                                     localBlocks - paddedBlocks, 0U};
            DataCopyPad(dst, src[offset], params, pad);
            dst = dst[batch * t_->crossUnitBTAligned];
            offset += static_cast<uint64_t>(batch) * t_->crossB;
            rows -= batch;
        }
    }

    __aicore__ inline void CopyOutCrossUnitD(uint64_t offset, LocalTensor<T> src, uint32_t rows, uint32_t bActual)
    {
        const uint32_t blockBytes = bActual * static_cast<uint32_t>(sizeof(T));
        const uint32_t alignedBlockBytes = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES *
                                           FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localRowBytes = t_->crossUnitBTAligned * static_cast<uint32_t>(sizeof(T));
        const uint32_t srcStrideBlocks = (localRowBytes - alignedBlockBytes) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t gmStrideBytes = (t_->crossB - bActual) * static_cast<uint32_t>(sizeof(T));
        while (rows > 0U) {
            const uint32_t batch = rows > 4095U ? 4095U : rows;
            DataCopyExtParams params{static_cast<uint16_t>(batch), blockBytes, srcStrideBlocks, gmStrideBytes, 0U};
            DataCopyPad(yGm_[offset], src, params);
            src = src[batch * t_->crossUnitBTAligned];
            offset += static_cast<uint64_t>(batch) * t_->crossB;
            rows -= batch;
        }
    }

    __aicore__ inline void GatherCrossUnitDOutput(LocalTensor<float> output, uint32_t rows, uint32_t bActual)
    {
        GatherMaskParams params{1U, static_cast<uint16_t>(rows),
                                static_cast<uint16_t>(t_->crossUnitBAligned / FLOOR_MOD_FP32_BLOCK_ELEMS), 0U};
        uint64_t gathered = 0U;
        GatherMask(output, output, static_cast<uint8_t>(7U), true, bActual, params, gathered);
    }

    __aicore__ inline void CopyOutCrossUnitDCompact(uint64_t offset, LocalTensor<T> src, uint32_t outerReal,
                                                    uint32_t aReal, uint32_t bActual)
    {
        if (bActual == t_->crossB && aReal == t_->crossA) {
            CopyOutCompact(yGm_, offset, src, outerReal * aReal * t_->crossM * bActual);
            return;
        }
        if (bActual == t_->crossB) {
            const uint32_t outerElements = aReal * t_->crossM * bActual;
            for (uint32_t outer = 0U; outer < outerReal; ++outer) {
                const uint64_t outerOffset = offset +
                                             static_cast<uint64_t>(outer) * t_->crossA * t_->crossM * t_->crossB;
                CopyOutCompact(yGm_, outerOffset, src[outer * outerElements], outerElements);
            }
            return;
        }
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            for (uint32_t a = 0U; a < aReal; ++a) {
                for (uint32_t m = 0U; m < t_->crossM; ++m) {
                    const uint32_t row = (outer * aReal + a) * t_->crossM + m;
                    const uint64_t rowOffset = offset +
                                               ((static_cast<uint64_t>(outer) * t_->crossA + a) * t_->crossM + m) *
                                                   t_->crossB;
                    CopyOutCompact(yGm_, rowOffset, src[row * bActual], bActual);
                }
            }
        }
    }

    __aicore__ inline void ComputeCrossUnitDQuotient(LocalTensor<float> quotient, LocalTensor<float> dense,
                                                     LocalTensor<float> seedBlocks, uint32_t bActual,
                                                     const BinaryRepeatParams& normal,
                                                     const BinaryRepeatParams& swappedDiv)
    {
        const uint32_t rowElems = t_->crossUnitBAligned;
        for (uint32_t mBase = 0U; mBase < t_->crossM; mBase += 255U) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(t_->crossM - mBase, 255U));
            const uint32_t denseBase = mBase * rowElems;
            const uint32_t seedBase = mBase * 8U;
            for (uint32_t segment = 0U; segment < bActual; segment += FLOOR_MOD_VECTOR_SEGMENT) {
                const uint32_t count = MinU32(bActual - segment, FLOOR_MOD_VECTOR_SEGMENT);
                if (t_->swapped != 0U) {
                    Div(quotient[denseBase + segment], seedBlocks[seedBase], dense[denseBase + segment],
                        static_cast<uint64_t>(count), repeats, swappedDiv);
                } else {
                    Div(quotient[denseBase + segment], dense[denseBase + segment], seedBlocks[seedBase],
                        static_cast<uint64_t>(count), repeats, normal);
                }
            }
        }
    }

    __aicore__ inline void FinishCrossUnitDRemainder(LocalTensor<float> output, LocalTensor<float> dense,
                                                     LocalTensor<float> seedBlocks, uint32_t bActual,
                                                     const BinaryRepeatParams& normal,
                                                     const BinaryRepeatParams& swappedDiv,
                                                     const BinaryRepeatParams& bothRows)
    {
        const uint32_t rowElems = t_->crossUnitBAligned;
        for (uint32_t mBase = 0U; mBase < t_->crossM; mBase += 255U) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(t_->crossM - mBase, 255U));
            const uint32_t denseBase = mBase * rowElems;
            const uint32_t seedBase = mBase * 8U;
            for (uint32_t segment = 0U; segment < bActual; segment += FLOOR_MOD_VECTOR_SEGMENT) {
                const uint32_t count = MinU32(bActual - segment, FLOOR_MOD_VECTOR_SEGMENT);
                if (t_->swapped != 0U) {
                    Mul(output[denseBase + segment], output[denseBase + segment], dense[denseBase + segment],
                        static_cast<uint64_t>(count), repeats, bothRows);
                    Sub(output[denseBase + segment], seedBlocks[seedBase], output[denseBase + segment],
                        static_cast<uint64_t>(count), repeats, swappedDiv);
                } else {
                    Mul(output[denseBase + segment], output[denseBase + segment], seedBlocks[seedBase],
                        static_cast<uint64_t>(count), repeats, normal);
                    Sub(output[denseBase + segment], dense[denseBase + segment], output[denseBase + segment],
                        static_cast<uint64_t>(count), repeats, bothRows);
                }
            }
        }
    }

    __aicore__ inline void ComputeCrossUnitD(LocalTensor<float> output, LocalTensor<float> dense,
                                             LocalTensor<float> seedBlocks, uint32_t bActual)
    {
        auto quotient = remFp_.Get<float>();
        const uint32_t rowElems = t_->crossUnitBAligned;
        const uint8_t rowBlocks = static_cast<uint8_t>(rowElems / 8U);
        const BinaryRepeatParams normal(1U, 1U, 0U, rowBlocks, rowBlocks, 1U);
        const BinaryRepeatParams swappedDiv(1U, 0U, 1U, rowBlocks, 1U, rowBlocks);
        const BinaryRepeatParams bothRows(1U, 1U, 1U, rowBlocks, rowBlocks, rowBlocks);
        ComputeCrossUnitDQuotient(quotient, dense, seedBlocks, bActual, normal, swappedDiv);
        Floor(output, quotient, floorTmp_.Get<uint8_t>(), t_->crossM * rowElems);
        FinishCrossUnitDRemainder(output, dense, seedBlocks, bActual, normal, swappedDiv, bothRows);
    }

    __aicore__ inline void ComputeCrossUnitDSegment(
        LocalTensor<float> quotient, LocalTensor<float> output, LocalTensor<float> dense, LocalTensor<float> seedBlocks,
        uint32_t outputBase, uint32_t denseBase, uint32_t seedBase, uint32_t segment, uint32_t count, uint8_t repeats,
        const BinaryRepeatParams& normalDiv, const BinaryRepeatParams& swappedDiv, const BinaryRepeatParams& normalMul,
        const BinaryRepeatParams& swappedMul, const BinaryRepeatParams& normalSub, const BinaryRepeatParams& swappedSub,
        const BinaryRepeatParams& normalFma, const BinaryRepeatParams& swappedFma, uint32_t operation)
    {
        if (operation == 0U) {
            if (t_->swapped != 0U) {
                Div(quotient[outputBase + segment], seedBlocks[seedBase], dense[denseBase + segment], count, repeats,
                    swappedDiv);
            } else {
                Div(quotient[outputBase + segment], dense[denseBase + segment], seedBlocks[seedBase], count, repeats,
                    normalDiv);
            }
            return;
        }
        if (operation == 1U) {
            if (t_->swapped != 0U) {
                FusedMulAdd(output[outputBase + segment], dense[denseBase + segment], seedBlocks[seedBase], count,
                            repeats, swappedFma);
            } else {
                FusedMulAdd(output[outputBase + segment], seedBlocks[seedBase], dense[denseBase + segment], count,
                            repeats, normalFma);
            }
            return;
        }
        if (t_->swapped != 0U) {
            Mul(output[outputBase + segment], output[outputBase + segment], dense[denseBase + segment], count, repeats,
                swappedMul);
            Sub(output[outputBase + segment], seedBlocks[seedBase], output[outputBase + segment], count, repeats,
                swappedSub);
        } else {
            Mul(output[outputBase + segment], output[outputBase + segment], seedBlocks[seedBase], count, repeats,
                normalMul);
            Sub(output[outputBase + segment], dense[denseBase + segment], output[outputBase + segment], count, repeats,
                normalSub);
        }
    }

    __aicore__ inline void ComputeCrossUnitDLoop(
        LocalTensor<float> quotient, LocalTensor<float> output, LocalTensor<float> dense, LocalTensor<float> seedBlocks,
        uint32_t outerReal, uint32_t aReal, uint32_t bActual, const BinaryRepeatParams& normalDiv,
        const BinaryRepeatParams& swappedDiv, const BinaryRepeatParams& normalMul, const BinaryRepeatParams& swappedMul,
        const BinaryRepeatParams& normalSub, const BinaryRepeatParams& swappedSub, const BinaryRepeatParams& normalFma,
        const BinaryRepeatParams& swappedFma, uint32_t operation)
    {
        const uint32_t rowElems = t_->crossUnitBAligned;
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            for (uint32_t m = 0U; m < t_->crossM; ++m) {
                const uint32_t denseBase = (outer * t_->crossM + m) * rowElems;
                for (uint32_t aBase = 0U; aBase < aReal; aBase += FLOOR_MOD_MAX_VECTOR_REPEATS) {
                    const uint8_t repeats = static_cast<uint8_t>(MinU32(aReal - aBase, FLOOR_MOD_MAX_VECTOR_REPEATS));
                    const uint32_t matrix = outer * aReal + aBase;
                    const uint32_t outputBase = (matrix * t_->crossM + m) * rowElems;
                    const uint32_t seedBase = (matrix * t_->crossM + m) * FLOOR_MOD_FP32_BLOCK_ELEMS;
                    for (uint32_t segment = 0U; segment < bActual; segment += FLOOR_MOD_VECTOR_SEGMENT) {
                        const uint32_t count = MinU32(bActual - segment, FLOOR_MOD_VECTOR_SEGMENT);
                        ComputeCrossUnitDSegment(quotient, output, dense, seedBlocks, outputBase, denseBase, seedBase,
                                                 segment, count, repeats, normalDiv, swappedDiv, normalMul, swappedMul,
                                                 normalSub, swappedSub, normalFma, swappedFma, operation);
                    }
                }
            }
        }
    }

    __aicore__ inline void ComputeCrossUnitDFallback(LocalTensor<float> output, LocalTensor<float> dense,
                                                     LocalTensor<float> seedBlocks, uint32_t outerReal, uint32_t aReal,
                                                     uint32_t bActual, uint32_t matrixElems)
    {
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            auto denseOuter = dense[outer * matrixElems];
            for (uint32_t a = 0U; a < aReal; ++a) {
                const uint32_t matrix = outer * aReal + a;
                ComputeCrossUnitD(output[matrix * matrixElems], denseOuter,
                                  seedBlocks[matrix * t_->crossM * FLOOR_MOD_FP32_BLOCK_ELEMS], bActual);
            }
        }
    }

    __aicore__ inline void ComputeCrossUnitDBatched(LocalTensor<float> output, LocalTensor<float> dense,
                                                    LocalTensor<float> seedBlocks, uint32_t outerReal, uint32_t aReal,
                                                    uint32_t bActual)
    {
        const uint32_t rowElems = t_->crossUnitBAligned;
        const uint32_t matrixElems = t_->crossM * rowElems;
        const uint32_t matrixBlocks = matrixElems / FLOOR_MOD_FP32_BLOCK_ELEMS;
        if (matrixBlocks > FLOOR_MOD_MAX_VECTOR_REPEATS || t_->crossM > FLOOR_MOD_MAX_VECTOR_REPEATS) {
            ComputeCrossUnitDFallback(output, dense, seedBlocks, outerReal, aReal, bActual, matrixElems);
            return;
        }

        auto quotient = remFp_.Get<float>();
        const uint32_t totalElems = outerReal * aReal * matrixElems;
        // Logical lanes are fully defined by Div below; padded lanes are not
        // observed by Mul/Sub, GatherMask, or UB2GM.
        const uint8_t matrixStride = static_cast<uint8_t>(matrixBlocks);
        const uint8_t seedMatrixStride = static_cast<uint8_t>(t_->crossM);
        const BinaryRepeatParams normalDiv(1U, 1U, 0U, matrixStride, 0U, seedMatrixStride);
        const BinaryRepeatParams swappedDiv(1U, 0U, 1U, matrixStride, seedMatrixStride, 0U);
        const BinaryRepeatParams normalMul(1U, 1U, 0U, matrixStride, matrixStride, seedMatrixStride);
        const BinaryRepeatParams swappedMul(1U, 1U, 1U, matrixStride, matrixStride, 0U);
        const BinaryRepeatParams normalSub(1U, 1U, 1U, matrixStride, 0U, matrixStride);
        const BinaryRepeatParams swappedSub(1U, 0U, 1U, matrixStride, seedMatrixStride, matrixStride);
        // The broadcast operand occupies one DataBlock per matrix. Keep
        // its block stride at zero within a repeat, while its repeat stride
        // advances to the next A row.
        const BinaryRepeatParams normalFma(1U, 0U, 1U, matrixStride, seedMatrixStride, 0U);
        const BinaryRepeatParams swappedFma(1U, 1U, 0U, matrixStride, 0U, seedMatrixStride);

        ComputeCrossUnitDLoop(quotient, output, dense, seedBlocks, outerReal, aReal, bActual, normalDiv, swappedDiv,
                              normalMul, swappedMul, normalSub, swappedSub, normalFma, swappedFma, 0U);
        // Keep Floor source and destination disjoint as required by the API.
        Floor(output, quotient, floorTmp_.Get<uint8_t>(), totalElems);
        if (t_->arithmeticMode == FLOOR_MOD_ARITH_FMA_NEG_DENOMINATOR) {
            // q=floor(x/y), so x-q*y is one vmadd when the reused denominator
            // is negated once.  Negating the broadcast operand is cheaper than
            // issuing separate full-output Mul and Sub passes.
            if (t_->swapped != 0U) {
                Muls(dense, dense, -1.0F, outerReal * matrixElems);
            } else {
                Muls(seedBlocks, seedBlocks, -1.0F, outerReal * aReal * t_->crossM * FLOOR_MOD_FP32_BLOCK_ELEMS);
            }
            ComputeCrossUnitDLoop(quotient, output, dense, seedBlocks, outerReal, aReal, bActual, normalDiv, swappedDiv,
                                  normalMul, swappedMul, normalSub, swappedSub, normalFma, swappedFma, 1U);
            return;
        }
        ComputeCrossUnitDLoop(quotient, output, dense, seedBlocks, outerReal, aReal, bActual, normalDiv, swappedDiv,
                              normalMul, swappedMul, normalSub, swappedSub, normalFma, swappedFma, 2U);
    }

    __aicore__ inline void ComputeCrossPaddedQuotient(LocalTensor<float> quotient, LocalTensor<float> dense,
                                                      LocalTensor<float> seed, uint32_t bActual)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(t_->crossDAligned / 8U);
        const BinaryRepeatParams normal(1U, 1U, 1U, rowBlocks, rowBlocks, 0U);
        const BinaryRepeatParams swappedDiv(1U, 1U, 1U, rowBlocks, 0U, rowBlocks);
        for (uint32_t m = 0U; m < t_->crossM; ++m) {
            const uint32_t rowBase = m * bActual * t_->crossDAligned;
            const uint32_t seedBase = m * t_->crossDAligned;
            for (uint32_t b = 0U; b < bActual; b += 255U) {
                const uint8_t repeats = static_cast<uint8_t>(MinU32(bActual - b, 255U));
                const uint32_t batchBase = rowBase + b * t_->crossDAligned;
                for (uint32_t d = 0U; d < t_->crossD; d += FLOOR_MOD_VECTOR_SEGMENT) {
                    const uint32_t count = MinU32(t_->crossD - d, FLOOR_MOD_VECTOR_SEGMENT);
                    if (t_->swapped != 0U) {
                        Div(quotient[batchBase + d], seed[seedBase + d], dense[batchBase + d],
                            static_cast<uint64_t>(count), repeats, swappedDiv);
                    } else {
                        Div(quotient[batchBase + d], dense[batchBase + d], seed[seedBase + d],
                            static_cast<uint64_t>(count), repeats, normal);
                    }
                }
            }
        }
    }

    __aicore__ inline void FloorCrossPaddedBatch(LocalTensor<float> quotient, uint32_t rows)
    {
        // A contiguous Floor over the padded slab is faster than a strided
        // Cast over only logical D.  The padding is never stored or consumed,
        // while the contiguous instruction shape preserves full vector
        // throughput and lets the public Floor implementation choose its
        // hardware-optimal sequence.
        Floor(quotient, quotient, floorTmp_.Get<uint8_t>(), rows * t_->crossDAligned);
    }

    __aicore__ inline void ComputeCrossPaddedRemainder(LocalTensor<float> output, LocalTensor<float> dense,
                                                       LocalTensor<float> seed, uint32_t bActual)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(t_->crossDAligned / 8U);
        const BinaryRepeatParams normal(1U, 1U, 1U, rowBlocks, rowBlocks, 0U);
        const BinaryRepeatParams swappedDiv(1U, 1U, 1U, rowBlocks, 0U, rowBlocks);
        const BinaryRepeatParams bothRows(1U, 1U, 1U, rowBlocks, rowBlocks, rowBlocks);
        for (uint32_t m = 0U; m < t_->crossM; ++m) {
            const uint32_t rowBase = m * bActual * t_->crossDAligned;
            const uint32_t seedBase = m * t_->crossDAligned;
            for (uint32_t b = 0U; b < bActual; b += 255U) {
                const uint8_t repeats = static_cast<uint8_t>(MinU32(bActual - b, 255U));
                const uint32_t batchBase = rowBase + b * t_->crossDAligned;
                for (uint32_t d = 0U; d < t_->crossD; d += FLOOR_MOD_VECTOR_SEGMENT) {
                    const uint32_t count = MinU32(t_->crossD - d, FLOOR_MOD_VECTOR_SEGMENT);
                    if (t_->swapped != 0U) {
                        Mul(output[batchBase + d], output[batchBase + d], dense[batchBase + d],
                            static_cast<uint64_t>(count), repeats, bothRows);
                        Sub(output[batchBase + d], seed[seedBase + d], output[batchBase + d],

                            static_cast<uint64_t>(count), repeats, swappedDiv);
                    } else {
                        Mul(output[batchBase + d], output[batchBase + d], seed[seedBase + d],
                            static_cast<uint64_t>(count), repeats, normal);
                        Sub(output[batchBase + d], dense[batchBase + d], output[batchBase + d],
                            static_cast<uint64_t>(count), repeats, bothRows);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutCrossCompactSlabs(uint64_t offset, LocalTensor<T> src, uint32_t outerReal,
                                                    uint32_t aReal, uint32_t bActual)
    {
        const uint32_t slabElements = bActual * t_->crossD;
        const uint32_t blockBytes = slabElements * static_cast<uint32_t>(sizeof(T));
        const uint32_t alignedBlockBytes = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES *
                                           FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t localSlabBytes = t_->crossUnitBTAligned * static_cast<uint32_t>(sizeof(T));
        const uint32_t srcStrideBlocks = (localSlabBytes - alignedBlockBytes) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t gmStrideBytes = (t_->crossB - bActual) * t_->crossD * static_cast<uint32_t>(sizeof(T));
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            const uint32_t rows = aReal * t_->crossM;
            const uint64_t outerOffset = offset + static_cast<uint64_t>(outer) * t_->crossA * t_->crossM * t_->crossB *
                                                      t_->crossD;
            auto outerSrc = src[outer * rows * t_->crossUnitBTAligned];
            uint32_t rowsLeft = rows;
            uint64_t gmOffset = outerOffset;
            while (rowsLeft > 0U) {
                const uint32_t batch = rowsLeft > 4095U ? 4095U : rowsLeft;
                DataCopyExtParams params{static_cast<uint16_t>(batch), blockBytes, srcStrideBlocks, gmStrideBytes, 0U};
                DataCopyPad(yGm_[gmOffset], outerSrc, params);
                outerSrc = outerSrc[batch * t_->crossUnitBTAligned];
                gmOffset += static_cast<uint64_t>(batch) * t_->crossB * t_->crossD;
                rowsLeft -= batch;
            }
        }
    }

    __aicore__ inline void GatherCrossCompactSlab(LocalTensor<float> dst, LocalTensor<float> src, uint32_t rows)
    {
        GatherMaskParams params{1U, static_cast<uint16_t>(rows),
                                static_cast<uint16_t>(t_->crossDAligned / FLOOR_MOD_FP32_BLOCK_ELEMS), 0U};
        uint64_t gathered = 0U;
        GatherMask(dst, src, static_cast<uint8_t>(7U), true, t_->crossD, params, gathered);
    }

    __aicore__ inline void ExpandCrossCompactSeed(LocalTensor<float> padded, LocalTensor<float> expanded,
                                                  LocalTensor<float> seed, uint32_t outerReal, uint32_t aReal,
                                                  uint32_t bActual)
    {
        const uint32_t srcShape[2] = {1U, t_->crossDAligned};
        const uint32_t dstShape[2] = {bActual, t_->crossDAligned};
        const uint32_t paddedSlabPitch = t_->crossBTile * t_->crossDAligned;
        auto sharedTmp = floorTmp_.Get<uint8_t>();
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            for (uint32_t a = 0U; a < aReal; ++a) {
                for (uint32_t m = 0U; m < t_->crossM; ++m) {
                    const uint32_t slab = (outer * aReal + a) * t_->crossM + m;
                    const uint32_t seedRow = slab * t_->crossDAligned;
                    if (t_->broadcastTmpBytes == 0U) {
                        Broadcast<float, 2, 0>(padded[slab * paddedSlabPitch], seed[seedRow], dstShape, srcShape);
                    } else {
                        Broadcast<float, 2, 0>(padded[slab * paddedSlabPitch], seed[seedRow], dstShape, srcShape,
                                               sharedTmp);
                    }
                }
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t slab = 0U; slab < outerReal * aReal * t_->crossM; ++slab) {
            GatherCrossCompactSlab(expanded[slab * t_->crossUnitBAligned], padded[slab * paddedSlabPitch], bActual);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeCrossCompactSlabs(LocalTensor<float> result, LocalTensor<float> dense,
                                                    LocalTensor<float> expandedSeed, uint32_t outerReal, uint32_t aReal,
                                                    uint32_t bActual)
    {
        const uint32_t slabElements = bActual * t_->crossD;
        const uint32_t slabPitch = t_->crossUnitBAligned;
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            for (uint32_t a = 0U; a < aReal; ++a) {
                for (uint32_t m = 0U; m < t_->crossM; ++m) {
                    const uint32_t outputSlab = (outer * aReal + a) * t_->crossM + m;
                    const uint32_t denseSlab = (outer * t_->crossM + m) * slabPitch;
                    const uint32_t outputBase = outputSlab * slabPitch;
                    if (t_->swapped != 0U) {
                        Div(result[outputBase], expandedSeed[outputBase], dense[denseSlab], slabElements);
                    } else {
                        Div(result[outputBase], dense[denseSlab], expandedSeed[outputBase], slabElements);
                    }
                }
            }
        }

        const uint32_t physicalElements = outerReal * aReal * t_->crossM * slabPitch;
        Floor(result, result, floorTmp_.Get<uint8_t>(), physicalElements);
        for (uint32_t outer = 0U; outer < outerReal; ++outer) {
            for (uint32_t a = 0U; a < aReal; ++a) {
                for (uint32_t m = 0U; m < t_->crossM; ++m) {
                    const uint32_t outputSlab = (outer * aReal + a) * t_->crossM + m;
                    const uint32_t denseSlab = (outer * t_->crossM + m) * slabPitch;
                    const uint32_t outputBase = outputSlab * slabPitch;
                    if (t_->swapped != 0U) {
                        Mul(result[outputBase], result[outputBase], dense[denseSlab], slabElements);
                        Sub(result[outputBase], expandedSeed[outputBase], result[outputBase], slabElements);
                    } else {
                        Mul(result[outputBase], result[outputBase], expandedSeed[outputBase], slabElements);
                        Sub(result[outputBase], dense[denseSlab], result[outputBase], slabElements);
                    }
                }
            }
        }
    }

    __aicore__ inline void LoadCrossSeed(const CrossTileInfo& tile, LocalTensor<T> seedT, GlobalTensor<T> seedGm)
    {
        if (tile.aBase == 0U && tile.aReal == t_->crossA) {
            const uint64_t offset = static_cast<uint64_t>(tile.outerBase) * t_->crossA * t_->crossM * t_->crossD;
            CopyInCrossRows(seedT, seedGm, offset, tile.outerReal * tile.aReal * t_->crossM, t_->crossDAligned);
            return;
        }
        for (uint32_t o = 0U; o < tile.outerReal; ++o) {
            const uint64_t offset = (static_cast<uint64_t>(tile.outerBase + o) * t_->crossA + tile.aBase) * t_->crossM *
                                    t_->crossD;
            CopyInCrossRows(seedT[o * tile.aReal * t_->crossM * t_->crossDAligned], seedGm, offset,
                            tile.aReal * t_->crossM, t_->crossDAligned);
        }
    }

    __aicore__ inline void LoadCrossDense(const CrossTileInfo& tile, LocalTensor<T> denseT, GlobalTensor<T> denseGm)
    {
        if (t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_COMPACT) {
            const uint64_t offset = (static_cast<uint64_t>(tile.outerBase) * t_->crossM * t_->crossB + tile.bBase) *
                                    t_->crossD;
            CopyInCrossCompactDense(denseT, denseGm, offset, tile.outerReal * t_->crossM, tile.bActual);
        } else if (tile.bActual == t_->crossB) {
            const uint64_t offset = static_cast<uint64_t>(tile.outerBase) * t_->crossM * t_->crossB * t_->crossD;
            CopyInCrossRows(denseT, denseGm, offset, tile.outerReal * t_->crossM * tile.bActual, t_->crossDAligned);
        } else {
            for (uint32_t o = 0U; o < tile.outerReal; ++o) {
                for (uint32_t m = 0U; m < t_->crossM; ++m) {
                    const uint64_t offset = ((static_cast<uint64_t>(tile.outerBase + o) * t_->crossM + m) * t_->crossB +
                                             tile.bBase) *
                                            t_->crossD;
                    const uint32_t local = (o * t_->crossM + m) * tile.bActual * t_->crossDAligned;
                    CopyInCrossRows(denseT[local], denseGm, offset, tile.bActual, t_->crossDAligned);
                }
            }
        }
    }

    __aicore__ inline void LoadCrossInput(uint64_t task, uint32_t slot, event_t inputFree, event_t inputReady)
    {
        const auto tile = DecodeCrossTileInfo(task);
        auto seedT = seedIn_.Get<T>()[slot * crossSeedCapacity_];
        auto denseT = denseIn_.Get<T>()[slot * crossDenseTCapacity_];
        GlobalTensor<T> seedGm = t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_;
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;

        WaitFlag<HardEvent::V_MTE2>(inputFree);
        LoadCrossSeed(tile, seedT, seedGm);
        LoadCrossDense(tile, denseT, denseGm);
        SetFlag<HardEvent::MTE2_V>(inputReady);
    }

    __aicore__ inline void InitCrossEvents(event_t inputFree[2], event_t inputReady[2], event_t outputReady[2],
                                           event_t outputFree[2])
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            inputFree[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            inputReady[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            outputReady[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            outputFree[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
        }
    }

    __aicore__ inline void PrepareCrossFp32(LocalTensor<float>& seed, LocalTensor<float>& dense, LocalTensor<T> seedT,
                                            LocalTensor<T> denseT, uint32_t seedCount, uint32_t denseRows,
                                            uint32_t denseLogical, bool compact)
    {
        if constexpr (std::is_same_v<T, float>) {
            seed = seedT.template ReinterpretCast<float>();
            dense = denseT.template ReinterpretCast<float>();
        } else {
            seed = seedFp_.Get<float>();
            dense = denseFp_.Get<float>();
            CastToFp32(seed, seedT, seedCount);
            if (compact) {
                CastUnitRowsToFp32(dense, denseT, denseRows, denseLogical);
            } else {
                CastToFp32(dense, denseT, denseLogical);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReleaseCrossEvents(event_t inputFree[2], event_t inputReady[2], event_t outputReady[2],
                                              event_t outputFree[2])
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(inputFree[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputReady[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outputReady[slot]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outputFree[slot]);
        }
    }

    __aicore__ inline void ProcessCrossCompactTask(const CrossTileInfo& tile, LocalTensor<float> result,
                                                   LocalTensor<float> seed, LocalTensor<float> dense, uint32_t slot,
                                                   event_t inputFree, event_t outputReady, event_t outputFree)
    {
        const auto expandedSeed = expSeedFp_.Get<float>();
        ExpandCrossCompactSeed(result, expandedSeed, seed, tile.outerReal, tile.aReal, tile.bActual);
        ComputeCrossCompactSlabs(result, dense, expandedSeed, tile.outerReal, tile.aReal, tile.bActual);
        LocalTensor<T> compactOutput;
        if constexpr (std::is_same_v<T, float>) {
            compactOutput = result.template ReinterpretCast<T>();
        } else {
            compactOutput = outT_.Get<T>()[slot * crossOutputTCapacity_];
            CastUnitRowsFromFp32(compactOutput, result, tile.outerReal * tile.aReal * t_->crossM,
                                 tile.bActual * t_->crossD);
        }
        SetFlag<HardEvent::V_MTE2>(inputFree);
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);
        const uint64_t outOffset = ((((static_cast<uint64_t>(tile.outerBase) * t_->crossA + tile.aBase) * t_->crossM) *
                                     t_->crossB) +
                                    tile.bBase) *
                                   t_->crossD;
        CopyOutCrossCompactSlabs(outOffset, compactOutput, tile.outerReal, tile.aReal, tile.bActual);
        SetFlag<HardEvent::MTE3_V>(outputFree);
    }

    __aicore__ inline void ComputeCrossPaddedTask(const CrossTileInfo& tile, LocalTensor<float> result,
                                                  LocalTensor<float> seed, LocalTensor<float> dense,
                                                  LocalTensor<T> outputBatch)
    {
        const uint32_t workElems = t_->crossM * tile.bActual * t_->crossDAligned;
        for (uint32_t o = 0U; o < tile.outerReal; ++o) {
            auto denseOuter = dense[o * workElems];
            for (uint32_t a = 0U; a < tile.aReal; ++a) {
                const uint32_t resultBase = (o * tile.aReal + a) * workElems;
                const uint32_t seedBase = (o * tile.aReal + a) * t_->crossM * t_->crossDAligned;
                ComputeCrossPaddedQuotient(result[resultBase], denseOuter, seed[seedBase], tile.bActual);
            }
        }
        FloorCrossPaddedBatch(result, tile.outerReal * tile.aReal * t_->crossM * tile.bActual);
        for (uint32_t o = 0U; o < tile.outerReal; ++o) {
            auto denseOuter = dense[o * workElems];
            for (uint32_t a = 0U; a < tile.aReal; ++a) {
                const uint32_t resultBase = (o * tile.aReal + a) * workElems;
                const uint32_t seedBase = (o * tile.aReal + a) * t_->crossM * t_->crossDAligned;
                ComputeCrossPaddedRemainder(result[resultBase], denseOuter, seed[seedBase], tile.bActual);
                if constexpr (!std::is_same_v<T, float>) {
                    CastFromFp32(outputBatch[resultBase], result[resultBase], workElems);
                }
            }
        }
    }

    __aicore__ inline void StoreCrossPaddedTask(const CrossTileInfo& tile, LocalTensor<T> outputBatch)
    {
        const uint32_t workElems = t_->crossM * tile.bActual * t_->crossDAligned;
        if (tile.aBase == 0U && tile.aReal == t_->crossA && tile.bActual == t_->crossB) {
            const uint64_t outOffset = static_cast<uint64_t>(tile.outerBase) * t_->crossA * t_->crossM * t_->crossB *
                                       t_->crossD;
            CopyOutCrossRows(outOffset, outputBatch, tile.outerReal * tile.aReal * t_->crossM * tile.bActual);
        } else if (tile.bActual == t_->crossB) {
            for (uint32_t o = 0U; o < tile.outerReal; ++o) {
                const uint64_t outOffset = (static_cast<uint64_t>(tile.outerBase + o) * t_->crossA + tile.aBase) *
                                           t_->crossM * t_->crossB * t_->crossD;
                CopyOutCrossRows(outOffset, outputBatch[o * tile.aReal * workElems],
                                 tile.aReal * t_->crossM * tile.bActual);
            }
        } else {
            for (uint32_t o = 0U; o < tile.outerReal; ++o) {
                for (uint32_t a = 0U; a < tile.aReal; ++a) {
                    const uint32_t resultBase = (o * tile.aReal + a) * workElems;
                    const uint64_t outOffset = (((static_cast<uint64_t>(tile.outerBase + o) * t_->crossA + tile.aBase +
                                                  a) *
                                                 t_->crossM * t_->crossB) +
                                                tile.bBase) *
                                               t_->crossD;
                    CopyOutCrossPadded(outOffset, outputBatch[resultBase], tile.bActual);
                }
            }
        }
    }

    __aicore__ inline void ProcessCrossPaddedTask(const CrossTileInfo& tile, LocalTensor<float> result,
                                                  LocalTensor<float> seed, LocalTensor<float> dense,
                                                  LocalTensor<T> outputBatch, event_t inputFree, event_t outputReady,
                                                  event_t outputFree)
    {
        ComputeCrossPaddedTask(tile, result, seed, dense, outputBatch);
        SetFlag<HardEvent::V_MTE2>(inputFree);
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);
        StoreCrossPaddedTask(tile, outputBatch);
        SetFlag<HardEvent::MTE3_V>(outputFree);
    }

    __aicore__ inline void ProcessCrossed()
    {
        if (t_->crossD == 1U) {
            ProcessCrossedUnitD();
            return;
        }
        const bool compactSlabs = t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_COMPACT;
        event_t inputFree[2];
        event_t inputReady[2];
        event_t outputReady[2];
        event_t outputFree[2];
        InitCrossEvents(inputFree, inputReady, outputReady, outputFree);
        uint64_t taskBegin = 0U;
        uint64_t taskEnd = 0U;
        GetTaskRange(t_->crossTotalTasks, taskBegin, taskEnd);
        ProcessCrossTaskRange(taskBegin, taskEnd, compactSlabs, inputFree, inputReady, outputReady, outputFree);
        ReleaseCrossEvents(inputFree, inputReady, outputReady, outputFree);
    }

    __aicore__ inline void ProcessCrossTaskRange(uint64_t taskBegin, uint64_t taskEnd, bool compactSlabs,
                                                 event_t* inputFree, event_t* inputReady, event_t* outputReady,
                                                 event_t* outputFree)
    {
        if (taskBegin < taskEnd) {
            LoadCrossInput(taskBegin, 0U, inputFree[0], inputReady[0]);
        }
        for (uint64_t task = taskBegin, iteration = 0U; task < taskEnd; ++task, ++iteration) {
            const auto tile = DecodeCrossTileInfo(task);
            const uint32_t seedCount = tile.outerReal * tile.aReal * t_->crossM * t_->crossDAligned;
            const uint32_t denseCount = compactSlabs ? tile.outerReal * t_->crossM * t_->crossUnitBAligned :
                                                       tile.outerReal * t_->crossM * tile.bActual * t_->crossDAligned;
            const uint32_t slot = iteration & 1U;
            auto seedT = seedIn_.Get<T>()[slot * crossSeedCapacity_];
            auto denseT = denseIn_.Get<T>()[slot * crossDenseTCapacity_];
            WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
            if (task + 1U < taskEnd) {
                LoadCrossInput(task + 1U, slot ^ 1U, inputFree[slot ^ 1U], inputReady[slot ^ 1U]);
            }
            WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
            LocalTensor<float> seed;
            LocalTensor<float> dense;
            PrepareCrossFp32(seed, dense, seedT, denseT, seedCount, tile.outerReal * t_->crossM,
                             compactSlabs ? tile.bActual * t_->crossD : denseCount, compactSlabs);
            auto result = expDenseFp_.Get<float>()[slot * crossOutputCapacity_];
            if (compactSlabs) {
                ProcessCrossCompactTask(tile, result, seed, dense, slot, inputFree[slot], outputReady[slot],
                                        outputFree[slot]);
                continue;
            }
            LocalTensor<T> outputBatch;
            if constexpr (std::is_same_v<T, float>) {
                outputBatch = result.template ReinterpretCast<T>();
            } else {
                outputBatch = outT_.Get<T>()[slot * crossOutputCapacity_];
            }
            ProcessCrossPaddedTask(tile, result, seed, dense, outputBatch, inputFree[slot], outputReady[slot],
                                   outputFree[slot]);
        }
    }
    __aicore__ inline void LoadCrossUnitInputs(const CrossTileInfo& tile, LocalTensor<T> seed, LocalTensor<T> dense)
    {
        GlobalTensor<T> seedGm = t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_;
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;
        const uint64_t seedOffset = (static_cast<uint64_t>(tile.outerBase) * t_->crossA + tile.aBase) * t_->crossM;
        if (tile.aReal == t_->crossA) {
            CopyInCompact(seed, seedGm, seedOffset, tile.outerReal * tile.aReal * t_->crossM);
        } else {
            for (uint32_t outer = 0U; outer < tile.outerReal; ++outer) {
                const uint64_t offset = (static_cast<uint64_t>(tile.outerBase + outer) * t_->crossA + tile.aBase) *
                                        t_->crossM;
                CopyInCompact(seed[outer * tile.aReal * t_->crossM], seedGm, offset, tile.aReal * t_->crossM);
            }
        }
        const uint64_t denseOffset = static_cast<uint64_t>(tile.outerBase) * t_->crossM * t_->crossB + tile.bBase;
        CopyInCrossUnitD(dense, denseGm, denseOffset, tile.outerReal * t_->crossM, tile.bActual);
    }

    __aicore__ inline LocalTensor<T> ComputeCrossUnitOutput(const CrossTileInfo& tile, LocalTensor<T> seedT,
                                                            LocalTensor<T> denseT, uint32_t slot)
    {
        LocalTensor<float> seed;
        LocalTensor<float> dense;
        const uint32_t seedCount = tile.outerReal * tile.aReal * t_->crossM;
        PrepareCrossFp32(seed, dense, seedT, denseT, seedCount, tile.outerReal * t_->crossM, tile.bActual, true);
        auto seedBlocks = expSeedFp_.Get<float>();
        for (uint32_t base = 0U; base < seedCount; base += 2040U) {
            const uint32_t batch = MinU32(seedCount - base, 2040U);
            Brcb(seedBlocks[base * 8U], seed[base], static_cast<uint8_t>((batch + 7U) / 8U), {1U, 8U});
        }
        PipeBarrier<PIPE_V>();
        auto result = expDenseFp_.Get<float>()[slot * crossOutputCapacity_];
        ComputeCrossUnitDBatched(result, dense, seedBlocks, tile.outerReal, tile.aReal, tile.bActual);
        const bool compact = t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_COMPACT;
        if (compact) {
            PipeBarrier<PIPE_V>();
            GatherCrossUnitDOutput(result, seedCount, tile.bActual);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same_v<T, float>) {
            return result.template ReinterpretCast<T>();
        } else {
            auto output = outT_.Get<T>()[slot * crossOutputTCapacity_];
            if (compact) {
                CastFromFp32(output, result, seedCount * tile.bActual);
            } else {
                CastUnitRowsFromFp32(output, result, seedCount, tile.bActual);
            }
            return output;
        }
    }

    __aicore__ inline void StoreCrossUnitOutput(const CrossTileInfo& tile, LocalTensor<T> output)
    {
        const uint64_t offset = (static_cast<uint64_t>(tile.outerBase) * t_->crossA + tile.aBase) * t_->crossM *
                                    t_->crossB +
                                tile.bBase;
        if (t_->reuseLayout == FLOOR_MOD_REUSE_PADDED_COMPACT) {
            CopyOutCrossUnitDCompact(offset, output, tile.outerReal, tile.aReal, tile.bActual);
        } else if (tile.aReal == t_->crossA) {
            CopyOutCrossUnitD(offset, output, tile.outerReal * tile.aReal * t_->crossM, tile.bActual);
        } else {
            for (uint32_t outer = 0U; outer < tile.outerReal; ++outer) {
                const uint64_t outerOffset = (static_cast<uint64_t>(tile.outerBase + outer) * t_->crossA + tile.aBase) *
                                                 t_->crossM * t_->crossB +
                                             tile.bBase;
                CopyOutCrossUnitD(outerOffset, output[outer * tile.aReal * t_->crossM * t_->crossUnitBTAligned],
                                  tile.aReal * t_->crossM, tile.bActual);
            }
        }
    }

    __aicore__ inline void ProcessCrossedUnitDImpl()
    {
        event_t inputFree[2], inputReady[2], outputReady[2], outputFree[2];
        InitCrossEvents(inputFree, inputReady, outputReady, outputFree);
        uint64_t begin = 0U;
        uint64_t end = 0U;
        GetTaskRange(t_->crossTotalTasks, begin, end);
        for (uint64_t task = begin; task < end; ++task) {
            const uint32_t slot = static_cast<uint32_t>((task - begin) & 1U);
            const auto tile = DecodeCrossTileInfo(task);
            auto seed = seedIn_.Get<T>()[slot * crossSeedCapacity_];
            auto dense = denseIn_.Get<T>()[slot * crossDenseTCapacity_];
            WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
            LoadCrossUnitInputs(tile, seed, dense);
            SetFlag<HardEvent::MTE2_V>(inputReady[slot]);
            WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
            auto output = ComputeCrossUnitOutput(tile, seed, dense, slot);
            SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
            SetFlag<HardEvent::V_MTE3>(outputReady[slot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady[slot]);
            StoreCrossUnitOutput(tile, output);
            SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
        }
        ReleaseCrossEvents(inputFree, inputReady, outputReady, outputFree);
    }

    __aicore__ inline void ProcessCrossedUnitD() { ProcessCrossedUnitDImpl(); }

private:
    const FloorModTilingData* t_ = nullptr;
    TPipe pipe_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    TBuf<QuePosition::VECIN> x1In_;
    TBuf<QuePosition::VECIN> x2In_;
    TQue<QuePosition::VECIN, 1> denseX1Queue_;
    TQue<QuePosition::VECIN, 1> denseX2Queue_;
    TQue<QuePosition::VECOUT, 1> denseYQueue_;
    TBuf<QuePosition::VECIN> seedIn_;
    TBuf<QuePosition::VECIN> denseIn_;
    TBuf<QuePosition::VECOUT> outT_;
    TBuf<TPosition::VECCALC> aFp_;
    TBuf<TPosition::VECCALC> bFp_;
    TBuf<TPosition::VECCALC> seedFp_;
    TBuf<TPosition::VECCALC> denseFp_;
    TBuf<TPosition::VECCALC> expSeedFp_;
    TBuf<TPosition::VECCALC> expDenseFp_;
    TBuf<TPosition::VECCALC> remFp_;
    TBuf<TPosition::VECCALC> floorTmp_;
    uint32_t denseCapacity_ = 0U;
    uint32_t rowCapacity_ = 0U;
    uint32_t seedTCapacity_ = 0U;
    uint32_t seedFpCapacity_ = 0U;
    uint32_t reuseSlotElems_ = 0U;
    uint32_t reuseTSlotElems_ = 0U;
    uint32_t posSeedCapacity_ = 0U;
    uint32_t posRowCapacity_ = 0U;
    uint32_t posWorkCapacity_ = 0U;
    uint32_t posExpandedCapacity_ = 0U;
    uint32_t crossSeedCapacity_ = 0U;
    uint32_t crossDenseTCapacity_ = 0U;
    uint32_t crossDenseCapacity_ = 0U;
    uint32_t crossRowCapacity_ = 0U;
    uint32_t crossTRowCapacity_ = 0U;
    uint32_t crossCompactCapacity_ = 0U;
    uint32_t crossOutputCapacity_ = 0U;
    uint32_t crossOutputTCapacity_ = 0U;
    uint32_t seedBroadcastCapacity_ = 0U;
};

// One input element broadcasts over a dense tensor. Each core owns a contiguous
// task interval and reads the scalar once. Single-wave inputs use one compact
// slot; multi-wave inputs ping-pong two raw slots so GM2UB, Vector, and UB2GM can
// overlap without UB2UB staging or generic task decoding.
template <typename T>
class FloorModScalarBroadcast {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const FloorModTilingData* tiling)
    {
        t_ = tiling;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        bufferNum_ = t_->scalarDoubleBuffer != 0U ? 2U : 1U;
        pipe_.InitBuffer(denseIn_, bufferNum_ * t_->maxTRowElems * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(denseFp_, t_->maxFp32RowElems * sizeof(float));
        }
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe_.InitBuffer(scalarIn_, FLOOR_MOD_UB_BLOCK_BYTES);
        }
        pipe_.InitBuffer(scalarFp_, FLOOR_MOD_VECTOR_SEGMENT * sizeof(float));
        pipe_.InitBuffer(quotientFp_, t_->maxFp32RowElems * sizeof(float));
        pipe_.InitBuffer(floorResultFp_, t_->maxFp32RowElems * sizeof(float));
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline float LoadScalar(GlobalTensor<T> scalarGm, LocalTensor<float> scalarVector, event_t mte2)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            auto scalarT = scalarIn_.Get<T>();
            DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
            DataCopyExtParams params{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(scalarT, scalarGm, params, pad);
            SetFlag<HardEvent::MTE2_V>(mte2);
            WaitFlag<HardEvent::MTE2_V>(mte2);
            Cast(scalarVector, scalarT, AscendC::RoundMode::CAST_NONE, 1U);
            PipeBarrier<PIPE_V>();
            return scalarVector.GetValue(0);
        }
        return static_cast<float>(scalarGm.GetValue(0));
    }

    __aicore__ inline void ProcessDoubleBufferedTiles(GlobalTensor<T> denseGm, LocalTensor<float> scalarVector,
                                                      event_t mte2, event_t vMte3, const event_t slotFree[2])
    {
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        uint64_t taskBegin = 0U;
        uint64_t taskEnd = 0U;
        GetTaskRange(taskBegin, taskEnd);
        uint32_t iteration = 0U;
        for (uint64_t task = taskBegin; task < taskEnd; ++task) {
            const uint32_t slot = iteration & 1U;
            const uint64_t offset = task * t_->e2Tile;
            const uint32_t count = static_cast<uint32_t>(MinU64(t_->e2Tile, t_->totalElements - offset));
            WaitFlag<HardEvent::MTE3_MTE2>(slotFree[slot]);
            auto denseT = denseIn_.Get<T>()[slot * t_->maxTRowElems];
            DataCopyExtParams params{1U, count * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(denseT, denseGm[offset], params, pad);
            SetFlag<HardEvent::MTE2_V>(mte2);
            WaitFlag<HardEvent::MTE2_V>(mte2);
            ComputeTile(denseT, scalarVector, count);
            SetFlag<HardEvent::V_MTE3>(vMte3);
            WaitFlag<HardEvent::V_MTE3>(vMte3);
            DataCopyPad(yGm_[offset], denseT, params);
            SetFlag<HardEvent::MTE3_MTE2>(slotFree[slot]);
            ++iteration;
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        if (block >= t_->coreNum || t_->totalElements == 0U || t_->e2Tile == 0U || t_->totalTasks == 0U ||
            t_->e2TileCount != t_->totalTasks || t_->e2Tile > t_->maxTRowElems ||
            t_->e2Tile > FLOOR_MOD_MAX_VECTOR_REPEATS * FLOOR_MOD_VECTOR_SEGMENT + FLOOR_MOD_VECTOR_SEGMENT - 1U) {
            return;
        }
        if (bufferNum_ == 1U) {
            ProcessSingleTile(block);
            return;
        }

        GlobalTensor<T> scalarGm = t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_;
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;
        auto scalarVector = scalarFp_.Get<float>();
        // Keep the scalar controller on the proven event ordering: a single
        // MTE2->V and V->MTE3 dependency is reused after each tile, while
        // slotFree protects the two raw input slots.  Issuing a second
        // MTE2->V event before the current tile has completed can leave the
        // vector pipe waiting forever on the DAV event scoreboard.
        const event_t mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        const event_t vMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        const event_t slotFree[2] = {static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>()),
                                     static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>())};
        SetFlag<HardEvent::MTE3_MTE2>(slotFree[0]);
        SetFlag<HardEvent::MTE3_MTE2>(slotFree[1]);

        Duplicate(scalarVector, LoadScalar(scalarGm, scalarVector, mte2), FLOOR_MOD_VECTOR_SEGMENT);
        ProcessDoubleBufferedTiles(denseGm, scalarVector, mte2, vMte3, slotFree);
        WaitFlag<HardEvent::MTE3_MTE2>(slotFree[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(slotFree[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(slotFree[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(slotFree[1]);
    }

private:
    __aicore__ inline void ProcessSingleTile(uint32_t block)
    {
        GlobalTensor<T> scalarGm = t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_;
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;
        auto scalarVector = scalarFp_.Get<float>();
        auto denseT = denseIn_.Get<T>();
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        LocalTensor<T> scalarT;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            scalarT = scalarIn_.Get<T>();
            DataCopyExtParams scalarParams{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(scalarT, scalarGm, scalarParams, pad);
        }
        const event_t inputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        float scalar = 0.0F;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            Cast(scalarVector, scalarT, AscendC::RoundMode::CAST_NONE, 1U);
            PipeBarrier<PIPE_V>();
            scalar = scalarVector.GetValue(0);
        } else {
            scalar = static_cast<float>(scalarGm.GetValue(0));
        }
        Duplicate(scalarVector, scalar, FLOOR_MOD_VECTOR_SEGMENT);
        const event_t outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t slotFree = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        uint64_t taskBegin = 0U;
        uint64_t taskEnd = 0U;
        GetTaskRange(taskBegin, taskEnd);
        for (uint64_t task = taskBegin; task < taskEnd; ++task) {
            const uint64_t offset = task * t_->e2Tile;
            const uint32_t count = static_cast<uint32_t>(MinU64(t_->e2Tile, t_->totalElements - offset));
            DataCopyExtParams copyParams{1U, count * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(denseT, denseGm[offset], copyParams, pad);
            SetFlag<HardEvent::MTE2_V>(inputReady);
            WaitFlag<HardEvent::MTE2_V>(inputReady);
            ComputeTile(denseT, scalarVector, count);
            SetFlag<HardEvent::V_MTE3>(outputReady);
            WaitFlag<HardEvent::V_MTE3>(outputReady);
            DataCopyPad(yGm_[offset], denseT, copyParams);
            SetFlag<HardEvent::MTE3_MTE2>(slotFree);
            WaitFlag<HardEvent::MTE3_MTE2>(slotFree);
        }
    }

    __aicore__ inline void ComputeTile(LocalTensor<T> denseT, LocalTensor<float> scalarVector, uint32_t count)
    {
        // GM traffic remains tile-sized. Only the vector API calls are split:
        // int64 cast count-mode cannot safely exceed 255 hardware repeats.
        uint32_t offset = 0U;
        while (offset < count) {
            const uint32_t chunk = MinU64(count - offset, FLOOR_MOD_SCALAR_COMPUTE_CHUNK);
            ComputeTileChunk(denseT[offset], scalarVector, chunk);
            offset += chunk;
        }
    }

    template <bool Remainder>
    __aicore__ inline void ComputeScalarStage(LocalTensor<float> quotient, LocalTensor<float> dense,
                                              LocalTensor<float> scalar, uint32_t count)
    {
        const BinaryRepeatParams normal(1U, 1U, 1U, 8U, 8U, 0U);
        const BinaryRepeatParams swapped(1U, 1U, 1U, 8U, 0U, 8U);
        const BinaryRepeatParams bothVectors(1U, 1U, 1U, 8U, 8U, 8U);
        const uint8_t repeats = static_cast<uint8_t>(count / FLOOR_MOD_VECTOR_SEGMENT);
        const uint32_t repeated = static_cast<uint32_t>(repeats) * FLOOR_MOD_VECTOR_SEGMENT;
        if (repeats > 0U) {
            ComputeBroadcastStage<Remainder>(quotient, dense, scalar, FLOOR_MOD_VECTOR_SEGMENT, repeats, normal,
                                             swapped, bothVectors, t_->swapped != 0U);
        }
        if (count > repeated) {
            ComputeBroadcastStage<Remainder>(quotient[repeated], dense[repeated], scalar, count - repeated, 1U, normal,
                                             swapped, bothVectors, t_->swapped != 0U);
        }
    }

    __aicore__ inline void ComputeTileChunk(LocalTensor<T> denseT, LocalTensor<float> scalarVector, uint32_t count)
    {
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = denseT.template ReinterpretCast<float>();
        } else {
            dense = denseFp_.Get<float>();
            CastToFp32(dense, denseT, count);
        }
        auto quotient = quotientFp_.Get<float>();
        auto floorResult = floorResultFp_.Get<float>();
        ComputeScalarStage<false>(quotient, dense, scalarVector, count);
        Floor(floorResult, quotient, floorTmp_.Get<uint8_t>(), count);
        ComputeScalarStage<true>(floorResult, dense, scalarVector, count);
        if constexpr (!std::is_same_v<T, float>) {
            CastFromFp32(denseT, dense, count);
        }
    }

    __aicore__ static inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs) { return lhs > rhs ? lhs : rhs; }
    __aicore__ static inline uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline void GetTaskRange(uint64_t& begin, uint64_t& end) const
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t base = t_->totalTasks / t_->coreNum;
        const uint64_t extra = t_->totalTasks % t_->coreNum;
        begin = block * base + (block < extra ? block : extra);
        end = begin + base + (block < extra ? 1U : 0U);
    }

    const FloorModTilingData* t_ = nullptr;
    TPipe pipe_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    TBuf<QuePosition::VECIN> scalarIn_;
    TBuf<QuePosition::VECIN> denseIn_;
    TBuf<TPosition::VECCALC> denseFp_;
    TBuf<TPosition::VECCALC> scalarFp_;
    TBuf<TPosition::VECCALC> quotientFp_;
    TBuf<TPosition::VECCALC> floorResultFp_;
    TBuf<TPosition::VECCALC> floorTmp_;
    uint32_t bufferNum_ = 1U;
};

// Compile-time path for a compact trailing scalar broadcast. Keeping this
// controller separate removes the generic task decoder and layout branches
// from the hot loop. It also makes D rows, rather than D tiles, the ownership
// unit: each core reads and writes contiguous spans for its entire interval.
template <typename T>
class FloorModDenseTailBatch {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const FloorModTilingData* tiling)
    {
        t_ = tiling;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        tileRows_ = t_->dTile;
        tailElements_ = t_->e2;
        tileElements_ = t_->maxTRowElems;
        bufferNum_ = t_->reuseSchedule != 0U ? 2U : 1U;

        pipe_.InitBuffer(x1Queue_, bufferNum_, tileElements_ * sizeof(T));
        pipe_.InitBuffer(x2Queue_, bufferNum_, tileElements_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, bufferNum_, tileElements_ * sizeof(T));
        const uint32_t seedBytes = (tileRows_ * sizeof(T) + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES *
                                   FLOOR_MOD_UB_BLOCK_BYTES;
        pipe_.InitBuffer(scalarSeed_, seedBytes);
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(x1Fp_, t_->maxFp32RowElems * sizeof(float));
            pipe_.InitBuffer(x2Fp_, t_->maxFp32RowElems * sizeof(float));
        }
        pipe_.InitBuffer(remainderFp_, t_->maxFp32RowElems * sizeof(float));
        pipe_.InitBuffer(floorTmp_, MaxU32(MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes), FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void Process()
    {
        if (t_->reuseLayout != FLOOR_MOD_REUSE_DENSE_TAIL_BATCH || GetBlockIdx() >= t_->coreNum || t_->D == 0U ||
            tailElements_ == 0U || tileRows_ == 0U) {
            return;
        }
        const uint64_t core = GetBlockIdx();
        const uint64_t base = t_->D / t_->coreNum;
        const uint64_t extra = t_->D % t_->coreNum;
        uint64_t rowOffset = core * base + (core < extra ? core : extra);
        uint64_t remainingRows = base + (core < extra ? 1U : 0U);
        if (remainingRows == 0U) {
            return;
        }
        if (bufferNum_ == 1U) {
            ProcessSingleBuffered(rowOffset, remainingRows);
            return;
        }

        ProcessDoubleBuffered(rowOffset, remainingRows);
    }

private:
    __aicore__ inline void ProcessDoubleBuffered(uint64_t rowOffset, uint64_t remainingRows)
    {
        uint32_t currentRows = MinU32(remainingRows, tileRows_);
        CopyIn(rowOffset, currentRows);
        uint64_t nextRowOffset = rowOffset + currentRows;
        remainingRows -= currentRows;
        bool hasPendingOutput = false;
        uint64_t pendingRowOffset = 0U;
        uint32_t pendingRows = 0U;
        while (currentRows > 0U) {
            auto x1 = x1Queue_.DeQue<T>();
            auto x2 = x2Queue_.DeQue<T>();
            auto scalar = scalarSeed_.Get<T>();
            auto expanded = t_->seedIsX1 != 0U ? x1 : x2;
            const uint32_t srcShape[2] = {currentRows, 1U};
            const uint32_t dstShape[2] = {currentRows, tailElements_};
            BroadCast<T, 2, 1>(expanded, scalar, dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            uint32_t nextRows = 0U;
            if (remainingRows > 0U) {
                const event_t seedFree = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(seedFree);
                WaitFlag<HardEvent::V_MTE2>(seedFree);
                nextRows = MinU32(remainingRows, tileRows_);
                CopyIn(nextRowOffset, nextRows);
                nextRowOffset += nextRows;
                remainingRows -= nextRows;
            }
            if (hasPendingOutput) {
                CopyOut(pendingRowOffset, pendingRows);
            }
            Compute(x1, x2, currentRows * tailElements_);
            x1Queue_.FreeTensor(x1);
            x2Queue_.FreeTensor(x2);
            hasPendingOutput = true;
            pendingRowOffset = rowOffset;
            pendingRows = currentRows;
            if (nextRows > 0U) {
                rowOffset = nextRowOffset - nextRows;
            }
            currentRows = nextRows;
        }
        if (hasPendingOutput) {
            CopyOut(pendingRowOffset, pendingRows);
        }
    }

    __aicore__ static inline uint32_t MinU32(uint64_t value, uint32_t limit)
    {
        return static_cast<uint32_t>(value < limit ? value : limit);
    }

    __aicore__ static inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs) { return lhs > rhs ? lhs : rhs; }

    __aicore__ inline void ProcessSingleBuffered(uint64_t rowOffset, uint64_t remainingRows)
    {
        while (remainingRows > 0U) {
            const uint32_t rows = MinU32(remainingRows, tileRows_);
            CopyIn(rowOffset, rows);
            auto x1 = x1Queue_.DeQue<T>();
            auto x2 = x2Queue_.DeQue<T>();
            auto scalar = scalarSeed_.Get<T>();
            auto expanded = t_->seedIsX1 != 0U ? x1 : x2;
            const uint32_t srcShape[2] = {rows, 1U};
            const uint32_t dstShape[2] = {rows, tailElements_};
            BroadCast<T, 2, 1>(expanded, scalar, dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            Compute(x1, x2, rows * tailElements_);
            x1Queue_.FreeTensor(x1);
            x2Queue_.FreeTensor(x2);
            CopyOut(rowOffset, rows);
            rowOffset += rows;
            remainingRows -= rows;
        }
    }

    __aicore__ inline void CopyIn(uint64_t rowOffset, uint32_t rows)
    {
        auto x1 = x1Queue_.AllocTensor<T>();
        auto x2 = x2Queue_.AllocTensor<T>();
        auto scalar = scalarSeed_.Get<T>();
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyExtParams scalarParams{1U, rows * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyExtParams denseParams{1U, rows * tailElements_ * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        if (t_->seedIsX1 != 0U) {
            DataCopyPad(scalar, x1Gm_[rowOffset], scalarParams, pad);
            DataCopyPad(x2, x2Gm_[rowOffset * tailElements_], denseParams, pad);
        } else {
            DataCopyPad(x1, x1Gm_[rowOffset * tailElements_], denseParams, pad);
            DataCopyPad(scalar, x2Gm_[rowOffset], scalarParams, pad);
        }
        x1Queue_.EnQue(x1);
        x2Queue_.EnQue(x2);
    }

    __aicore__ inline void Compute(LocalTensor<T> x1, LocalTensor<T> x2, uint32_t count)
    {
        auto output = outputQueue_.AllocTensor<T>();
        auto quotient = remainderFp_.Get<float>();
        LocalTensor<float> a;
        LocalTensor<float> b;
        LocalTensor<float> result;
        if constexpr (std::is_same_v<T, float>) {
            a = x1.template ReinterpretCast<float>();
            b = x2.template ReinterpretCast<float>();
            result = output.template ReinterpretCast<float>();
        } else {
            a = x1Fp_.Get<float>();
            b = x2Fp_.Get<float>();
            Cast(a, x1, AscendC::RoundMode::CAST_NONE, count);
            Cast(b, x2, AscendC::RoundMode::CAST_NONE, count);
            result = a;
        }
        Div(quotient, a, b, count);
        Floor(quotient, quotient, floorTmp_.Get<uint8_t>(), count);
        Mul(quotient, quotient, b, count);
        Sub(result, a, quotient, count);
        if constexpr (!std::is_same_v<T, float>) {
            if constexpr (std::is_same_v<T, half>) {
                Cast(output, result, AscendC::RoundMode::CAST_NONE, count);
            } else {
                Cast(output, result, AscendC::RoundMode::CAST_RINT, count);
            }
        }
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(uint64_t rowOffset, uint32_t rows)
    {
        auto output = outputQueue_.DeQue<T>();
        DataCopyExtParams params{1U, rows * tailElements_ * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(yGm_[rowOffset * tailElements_], output, params);
        outputQueue_.FreeTensor(output);
    }

    const FloorModTilingData* t_ = nullptr;
    TPipe pipe_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, 2> x1Queue_;
    TQue<QuePosition::VECIN, 2> x2Queue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<QuePosition::VECIN> scalarSeed_;
    TBuf<TPosition::VECCALC> x1Fp_;
    TBuf<TPosition::VECCALC> x2Fp_;
    TBuf<TPosition::VECCALC> remainderFp_;
    TBuf<TPosition::VECCALC> floorTmp_;
    uint32_t tileRows_ = 0U;
    uint32_t tailElements_ = 0U;
    uint32_t tileElements_ = 0U;
    uint32_t bufferNum_ = 1U;
};

// Compile-time path for [D] -> [rows,D] reuse. The materialized layout moves
// each dense [batch,D] slab contiguously, broadcasts the seed with an aligned
// trailing dimension, and gathers it to compact storage before arithmetic.
// The repeat layout batches padded rows with DataCopyPad and reuses the seed
// through zero-stride vector repeats. Neither layout performs an UB2UB copy.
template <typename T>
class FloorModCompactRowBatch {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const FloorModTilingData* tiling)
    {
        t_ = tiling;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        width_ = static_cast<uint32_t>(t_->D);
        tileTElems_ = t_->maxTRowElems;
        tileFpElems_ = t_->maxFp32RowElems;
        tRowElems_ = AlignElements(width_);
        fpRowElems_ = (width_ + 7U) / 8U * 8U;
        seedTElems_ = tRowElems_;
        seedFpElems_ = fpRowElems_;

        pipe_.InitBuffer(denseIn_, 2U * tileTElems_ * sizeof(T));
        pipe_.InitBuffer(seedIn_, seedTElems_ * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(seedFp_, seedFpElems_ * sizeof(float));
            pipe_.InitBuffer(denseFp_, tileFpElems_ * sizeof(float));
        }
        if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_ROW_BATCH) {
            pipe_.InitBuffer(expandedSeedFp_, t_->batchRows * fpRowElems_ * sizeof(float));
        }
        pipe_.InitBuffer(remainderFp_, tileFpElems_ * sizeof(float));
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, t_->broadcastTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
    }

    __aicore__ inline void Process()
    {
        if ((t_->reuseLayout != FLOOR_MOD_REUSE_COMPACT_ROW_BATCH &&
             t_->reuseLayout != FLOOR_MOD_REUSE_COMPACT_ROW_REPEAT) ||
            GetBlockIdx() >= t_->coreNum || width_ == 0U || t_->e1 == 0U || t_->batchRows == 0U) {
            return;
        }

        uint64_t row = 0U;
        uint64_t remainingRows = 0U;
        GetRowRange(row, remainingRows);
        if (remainingRows == 0U) {
            return;
        }
        CompactEvents events;
        InitEvents(events);
        const auto seed = PrepareSeed(events.seedReady);
        ProcessRowBatches(row, remainingRows, seed, events);
        ReleaseEvents(events);
    }

private:
    struct CompactEvents {
        event_t seedReady;
        event_t inputReady[2];
        event_t outputReady;
        event_t slotFree[2];
    };

    __aicore__ inline void GetRowRange(uint64_t& row, uint64_t& count) const
    {
        const uint64_t part = GetBlockIdx();
        const uint64_t rowsBase = t_->e1 / t_->rowPartitions;
        const uint64_t rowsExtra = t_->e1 % t_->rowPartitions;
        row = rowsBase * part + (part < rowsExtra ? part : rowsExtra);
        count = rowsBase + (part < rowsExtra ? 1U : 0U);
    }

    __aicore__ inline void InitEvents(CompactEvents& events)
    {
        events.seedReady = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        events.inputReady[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        events.inputReady[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        events.outputReady = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        events.slotFree[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        events.slotFree[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        SetFlag<HardEvent::MTE3_MTE2>(events.slotFree[0]);
        SetFlag<HardEvent::MTE3_MTE2>(events.slotFree[1]);
    }

    __aicore__ inline void ReleaseEvents(const CompactEvents& events)
    {
        WaitFlag<HardEvent::MTE3_MTE2>(events.slotFree[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(events.slotFree[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(events.seedReady);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(events.inputReady[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(events.inputReady[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(events.outputReady);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(events.slotFree[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(events.slotFree[1]);
    }

    __aicore__ inline LocalTensor<float> PrepareSeed(event_t seedReady)
    {
        LoadSeed(seedReady);
        WaitFlag<HardEvent::MTE2_V>(seedReady);
        if constexpr (std::is_same_v<T, float>) {
            return seedIn_.Get<T>().template ReinterpretCast<float>();
        }
        auto seed = seedFp_.Get<float>();
        Duplicate(seed, 0.0f, seedFpElems_);
        PipeBarrier<PIPE_V>();
        Cast(seed, seedIn_.Get<T>(), AscendC::RoundMode::CAST_NONE, width_);
        PipeBarrier<PIPE_V>();
        return seed;
    }

    __aicore__ inline LocalTensor<float> ExpandSeed(LocalTensor<float> seed, uint32_t rows)
    {
        auto expanded = expandedSeedFp_.Get<float>();
        const uint32_t srcShape[2] = {1U, fpRowElems_};
        const uint32_t dstShape[2] = {rows, fpRowElems_};
        if (t_->broadcastTmpBytes == 0U) {
            BroadCast<float, 2, 0>(expanded, seed, dstShape, srcShape);
        } else {
            auto sharedTmp = floorTmp_.Get<uint8_t>();
            BroadCast<float, 2, 0>(expanded, seed, dstShape, srcShape, sharedTmp);
        }
        PipeBarrier<PIPE_V>();
        if (fpRowElems_ != width_) {
            GatherMaskParams params{1U, static_cast<uint16_t>(rows),
                                    static_cast<uint16_t>(fpRowElems_ / FLOOR_MOD_FP32_BLOCK_ELEMS), 0U};
            uint64_t gathered = 0U;
            GatherMask(expanded, expanded, static_cast<uint8_t>(7U), true, width_, params, gathered);
            PipeBarrier<PIPE_V>();
        }
        return expanded;
    }

    __aicore__ inline uint32_t PrefetchNext(uint32_t slot, uint64_t row, uint64_t& remainingRows,
                                            const CompactEvents& events)
    {
        if (remainingRows == 0U) {
            return 0U;
        }
        const uint32_t rows = MinU32(remainingRows, t_->batchRows);
        WaitFlag<HardEvent::MTE3_MTE2>(events.slotFree[slot]);
        LoadDense(slot, row, rows);
        SetFlag<HardEvent::MTE2_V>(events.inputReady[slot]);
        remainingRows -= rows;
        return rows;
    }

    __aicore__ inline void ProcessRowBatches(uint64_t row, uint64_t remainingRows, LocalTensor<float> seed,
                                             const CompactEvents& events)
    {
        uint32_t currentSlot = 0U;
        uint32_t currentRows = PrefetchNext(currentSlot, row, remainingRows, events);
        while (currentRows > 0U) {
            WaitFlag<HardEvent::MTE2_V>(events.inputReady[currentSlot]);
            auto dense = denseIn_.Get<T>()[currentSlot * tileTElems_];
            LocalTensor<float> expanded;
            if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_ROW_BATCH) {
                expanded = ExpandSeed(seed, currentRows);
            }
            const uint32_t nextSlot = currentSlot ^ 1U;
            const uint64_t nextRow = row + currentRows;
            const uint32_t nextRows = PrefetchNext(nextSlot, nextRow, remainingRows, events);
            if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_ROW_BATCH) {
                ComputeMaterialized(dense, expanded, currentRows);
            } else {
                ComputeRepeated(dense, seed, currentRows);
            }
            SetFlag<HardEvent::V_MTE3>(events.outputReady);
            WaitFlag<HardEvent::V_MTE3>(events.outputReady);
            StoreDense(dense, row, currentRows);
            SetFlag<HardEvent::MTE3_MTE2>(events.slotFree[currentSlot]);
            row = nextRow;
            currentRows = nextRows;
            currentSlot = nextSlot;
        }
    }

    __aicore__ static inline uint32_t MinU32(uint64_t value, uint32_t limit)
    {
        return static_cast<uint32_t>(value < limit ? value : limit);
    }

    __aicore__ static inline uint32_t MaxU32(uint32_t a, uint32_t b, uint32_t c)
    {
        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }

    __aicore__ static inline uint32_t AlignElements(uint32_t elements)
    {
        constexpr uint32_t alignment = FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T) > 8U ?
                                           FLOOR_MOD_UB_BLOCK_BYTES / sizeof(T) :
                                           8U;
        return (elements + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline void LoadSeed(event_t ready)
    {
        GlobalTensor<T> seedGm = t_->seedIsX1 != 0U ? x1Gm_ : x2Gm_;
        DataCopyExtParams params{1U, width_ * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(seedIn_.Get<T>(), seedGm, params, pad);
        SetFlag<HardEvent::MTE2_V>(ready);
    }

    __aicore__ inline void LoadDense(uint32_t slot, uint64_t row, uint32_t rows)
    {
        GlobalTensor<T> denseGm = t_->seedIsX1 != 0U ? x2Gm_ : x1Gm_;
        if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_ROW_BATCH) {
            const uint32_t compactBytes = rows * width_ * static_cast<uint32_t>(sizeof(T));
            DataCopyExtParams params{1U, compactBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
            DataCopyPad(denseIn_.Get<T>()[slot * tileTElems_], denseGm[row * width_], params, pad);
            return;
        }

        const uint32_t blockBytes = width_ * static_cast<uint32_t>(sizeof(T));
        const uint32_t rowBlocks = tRowElems_ * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t validBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(rows), blockBytes, 0U, rowBlocks - validBlocks, 0U};
        DataCopyPadExtParams<T> pad{true, 0U, static_cast<uint8_t>(tRowElems_ - width_), static_cast<T>(0)};
        DataCopyPad(denseIn_.Get<T>()[slot * tileTElems_], denseGm[row * width_], params, pad);
    }

    __aicore__ inline void CastRowsToFp32(LocalTensor<float> dst, LocalTensor<T> src, uint32_t rows)
    {
        constexpr uint32_t castMask = 256U / (sizeof(T) > sizeof(float) ? sizeof(T) : sizeof(float));
        const uint8_t dstRepStride = static_cast<uint8_t>(fpRowElems_ / 8U);
        const uint8_t srcRepStride = static_cast<uint8_t>(tRowElems_ * sizeof(T) / FLOOR_MOD_UB_BLOCK_BYTES);
        const UnaryRepeatParams params(1U, 1U, dstRepStride, srcRepStride);
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += FLOOR_MOD_MAX_VECTOR_REPEATS) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(rows - rowBase, FLOOR_MOD_MAX_VECTOR_REPEATS));
            for (uint32_t segment = 0U; segment < width_; segment += castMask) {
                const uint32_t count = MinU32(width_ - segment, castMask);
                Cast(dst[rowBase * fpRowElems_ + segment], src[rowBase * tRowElems_ + segment],
                     AscendC::RoundMode::CAST_NONE, static_cast<uint64_t>(count), repeats, params);
            }
        }
    }

    __aicore__ inline void CastRowsFromFp32(LocalTensor<T> dst, LocalTensor<float> src, uint32_t rows)
    {
        constexpr uint32_t castMask = 256U / (sizeof(T) > sizeof(float) ? sizeof(T) : sizeof(float));
        const uint8_t dstRepStride = static_cast<uint8_t>(tRowElems_ * sizeof(T) / FLOOR_MOD_UB_BLOCK_BYTES);
        const uint8_t srcRepStride = static_cast<uint8_t>(fpRowElems_ / 8U);
        const UnaryRepeatParams params(1U, 1U, dstRepStride, srcRepStride);
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += FLOOR_MOD_MAX_VECTOR_REPEATS) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(rows - rowBase, FLOOR_MOD_MAX_VECTOR_REPEATS));
            for (uint32_t segment = 0U; segment < width_; segment += castMask) {
                const uint32_t count = MinU32(width_ - segment, castMask);
                if constexpr (std::is_same_v<T, half>) {
                    Cast(dst[rowBase * tRowElems_ + segment], src[rowBase * fpRowElems_ + segment],
                         AscendC::RoundMode::CAST_NONE, static_cast<uint64_t>(count), repeats, params);
                } else {
                    Cast(dst[rowBase * tRowElems_ + segment], src[rowBase * fpRowElems_ + segment],
                         AscendC::RoundMode::CAST_RINT, static_cast<uint64_t>(count), repeats, params);
                }
            }
        }
    }

    __aicore__ inline LocalTensor<float> PrepareDense(LocalTensor<T> denseT, uint32_t count, uint32_t rows)
    {
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = denseT.template ReinterpretCast<float>();
        } else {
            dense = denseFp_.Get<float>();
            Duplicate(dense, 0.0f, count);
            PipeBarrier<PIPE_V>();
            CastRowsToFp32(dense, denseT, rows);
            PipeBarrier<PIPE_V>();
        }
        return dense;
    }

    __aicore__ inline void ComputeMaterialized(LocalTensor<T> denseT, LocalTensor<float> seed, uint32_t rows)
    {
        const uint32_t count = rows * width_;
        LocalTensor<float> dense;
        if constexpr (std::is_same_v<T, float>) {
            dense = denseT.template ReinterpretCast<float>();
        } else {
            dense = denseFp_.Get<float>();
            Cast(dense, denseT, AscendC::RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        auto quotient = remainderFp_.Get<float>();
        ComputeFloorRemainder(quotient, dense, seed, floorTmp_.Get<uint8_t>(), count, t_->swapped != 0U);
        if constexpr (!std::is_same_v<T, float>) {
            if constexpr (std::is_same_v<T, half>) {
                Cast(denseT, dense, AscendC::RoundMode::CAST_NONE, count);
            } else {
                Cast(denseT, dense, AscendC::RoundMode::CAST_RINT, count);
            }
        }
    }

    __aicore__ inline void ComputeRepeated(LocalTensor<T> denseT, LocalTensor<float> seed, uint32_t rows)
    {
        const uint32_t count = rows * fpRowElems_;
        auto dense = PrepareDense(denseT, count, rows);
        auto quotient = remainderFp_.Get<float>();
        // Floor runs contiguously over the aligned slab. Initialize padding so
        // it never consumes undefined data, then overwrite every logical lane
        // through repeated row instructions that reuse seed with stride zero.
        Duplicate(quotient, 0.0f, count);
        PipeBarrier<PIPE_V>();
        ComputeRepeatedQuotient(quotient, dense, seed, rows);
        Floor(quotient, quotient, floorTmp_.Get<uint8_t>(), count);
        FinishRepeatedRemainder(quotient, dense, seed, rows);
        if constexpr (!std::is_same_v<T, float>) {
            CastRowsFromFp32(denseT, dense, rows);
        }
    }

    __aicore__ inline void ComputeRepeatedQuotient(LocalTensor<float> quotient, LocalTensor<float> dense,
                                                   LocalTensor<float> seed, uint32_t rows)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(fpRowElems_ / FLOOR_MOD_FP32_BLOCK_ELEMS);
        const BinaryRepeatParams normal(1U, 1U, 1U, rowBlocks, rowBlocks, 0U);
        const BinaryRepeatParams swapped(1U, 1U, 1U, rowBlocks, 0U, rowBlocks);
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += FLOOR_MOD_MAX_VECTOR_REPEATS) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(rows - rowBase, FLOOR_MOD_MAX_VECTOR_REPEATS));
            const uint32_t base = rowBase * fpRowElems_;
            for (uint32_t segment = 0U; segment < width_; segment += FLOOR_MOD_VECTOR_SEGMENT) {
                const uint32_t valid = MinU32(width_ - segment, FLOOR_MOD_VECTOR_SEGMENT);
                if (t_->swapped != 0U) {
                    Div(quotient[base + segment], seed[segment], dense[base + segment], static_cast<uint64_t>(valid),
                        repeats, swapped);
                } else {
                    Div(quotient[base + segment], dense[base + segment], seed[segment], static_cast<uint64_t>(valid),
                        repeats, normal);
                }
            }
        }
    }

    __aicore__ inline void FinishRepeatedRemainder(LocalTensor<float> quotient, LocalTensor<float> dense,
                                                   LocalTensor<float> seed, uint32_t rows)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(fpRowElems_ / FLOOR_MOD_FP32_BLOCK_ELEMS);
        const BinaryRepeatParams normal(1U, 1U, 1U, rowBlocks, rowBlocks, 0U);
        const BinaryRepeatParams swapped(1U, 1U, 1U, rowBlocks, 0U, rowBlocks);
        const BinaryRepeatParams bothRows(1U, 1U, 1U, rowBlocks, rowBlocks, rowBlocks);
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += FLOOR_MOD_MAX_VECTOR_REPEATS) {
            const uint8_t repeats = static_cast<uint8_t>(MinU32(rows - rowBase, FLOOR_MOD_MAX_VECTOR_REPEATS));
            const uint32_t base = rowBase * fpRowElems_;
            for (uint32_t segment = 0U; segment < width_; segment += FLOOR_MOD_VECTOR_SEGMENT) {
                const uint32_t valid = MinU32(width_ - segment, FLOOR_MOD_VECTOR_SEGMENT);
                if (t_->swapped != 0U) {
                    Mul(quotient[base + segment], quotient[base + segment], dense[base + segment],
                        static_cast<uint64_t>(valid), repeats, bothRows);
                    Sub(dense[base + segment], seed[segment], quotient[base + segment], static_cast<uint64_t>(valid),
                        repeats, swapped);
                } else {
                    Mul(quotient[base + segment], quotient[base + segment], seed[segment], static_cast<uint64_t>(valid),
                        repeats, normal);
                    Sub(dense[base + segment], dense[base + segment], quotient[base + segment],
                        static_cast<uint64_t>(valid), repeats, bothRows);
                }
            }
        }
    }

    __aicore__ inline void StoreDense(LocalTensor<T> denseT, uint64_t row, uint32_t rows)
    {
        if (t_->reuseLayout == FLOOR_MOD_REUSE_COMPACT_ROW_BATCH) {
            const uint32_t compactBytes = rows * width_ * static_cast<uint32_t>(sizeof(T));
            DataCopyExtParams params{1U, compactBytes, 0U, 0U, 0U};
            DataCopyPad(yGm_[row * width_], denseT, params);
            return;
        }
        const uint32_t blockBytes = width_ * static_cast<uint32_t>(sizeof(T));
        const uint32_t rowBlocks = tRowElems_ * static_cast<uint32_t>(sizeof(T)) / FLOOR_MOD_UB_BLOCK_BYTES;
        const uint32_t validBlocks = (blockBytes + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(rows), blockBytes, rowBlocks - validBlocks, 0U, 0U};
        DataCopyPad(yGm_[row * width_], denseT, params);
    }

    const FloorModTilingData* t_ = nullptr;
    TPipe pipe_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    TBuf<QuePosition::VECIN> seedIn_;
    TBuf<QuePosition::VECIN> denseIn_;
    TBuf<TPosition::VECCALC> seedFp_;
    TBuf<TPosition::VECCALC> denseFp_;
    TBuf<TPosition::VECCALC> expandedSeedFp_;
    TBuf<TPosition::VECCALC> remainderFp_;
    TBuf<TPosition::VECCALC> floorTmp_;
    uint32_t width_ = 0U;
    uint32_t seedTElems_ = 0U;
    uint32_t seedFpElems_ = 0U;
    uint32_t tRowElems_ = 0U;
    uint32_t fpRowElems_ = 0U;
    uint32_t tileTElems_ = 0U;
    uint32_t tileFpElems_ = 0U;
};

__aicore__ static inline uint32_t RoundDoubleSubnormalToFloatFast(uint64_t significand, int32_t unbiasedExp)
{
    const uint32_t shift = static_cast<uint32_t>(-unbiasedExp - 97);
    uint64_t mantissa = significand >> shift;
    const uint64_t remainder = significand & ((1ULL << shift) - 1ULL);
    const uint64_t halfway = 1ULL << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (mantissa & 1ULL) != 0ULL)) {
        ++mantissa;
    }
    return static_cast<uint32_t>(mantissa);
}

__aicore__ static inline uint32_t RoundDoubleNormalToFloatFast(uint64_t significand, int32_t& unbiasedExp)
{
    uint64_t mantissa = significand >> 29U;
    const uint64_t remainder = significand & ((1ULL << 29U) - 1ULL);
    const uint64_t halfway = 1ULL << 28U;
    if (remainder > halfway || (remainder == halfway && (mantissa & 1ULL) != 0ULL)) {
        ++mantissa;
    }
    if (mantissa == (1ULL << 24U)) {
        mantissa >>= 1U;
        ++unbiasedExp;
    }
    if (unbiasedExp > 127) {
        return 0x7f800000U;
    }
    const uint32_t exponent = static_cast<uint32_t>(unbiasedExp + 127);
    return (exponent << 23U) | (static_cast<uint32_t>(mantissa) & 0x7fffffU);
}

__aicore__ static inline uint32_t ConvertFiniteDoubleBitsToFloatFast(uint32_t sign, uint32_t exp, uint64_t frac)
{
    int32_t unbiasedExp = static_cast<int32_t>(exp) - 1023;
    const uint64_t significand = (1ULL << 52U) | frac;
    if (unbiasedExp > 127) {
        return sign | 0x7f800000U;
    }
    if (unbiasedExp < -150) {
        return sign;
    }
    if (unbiasedExp < -126) {
        return sign | RoundDoubleSubnormalToFloatFast(significand, unbiasedExp);
    }
    return sign | RoundDoubleNormalToFloatFast(significand, unbiasedExp);
}

__aicore__ static inline float DoubleBitsToFloatFast(uint64_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits >> 63U) << 31U;
    const uint32_t exp = static_cast<uint32_t>((bits >> 52U) & 0x7ffU);
    const uint64_t frac = bits & 0x000fffffffffffffULL;
    union Bits {
        uint32_t u;
        float f;
    } out;
    out.u = exp == 0U ? sign : ConvertFiniteDoubleBitsToFloatFast(sign, exp, frac);
    if (exp == 0x7ffU) {
        out.u = sign | 0x7f800000U | (frac == 0U ? 0U : static_cast<uint32_t>(frac >> 29U) | 1U);
    }
    return out.f;
}

__aicore__ static inline uint64_t FloatToDoubleBitsFast(float value)
{
    union Bits {
        float f;
        uint32_t u;
    } in;
    in.f = value;
    const uint64_t sign = static_cast<uint64_t>(in.u >> 31U) << 63U;
    const uint32_t exp = (in.u >> 23U) & 0xffU;
    const uint32_t frac = in.u & 0x7fffffU;
    if (exp == 0U) {
        return sign;
    }
    if (exp == 0xffU) {
        return sign | (0x7ffULL << 52U) | (static_cast<uint64_t>(frac) << 29U);
    }
    return sign | (static_cast<uint64_t>(exp - 127U + 1023U) << 52U) | (static_cast<uint64_t>(frac) << 29U);
}

constexpr uint64_t VECTOR_MASK_U16 = 128ULL;
constexpr uint64_t VECTOR_MASK_U32 = 64ULL;
constexpr uint32_t VECTOR_REPEAT_STRIDE = 8U;

__aicore__ inline void VectorAnd32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src0, LocalTensor<int32_t> src1,
                                   uint32_t count)
{
    const uint32_t words = count * 2U;
    const uint8_t repeat = static_cast<uint8_t>((words + 127U) / 128U);
    const BinaryRepeatParams params{1U, 1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    And(dst.ReinterpretCast<uint16_t>(), src0.ReinterpretCast<uint16_t>(), src1.ReinterpretCast<uint16_t>(),
        VECTOR_MASK_U16, repeat, params);
}

__aicore__ inline void VectorOr32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src0, LocalTensor<int32_t> src1,
                                  uint32_t count)
{
    const uint32_t words = count * 2U;
    const uint8_t repeat = static_cast<uint8_t>((words + 127U) / 128U);
    const BinaryRepeatParams params{1U, 1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    Or(dst.ReinterpretCast<uint16_t>(), src0.ReinterpretCast<uint16_t>(), src1.ReinterpretCast<uint16_t>(),
       VECTOR_MASK_U16, repeat, params);
}

__aicore__ inline void VectorShiftLeft32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, uint32_t shift,
                                         uint32_t count)
{
    const uint32_t repeat = (count + 63U) / 64U;
    const UnaryRepeatParams params{1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    ShiftLeft(dst.ReinterpretCast<int32_t>(), src.ReinterpretCast<int32_t>(), static_cast<int32_t>(shift),
              VECTOR_MASK_U32, static_cast<uint8_t>(repeat), params);
}

__aicore__ inline void VectorShiftRight32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, uint32_t shift,
                                          uint32_t count)
{
    const uint32_t repeat = (count + 63U) / 64U;
    const UnaryRepeatParams params{1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    ShiftRight(dst.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), shift, VECTOR_MASK_U32,
               static_cast<uint8_t>(repeat), params);
}

__aicore__ inline void VectorAdd32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src0, LocalTensor<int32_t> src1,
                                   uint32_t count)
{
    const uint32_t repeat = (count + 63U) / 64U;
    const BinaryRepeatParams params{1U, 1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    Add(dst, src0, src1, VECTOR_MASK_U32, static_cast<uint8_t>(repeat), params);
}

__aicore__ inline void VectorAddScalar32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t value,
                                         uint32_t count)
{
    const uint32_t repeat = (count + 63U) / 64U;
    const UnaryRepeatParams params{1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    Adds(dst, src, value, VECTOR_MASK_U32, static_cast<uint8_t>(repeat), params);
}

__aicore__ inline void VectorAndScalar32(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t value,
                                         LocalTensor<int32_t> scratch, uint32_t count)
{
    Duplicate(scratch, value, count);
    VectorAnd32(dst, src, scratch, count);
}

__aicore__ inline void VectorOrMask(LocalTensor<uint8_t> dst, LocalTensor<uint8_t> src, uint32_t maskBytes)
{
    const uint32_t words = (maskBytes + 1U) / 2U;
    const uint8_t repeat = static_cast<uint8_t>((words + 127U) / 128U);
    const BinaryRepeatParams params{1U, 1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    Or(dst.ReinterpretCast<uint16_t>(), dst.ReinterpretCast<uint16_t>(), src.ReinterpretCast<uint16_t>(),
       VECTOR_MASK_U16, repeat, params);
}

__aicore__ inline void ConvertDoubleWordsToFloat(LocalTensor<int32_t> value, LocalTensor<int32_t> low,
                                                 LocalTensor<int32_t> scratch0, LocalTensor<int32_t> scratch1,
                                                 LocalTensor<int32_t> scratch2, uint32_t count)
{
    // value initially contains the high word, low the low word. The result is
    // the correctly rounded binary32 bit pattern (RNE), matching a hardware
    // double->float cast for normal values.
    //
    // Keep the low 20 fraction bits and append the top three bits of the low
    // word. The remaining 29 low bits decide rounding.
    VectorShiftLeft32(scratch0, value, 12U, count);
    VectorShiftRight32(scratch0, scratch0, 9U, count);
    VectorShiftRight32(scratch1, low, 29U, count);
    VectorOr32(scratch0, scratch0, scratch1, count);

    // sticky = (low & 0x0fffffff) != 0.
    VectorShiftLeft32(scratch1, low, 4U, count);
    VectorShiftRight32(scratch1, scratch1, 4U, count);
    Mins(scratch1, scratch1, 1, count);

    // guard = bit 28 of the low word.
    VectorShiftLeft32(scratch2, low, 3U, count);
    VectorShiftRight32(scratch2, scratch2, 31U, count);

    // Reuse low as the retained mantissa's least-significant bit.
    VectorShiftLeft32(low, scratch0, 31U, count);
    VectorShiftRight32(low, low, 31U, count);

    // round = guard && (sticky || lsb).
    VectorOr32(scratch1, scratch1, low, count);
    VectorAnd32(scratch2, scratch2, scratch1, count);
    VectorAdd32(scratch0, scratch0, scratch2, count);

    // Mantissa overflow is carried into the exponent.
    VectorShiftRight32(scratch1, scratch0, 23U, count);
    VectorShiftLeft32(scratch0, scratch0, 9U, count);
    VectorShiftRight32(scratch0, scratch0, 9U, count);

    VectorShiftRight32(scratch2, value, 20U, count);
    VectorShiftLeft32(scratch2, scratch2, 21U, count);
    VectorShiftRight32(scratch2, scratch2, 21U, count);
    VectorAddScalar32(scratch2, scratch2, -896, count);
    // Clamp zero/subnormal and overflow/Inf/NaN exponent fields. The result
    // is then classified by the existing input/result safety checks.
    Mins(scratch2, scratch2, 255, count);
    Maxs(scratch2, scratch2, 0, count);
    VectorAdd32(scratch2, scratch2, scratch1, count);
    VectorShiftLeft32(scratch2, scratch2, 23U, count);

    VectorShiftRight32(low, value, 31U, count);
    VectorShiftLeft32(low, low, 31U, count);

    VectorOr32(value, scratch0, scratch2, count);
    VectorOr32(value, value, low, count);
}

__aicore__ inline void ConvertFloatWordsToDouble(LocalTensor<int32_t> value, LocalTensor<int32_t> low,
                                                 LocalTensor<int32_t> high, LocalTensor<int32_t> scratch0,
                                                 LocalTensor<int32_t> scratch1, uint32_t count)
{
    VectorAndScalar32(low, value, 0x007fffff, scratch0, count);
    VectorShiftLeft32(low, low, 29U, count);
    VectorAndScalar32(high, value, 0x7f800000, scratch0, count);
    VectorShiftRight32(high, high, 3U, count);
    VectorAddScalar32(high, high, static_cast<int32_t>(896U << 20U), count);
    VectorAndScalar32(scratch0, value, 0x007fffff, scratch1, count);
    VectorShiftRight32(scratch0, scratch0, 3U, count);
    VectorOr32(high, high, scratch0, count);
    VectorAndScalar32(scratch1, value, static_cast<int32_t>(0x80000000U), scratch0, count);
    VectorOr32(high, high, scratch1, count);
}

__aicore__ inline void PackDoubleWords(LocalTensor<uint64_t> out, LocalTensor<uint64_t> combined,
                                       LocalTensor<int32_t> low, LocalTensor<int32_t> high, LocalTensor<int32_t> index,
                                       uint32_t count, uint32_t packedStride)
{
    auto outWords = out.ReinterpretCast<int32_t>();
    auto combinedWords = combined.ReinterpretCast<int32_t>();
    const uint32_t pairCount = count * 2U;

    VectorAddScalar32(combinedWords, low, 0, count);
    VectorAddScalar32(combinedWords[packedStride], high, 0, count);

    // Byte offsets are initialized once and remain valid for partial tiles.
    Gather(outWords.ReinterpretCast<uint32_t>(), combinedWords.ReinterpretCast<uint32_t>(),
           index.ReinterpretCast<uint32_t>(), 0U, pairCount);
}

__aicore__ inline void VectorNotMask(LocalTensor<uint8_t> mask, uint32_t maskBytes)
{
    const uint32_t words = (maskBytes + 1U) / 2U;
    const uint8_t repeat = static_cast<uint8_t>((words + 127U) / 128U);
    const UnaryRepeatParams params{1U, 1U, VECTOR_REPEAT_STRIDE, VECTOR_REPEAT_STRIDE};
    Not(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), VECTOR_MASK_U16, repeat, params);
}

// Mark inputs for which the bounded FP32 approximation cannot guarantee the
// required accuracy.
__aicore__ inline void MarkUnsafeInputs(LocalTensor<float> dividend, LocalTensor<float> divisor,
                                        LocalTensor<int32_t> scratch, LocalTensor<uint8_t> unsafeMask,
                                        LocalTensor<uint8_t> tmpMask, uint32_t count, uint32_t maskBytes)
{
    constexpr float kMaxSafeMagnitude = 128.0f;
    constexpr float kMinNormalMagnitude = 1.1754943508222875e-38f;
    auto scratchFloat = scratch.ReinterpretCast<float>();
    const uint32_t vectorCount = (count + 63U) & ~63U;

    Abs(scratchFloat, dividend, count);
    // Invert the ordered <= predicate: NaNs must be unsafe too. An ordered
    // NE self-comparison does not reliably classify NaNs on this architecture.
    Compares(tmpMask, scratchFloat, kMaxSafeMagnitude, CMPMODE::LE, vectorCount);
    VectorNotMask(tmpMask, maskBytes);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);
    Compares(tmpMask, scratchFloat, kMinNormalMagnitude, CMPMODE::LT, vectorCount);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);

    Abs(scratchFloat, divisor, count);
    Compares(tmpMask, scratchFloat, 1.0f, CMPMODE::LT, vectorCount);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);
    // Invert the ordered <= predicate: NaNs must be unsafe too. An ordered
    // NE self-comparison does not reliably classify NaNs on this architecture.
    Compares(tmpMask, scratchFloat, kMaxSafeMagnitude, CMPMODE::LE, vectorCount);
    VectorNotMask(tmpMask, maskBytes);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);
}

// A quotient too close to an integer can make floor() select the adjacent
// quotient after the binary64->binary32 conversion. Subnormal results can also
// vanish in the FP32 subtraction, so both cases use the exact significand path.
__aicore__ inline void MarkUnsafeResultAndQuotient(LocalTensor<float> value, LocalTensor<float> quotient,
                                                   LocalTensor<float> fraction, LocalTensor<int32_t> scratch,
                                                   LocalTensor<uint8_t> unsafeMask, LocalTensor<uint8_t> tmpMask,
                                                   uint32_t count, uint32_t maskBytes)
{
    constexpr float kMinNormalMagnitude = 1.1754943508222875e-38f;
    constexpr float kBoundaryEpsilon = 1.0e-4f;
    constexpr float kBoundaryGuard = 1.0f - kBoundaryEpsilon;
    auto scratchFloat = scratch.ReinterpretCast<float>();
    const uint32_t vectorCount = (count + 63U) & ~63U;

    Abs(scratchFloat, value, count);
    Compares(tmpMask, scratchFloat, kMinNormalMagnitude, CMPMODE::LT, vectorCount);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);
    // Inputs accepted by MarkUnsafeInputs are finite, |x| <= 128 and
    // 1 <= |y| <= 128, so the quotient and subtraction cannot overflow.
    Compares(tmpMask, fraction, kBoundaryEpsilon, CMPMODE::LE, vectorCount);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);
    Compares(tmpMask, fraction, kBoundaryGuard, CMPMODE::GE, vectorCount);
    VectorOrMask(unsafeMask, tmpMask, maskBytes);
}

// FP64 storage path for DAV C220. The architecture has no FP64 vector
// arithmetic or vector Cast, so double values stay as uint64 bit patterns in
// GM/UB. Ordinary values use the vectorized FP32 chain. Values that are
// outside the precision-safe range, or whose quotient is too close to an
// integer boundary, are recomputed from the exact IEEE-754 significands.
// This keeps the fast path while covering the precision gaps of the FP32
// approximation without an AICPU fallback.
class FloorModFp64Storage {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const FloorModTilingData* tiling)
    {
        t_ = tiling;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(y));
        yWordsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y));
        tile_ = t_->denseTile == 0U ? 1U : t_->denseTile;
        aligned_ = t_->maxFp32RowElems == 0U ? 64U : ((t_->maxFp32RowElems + 63U) & ~63U);
        pipe_.InitBuffer(x1In_, aligned_ * sizeof(uint64_t));
        pipe_.InitBuffer(x2In_, aligned_ * sizeof(uint64_t));
        pipe_.InitBuffer(aFp_, aligned_ * sizeof(float));
        pipe_.InitBuffer(bFp_, aligned_ * sizeof(float));
        pipe_.InitBuffer(remFp_, aligned_ * sizeof(float));
        const uint32_t scratchBytes = MaxU32(aligned_ * sizeof(int32_t), 256U);
        pipe_.InitBuffer(scratch0_, scratchBytes);
        pipe_.InitBuffer(scratch1_, scratchBytes);
        pipe_.InitBuffer(scratch2_, scratchBytes);
        pipe_.InitBuffer(index_, 2U * aligned_ * sizeof(int32_t));
        maskStride_ = MaxU32(((aligned_ + 7U) / 8U + FLOOR_MOD_UB_BLOCK_BYTES - 1U) / FLOOR_MOD_UB_BLOCK_BYTES *
                                 FLOOR_MOD_UB_BLOCK_BYTES,
                             256U);
        pipe_.InitBuffer(mask_, 2U * maskStride_);
        pipe_.InitBuffer(floorTmp_, MaxU32(t_->floorTmpBytes, FLOOR_MOD_UB_BLOCK_BYTES));
        // A fixed plane stride lets every tile reuse the same interleave map,
        // including short tiles at core and broadcast boundaries.
        auto index = index_.Get<int32_t>();
        auto odd = x1In_.Get<int32_t>();
        const uint32_t pairCount = 2U * aligned_;
        CreateVecIndex(index, static_cast<int32_t>(0), pairCount);
        VectorShiftLeft32(odd, index, 31U, pairCount);
        VectorShiftRight32(odd, odd, 31U, pairCount);
        VectorShiftRight32(index, index, 1U, pairCount);
        Muls(odd, odd, static_cast<int32_t>(aligned_), pairCount);
        VectorAdd32(index, index, odd, pairCount);
        Muls(index, index, 4, pairCount);
        const event_t mapReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(mapReady);
        WaitFlag<HardEvent::V_MTE2>(mapReady);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= t_->coreNum || t_->totalElements == 0U) {
            return;
        }
        const uint64_t block = GetBlockIdx();
        const uint64_t base = t_->totalElements / t_->coreNum;
        const uint64_t extra = t_->totalElements % t_->coreNum;
        uint64_t offset = block * base + (block < extra ? block : extra);
        const uint64_t end = offset + base + (block < extra ? 1U : 0U);
        const bool x2TrailingBroadcast = t_->x2Elements < t_->totalElements && HasTrailingBroadcast(t_->x2Stride);
        ProcessRange(offset, end, x2TrailingBroadcast);
    }

private:
    // Expand a trailing-broadcast source into float lanes. Each broadcast run
    // has one source value, so only the run boundaries are scalar; the bulk is
    // filled with vector Duplicate. Scalar head/tail lanes are written first
    // and synchronized once before the vector middle sections.
    __aicore__ inline void BroadcastRunsToFloat(LocalTensor<float> dst, GlobalTensor<uint64_t> src, uint64_t elements,
                                                uint64_t offset, uint32_t count, const uint64_t* strides)
    {
        uint32_t local = 0U;
        while (local < count) {
            const uint64_t flat = offset + local;
            const uint64_t srcOffset = SourceOffset(flat, strides, elements);
            const uint32_t run = MinU32(TrailingBroadcastRun(flat, strides), count - local);
            const float value = DoubleBitsToFloatFast(src.GetValue(srcOffset));
            uint32_t head = 0U;
            while (head < run && ((local + head) & 7U) != 0U) {
                dst.SetValue(local + head, value);
                ++head;
            }
            const uint32_t vectorRun = (run - head) & ~7U;
            uint32_t tail = head + vectorRun;
            while (tail < run) {
                dst.SetValue(local + tail, value);
                ++tail;
            }
            local += run;
        }

        const event_t scalarReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarReady);
        WaitFlag<HardEvent::S_V>(scalarReady);

        local = 0U;
        while (local < count) {
            const uint64_t flat = offset + local;
            const uint64_t srcOffset = SourceOffset(flat, strides, elements);
            const uint32_t run = MinU32(TrailingBroadcastRun(flat, strides), count - local);
            const float value = DoubleBitsToFloatFast(src.GetValue(srcOffset));
            uint32_t head = 0U;
            while (head < run && ((local + head) & 7U) != 0U) {
                ++head;
            }
            const uint32_t vectorRun = (run - head) & ~7U;
            if (vectorRun != 0U) {
                Duplicate(dst[local + head], value, vectorRun);
            }
            local += run;
        }
    }

    __aicore__ inline void ProcessRange(uint64_t offset, uint64_t end, bool x2TrailingBroadcast)
    {
        while (offset < end) {
            uint32_t count = static_cast<uint32_t>(MinU64(end - offset, tile_));
            if (t_->x1Elements != t_->totalElements && !HasTrailingBroadcast(t_->x1Stride)) {
                count = MinU32(count, ContiguousSourceRun(offset, t_->x1Stride));
            }
            if (t_->x2Elements != t_->totalElements && !x2TrailingBroadcast) {
                count = MinU32(count, ContiguousSourceRun(offset, t_->x2Stride));
            }
            ProcessTile(offset, count, x2TrailingBroadcast);
            offset += count;
        }
    }

    __aicore__ inline void ConvertInputBits(LocalTensor<uint64_t> bits, LocalTensor<float> values, uint32_t count)
    {
        auto low = remFp_.Get<int32_t>();
        auto high = values.ReinterpretCast<int32_t>();
        const uint32_t words = count * 2U;
        const uint32_t mask = MinU32(words, 64U);
        const uint8_t repeats = static_cast<uint8_t>((words + 63U) / 64U);
        uint64_t gathered = 0U;
        GatherMask(low, bits.ReinterpretCast<int32_t>(), static_cast<uint8_t>(1U), true, mask, {1U, repeats, 8U, 0U},
                   gathered);
        GatherMask(high, bits.ReinterpretCast<int32_t>(), static_cast<uint8_t>(2U), true, mask, {1U, repeats, 8U, 0U},
                   gathered);
        ConvertDoubleWordsToFloat(high, low, scratch0_.Get<int32_t>(), scratch1_.Get<int32_t>(),
                                  scratch2_.Get<int32_t>(), count);
    }

    __aicore__ inline void LoadDoubleInputs(uint64_t offset, uint32_t count, bool x1Broadcast, bool x2Broadcast)
    {
        if (!x1Broadcast) {
            LoadBits(x1In_.Get<uint64_t>(), x1Gm_, t_->x1Elements, offset, count, t_->x1Stride);
        }
        if (!x2Broadcast) {
            LoadBits(x2In_.Get<uint64_t>(), x2Gm_, t_->x2Elements, offset, count, t_->x2Stride);
        }
        const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ready);
        WaitFlag<HardEvent::MTE2_S>(ready);
        if (x1Broadcast) {
            BroadcastRunsToFloat(aFp_.Get<float>(), x1Gm_, t_->x1Elements, offset, count, t_->x1Stride);
        } else {
            ConvertInputBits(x1In_.Get<uint64_t>(), aFp_.Get<float>(), count);
        }
        if (x2Broadcast) {
            BroadcastRunsToFloat(bFp_.Get<float>(), x2Gm_, t_->x2Elements, offset, count, t_->x2Stride);
        } else {
            ConvertInputBits(x2In_.Get<uint64_t>(), bFp_.Get<float>(), count);
        }
    }

    __aicore__ inline void ComputeDoubleApproximation(uint32_t count)
    {
        auto a = aFp_.Get<float>();
        auto b = bFp_.Get<float>();
        auto rem = remFp_.Get<float>();
        auto scratch0 = scratch0_.Get<int32_t>();
        auto scratch1 = scratch1_.Get<int32_t>();
        auto scratch2 = scratch2_.Get<int32_t>();
        auto unsafeMask = mask_.Get<uint8_t>();
        auto tmpMask = unsafeMask[maskStride_];
        const uint32_t maskBytes = (count + 7U) / 8U;
        Duplicate(unsafeMask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0U), (maskStride_ + 1U) / 2U);
        MarkUnsafeInputs(a, b, scratch2, unsafeMask, tmpMask, count, maskBytes);
        const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(ready);
        WaitFlag<HardEvent::S_V>(ready);
        Div(rem, a, b, count);
        auto quotient = scratch0.ReinterpretCast<float>();
        Floor(quotient, rem, floorTmp_.Get<uint8_t>(), count);
        auto fraction = scratch2.ReinterpretCast<float>();
        Sub(fraction, rem, quotient, count);
        Mul(rem, quotient, b, count);
        Sub(a, a, rem, count);
        MarkUnsafeResultAndQuotient(a, quotient, fraction, scratch1, unsafeMask, tmpMask, count, maskBytes);
        ConvertFloatWordsToDouble(a.ReinterpretCast<int32_t>(), rem.ReinterpretCast<int32_t>(),
                                  b.ReinterpretCast<int32_t>(), scratch0, scratch1, count);
    }

    __aicore__ inline void CorrectDoubleResults(uint64_t offset, uint32_t count, bool x1Broadcast, bool x2Broadcast)
    {
        const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);
        auto mask = mask_.Get<uint8_t>();
        auto low = remFp_.Get<int32_t>();
        auto high = bFp_.Get<int32_t>();
        for (uint32_t byte = 0U; byte < (count + 7U) / 8U; ++byte) {
            uint32_t bits = mask.GetValue(byte);
            while (bits != 0U) {
                const uint32_t index = byte * 8U + static_cast<uint32_t>(__builtin_ctz(bits));
                if (index < count) {
                    const uint64_t x = x1Broadcast ?
                                           x1Gm_.GetValue(SourceOffset(offset + index, t_->x1Stride, t_->x1Elements)) :
                                           x1In_.Get<uint64_t>().GetValue(index);
                    const uint64_t divisor = x2Broadcast ? x2Gm_.GetValue(SourceOffset(offset + index, t_->x2Stride,
                                                                                       t_->x2Elements)) :
                                                           x2In_.Get<uint64_t>().GetValue(index);
                    const uint64_t result = DoubleFloorModBits(x, divisor);
                    low.SetValue(index, static_cast<int32_t>(static_cast<uint32_t>(result)));
                    high.SetValue(index, static_cast<int32_t>(static_cast<uint32_t>(result >> 32U)));
                }
                bits &= bits - 1U;
            }
        }
    }

    __aicore__ inline void StoreDoubleResults(uint64_t offset, uint32_t count)
    {
        const event_t exactReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(exactReady);
        WaitFlag<HardEvent::S_V>(exactReady);
        auto output = x1In_.Get<uint64_t>();
        PackDoubleWords(output, x2In_.Get<uint64_t>(), remFp_.Get<int32_t>(), bFp_.Get<int32_t>(),
                        index_.Get<int32_t>(), count, aligned_);
        const event_t outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);
        DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(uint64_t)), 0U, 0U, 0U};
        DataCopyPad(yGm_[offset], output, params);
        const event_t outputDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(outputDone);
        WaitFlag<HardEvent::MTE3_S>(outputDone);
    }

    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count, bool x2TrailingBroadcast)
    {
        const bool x1TrailingBroadcast = t_->x1Elements < t_->totalElements && HasTrailingBroadcast(t_->x1Stride);
        LoadDoubleInputs(offset, count, x1TrailingBroadcast, x2TrailingBroadcast);
        ComputeDoubleApproximation(count);
        CorrectDoubleResults(offset, count, x1TrailingBroadcast, x2TrailingBroadcast);
        StoreDoubleResults(offset, count);
    }

    __aicore__ static inline uint64_t MinU64(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ static inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }
    __aicore__ static inline uint32_t MaxU32(uint32_t a, uint32_t b) { return a > b ? a : b; }

    __aicore__ inline bool HasTrailingBroadcast(const uint64_t* strides) const

    {
        return t_->rank != 0U && strides[t_->rank - 1U] == 0U;
    }

    // Return the number of consecutive flattened output elements that map to
    // the same source element when trailing output dimensions are broadcast.
    __aicore__ inline uint32_t TrailingBroadcastRun(uint64_t flat, const uint64_t* strides) const
    {
        uint64_t suffix = 1U;
        uint64_t run = 1U;
        bool started = false;
        for (uint32_t axis = t_->rank; axis-- > 0U;) {
            const uint64_t extent = t_->outShape[axis];
            if (extent == 0U || strides[axis] != 0U) {
                break;
            }
            const uint64_t coord = (flat / suffix) % extent;
            const uint64_t axisRemaining = extent - coord;
            run = started ? run * axisRemaining : axisRemaining;
            started = true;
            suffix *= extent;
        }
        return !started || run == 0U ? 1U : static_cast<uint32_t>(run);
    }

    // Return the number of flattened output elements whose source addresses are
    // consecutive. This turns leading-dimension and cross broadcasts into a few
    // ordinary DMA runs instead of a scalar GetValue/SetValue loop per element.
    __aicore__ inline uint32_t ContiguousSourceRun(uint64_t flat, const uint64_t* strides) const
    {
        uint64_t suffix = 1U;
        uint64_t run = 1U;
        bool started = false;
        for (uint32_t axis = t_->rank; axis-- > 0U;) {
            const uint64_t extent = t_->outShape[axis];
            if (extent == 0U || strides[axis] == 0U) {
                break;
            }
            if (axis + 1U == t_->rank) {
                if (strides[axis] != 1U) {
                    break;
                }
            } else if (strides[axis] != suffix) {
                break;
            }
            const uint64_t coord = (flat / suffix) % extent;
            if (!started) {
                run = (extent - coord) * suffix;
                started = true;
            } else {
                run += (extent - coord - 1U) * suffix;
            }
            suffix *= extent;
        }
        return !started || run == 0U ? 1U : static_cast<uint32_t>(run);
    }

    __aicore__ static inline uint64_t DoubleFromSignificand(uint64_t significand, int32_t exponent, uint64_t sign)
    {
        if (significand == 0U) {
            return sign << 63U;
        }
        const uint32_t length = 64U - static_cast<uint32_t>(__builtin_clzll(significand));
        const int32_t unbiasedExp = exponent + static_cast<int32_t>(length) - 1;
        if (unbiasedExp >= -1022) {
            significand <<= (53U - length);
            return (sign << 63U) | (static_cast<uint64_t>(unbiasedExp + 1023) << 52U) |
                   (significand & 0x000fffffffffffffULL);
        }
        const uint32_t shift = static_cast<uint32_t>(exponent + 1074);
        significand <<= shift;
        return (sign << 63U) | (significand & 0x000fffffffffffffULL);
    }

    // Round an exact integer represented by hi:lo * 2^exponent to binary64.
    __aicore__ static inline uint64_t DoubleFromRoundedInteger128(uint64_t hi, uint64_t lo, int32_t exponent,
                                                                  uint64_t sign)
    {
        const uint32_t length = hi != 0U ? 128U - static_cast<uint32_t>(__builtin_clzll(hi)) :
                                           (lo != 0U ? 64U - static_cast<uint32_t>(__builtin_clzll(lo)) : 0U);
        if (length == 0U) {
            return sign << 63U;
        }
        if (length <= 53U) {
            return DoubleFromSignificand(lo, exponent, sign);
        }
        uint32_t shift = length - 53U;
        uint64_t significand = lo >> shift;
        if (hi != 0U) {
            significand |= hi << (64U - shift);
        }
        const uint64_t remainder = lo & ((1ULL << shift) - 1ULL);
        const uint64_t halfway = 1ULL << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (significand & 1U) != 0U)) {
            ++significand;
        }
        if (significand == (1ULL << 53U)) {
            significand >>= 1U;
            ++shift;
        }
        return DoubleFromSignificand(significand, exponent + static_cast<int32_t>(shift), sign);
    }

    __aicore__ __attribute__((noinline)) static uint64_t DoubleModuloShift(uint64_t mantissa, uint32_t shift,
                                                                           uint64_t modulus)
    {
        uint64_t remainder = mantissa % modulus;
        while (shift != 0U) {
            const uint32_t chunk = shift > 10U ? 10U : shift;
            remainder = (remainder << chunk) % modulus;
            shift -= chunk;
        }
        return remainder;
    }

    // Compute floor-mod directly from binary64 bit patterns. The inputs are
    // represented as integer significands times powers of two. For equal or
    // nearby exponents the remainder is obtained by subtraction/modulo; larger
    // exponent gaps are reduced in safe 10-bit chunks without a 128-bit division.
    __aicore__ static inline uint64_t SubtractDoubleMagnitudes(uint64_t xMantissa, int32_t xExponent,
                                                               uint64_t divisorMantissa, int32_t divisorExponent,
                                                               uint64_t divisorAbs, uint64_t divisorSign)
    {
        const uint32_t exponentGap = static_cast<uint32_t>(divisorExponent - xExponent);
        if (exponentGap > 53U) {
            return divisorAbs | (divisorSign << 63U);
        }
        if (exponentGap == 53U) {
            const bool roundToDivisor = xMantissa == (1ULL << 52U) && (divisorMantissa & 1U) == 0U;
            return roundToDivisor ? (divisorAbs | (divisorSign << 63U)) :
                                    DoubleFromSignificand(divisorMantissa - 1U, divisorExponent, divisorSign);
        }
        if (exponentGap == 0U) {
            return DoubleFromSignificand(divisorMantissa - xMantissa, xExponent, divisorSign);
        }
        uint64_t low = divisorMantissa << exponentGap;
        uint64_t high = divisorMantissa >> (64U - exponentGap);
        if (low < xMantissa) {
            --high;
        }
        low -= xMantissa;
        return DoubleFromRoundedInteger128(high, low, xExponent, divisorSign);
    }

    __aicore__ static inline uint64_t FiniteDoubleFloorMod(uint64_t xBits, uint64_t divisorBits)
    {
        const uint64_t xSign = xBits >> 63U;
        const uint64_t divisorSign = divisorBits >> 63U;
        const uint64_t divisorAbs = divisorBits & 0x7fffffffffffffffULL;
        const uint32_t xExpField = static_cast<uint32_t>((xBits >> 52U) & 0x7ffU);
        const uint32_t divisorExpField = static_cast<uint32_t>((divisorBits >> 52U) & 0x7ffU);
        const uint64_t xFraction = xBits & 0x000fffffffffffffULL;
        const uint64_t divisorFraction = divisorBits & 0x000fffffffffffffULL;
        const uint64_t xMantissa = xExpField == 0U ? xFraction : ((1ULL << 52U) | xFraction);
        const uint64_t divisorMantissa = divisorExpField == 0U ? divisorFraction : ((1ULL << 52U) | divisorFraction);
        const int32_t xExponent = xExpField == 0U ? -1074 : static_cast<int32_t>(xExpField) - 1075;
        const int32_t divisorExponent = divisorExpField == 0U ? -1074 : static_cast<int32_t>(divisorExpField) - 1075;

        const bool smallerMagnitude = xExponent < divisorExponent ||
                                      (xExponent == divisorExponent && xMantissa < divisorMantissa);
        if (smallerMagnitude) {
            if (xSign == divisorSign) {
                return xBits;
            }
            return SubtractDoubleMagnitudes(xMantissa, xExponent, divisorMantissa, divisorExponent, divisorAbs,
                                            divisorSign);
        }

        const uint32_t exponentGap = static_cast<uint32_t>(xExponent - divisorExponent);
        uint64_t remainder;
        if (exponentGap == 0U && divisorMantissa >= (1ULL << 52U)) {
            remainder = xMantissa - divisorMantissa;
        } else if (exponentGap <= 3U && divisorMantissa >= (1ULL << 52U)) {
            uint64_t shifted = xMantissa << exponentGap;
            while (shifted >= divisorMantissa) {
                shifted -= divisorMantissa;
            }
            remainder = shifted;
        } else {
            remainder = DoubleModuloShift(xMantissa, exponentGap, divisorMantissa);
        }
        if (remainder == 0U) {
            return divisorSign << 63U;
        }
        if (xSign != divisorSign) {
            remainder = divisorMantissa - remainder;
        }
        return DoubleFromSignificand(remainder, divisorExponent, divisorSign);
    }

    __aicore__ __attribute__((noinline)) static uint64_t DoubleFloorModBits(uint64_t xBits, uint64_t divisorBits)
    {
        constexpr uint64_t kSignBit = 1ULL << 63U;
        constexpr uint64_t kExponentMask = 0x7ffULL;
        constexpr uint64_t kFractionMask = 0x000fffffffffffffULL;
        constexpr uint64_t kQuietBit = 1ULL << 51U;
        constexpr uint64_t kCanonicalNan = 0x7ff8000000000000ULL;

        const uint64_t xSign = xBits >> 63U;
        const uint64_t divisorSign = divisorBits >> 63U;
        const uint64_t xAbs = xBits & ~kSignBit;
        const uint64_t divisorAbs = divisorBits & ~kSignBit;
        const uint32_t xExpField = static_cast<uint32_t>((xAbs >> 52U) & kExponentMask);
        const uint32_t divisorExpField = static_cast<uint32_t>((divisorAbs >> 52U) & kExponentMask);
        const uint64_t xFraction = xAbs & kFractionMask;
        const uint64_t divisorFraction = divisorAbs & kFractionMask;

        if (xExpField == 0x7ffU && xFraction != 0U) {
            return xBits | kQuietBit;
        }
        if (divisorExpField == 0x7ffU && divisorFraction != 0U) {
            return divisorBits | kQuietBit;
        }
        if (xExpField == 0x7ffU || divisorAbs == 0U) {
            return kCanonicalNan;
        }
        if (divisorExpField == 0x7ffU) {
            if (xAbs == 0U) {
                return divisorSign << 63U;
            }
            return xSign == divisorSign ? xBits : divisorBits;
        }
        if (xAbs == 0U) {
            return divisorSign << 63U;
        }

        return FiniteDoubleFloorMod(xBits, divisorBits);
    }

    __aicore__ inline uint64_t SourceOffset(uint64_t flat, const uint64_t* strides, uint64_t elements) const
    {
        if (elements == t_->totalElements) {
            return flat;
        }
        uint64_t offset = 0U;
        for (uint32_t axis = t_->rank; axis-- > 0U;) {
            const uint64_t extent = t_->outShape[axis];
            const uint64_t coord = extent == 0U ? 0U : flat % extent;
            flat = extent == 0U ? 0U : flat / extent;
            offset += coord * strides[axis];
        }
        return offset;
    }

    __aicore__ inline void LoadBits(LocalTensor<uint64_t> dst, GlobalTensor<uint64_t> src, uint64_t elements,
                                    uint64_t offset, uint32_t count, const uint64_t* strides)
    {
        DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
        if (elements == t_->totalElements) {
            DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(uint64_t)), 0U, 0U, 0U};
            DataCopyPad(dst, src[offset], params, pad);
            return;
        }
        uint32_t local = 0U;
        while (local < count) {
            const uint64_t flat = offset + local;
            const uint32_t run = MinU32(ContiguousSourceRun(flat, strides), count - local);
            const uint64_t srcOffset = SourceOffset(flat, strides, elements);
            DataCopyExtParams params{1U, static_cast<uint32_t>(run * sizeof(uint64_t)), 0U, 0U, 0U};
            DataCopyPad(dst[local], src[srcOffset], params, pad);
            local += run;
        }
    }

    const FloorModTilingData* t_ = nullptr;
    TPipe pipe_;
    GlobalTensor<uint64_t> x1Gm_;
    GlobalTensor<uint64_t> x2Gm_;
    GlobalTensor<uint64_t> yGm_;
    GlobalTensor<uint32_t> yWordsGm_;
    TBuf<QuePosition::VECIN> x1In_;
    TBuf<QuePosition::VECIN> x2In_;
    TBuf<TPosition::VECCALC> aFp_;
    TBuf<TPosition::VECCALC> bFp_;
    TBuf<TPosition::VECCALC> remFp_;
    TBuf<TPosition::VECCALC> scratch0_;
    TBuf<TPosition::VECCALC> scratch1_;
    TBuf<TPosition::VECCALC> scratch2_;
    TBuf<TPosition::VECCALC> index_;
    TBuf<TPosition::VECCALC> mask_;
    TBuf<TPosition::VECCALC> floorTmp_;
    uint32_t tile_ = 1U;
    uint32_t aligned_ = 8U;
    uint32_t maskStride_ = FLOOR_MOD_UB_BLOCK_BYTES;
};

template <typename T, uint32_t KernelPath>
__aicore__ inline void RunFloorMod(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
                                   const FloorModTilingData* tiling)
{
    if constexpr (KernelPath == FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH) {
        FloorModDenseTailBatch<T> op;
        op.Init(x1, x2, y, tiling);
        op.Process();
    } else if constexpr (KernelPath == FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH) {
        FloorModCompactRowBatch<T> op;
        op.Init(x1, x2, y, tiling);
        op.Process();
    } else if constexpr (KernelPath == FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST) {
        FloorModScalarBroadcast<T> op;
        op.Init(x1, x2, y, tiling);
        op.Process();
    } else if constexpr (KernelPath == FLOOR_MOD_TPL_PATH_FP64_STORAGE) {
        FloorModFp64Storage op;
        op.Init(x1, x2, y, tiling);
        op.Process();
    } else if constexpr (KernelPath == FLOOR_MOD_TPL_PATH_SMALL_DENSE) {
        FloorModVector<T, true> op;
        op.Init(x1, x2, y, tiling);
        op.Process();
    } else {
        FloorModVector<T> op;
        op.Init(x1, x2, y, tiling);
        op.Process();
    }
}

template <int D_T_X1, int D_T_X2, int D_T_Y, int EXPECTED>
struct FloorModDtypeMatch {
    static constexpr bool value = D_T_X1 == EXPECTED && D_T_X2 == EXPECTED && D_T_Y == EXPECTED;
};

template <int D_T_X1, int D_T_X2, int D_T_Y, uint32_t KERNEL_PATH>
__aicore__ inline void FloorModKernelImpl(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
                                          const FloorModTilingData* tiling)
{
    if constexpr (FloorModDtypeMatch<D_T_X1, D_T_X2, D_T_Y, FLOOR_MOD_TPL_INT32>::value) {
        RunFloorMod<int32_t, KERNEL_PATH>(x1, x2, y, tiling);
    } else if constexpr (FloorModDtypeMatch<D_T_X1, D_T_X2, D_T_Y, FLOOR_MOD_TPL_INT64>::value) {
        RunFloorMod<int64_t, KERNEL_PATH>(x1, x2, y, tiling);
    } else if constexpr (FloorModDtypeMatch<D_T_X1, D_T_X2, D_T_Y, FLOOR_MOD_TPL_DOUBLE>::value) {
        if constexpr (KERNEL_PATH == FLOOR_MOD_TPL_PATH_FP64_STORAGE) {
            FloorModFp64Storage op;
            op.Init(x1, x2, y, tiling);
            op.Process();
        }
    } else if constexpr (FloorModDtypeMatch<D_T_X1, D_T_X2, D_T_Y, FLOOR_MOD_TPL_FP16>::value) {
        RunFloorMod<half, KERNEL_PATH>(x1, x2, y, tiling);
    } else if constexpr (FloorModDtypeMatch<D_T_X1, D_T_X2, D_T_Y, FLOOR_MOD_TPL_FP32>::value) {
        RunFloorMod<float, KERNEL_PATH>(x1, x2, y, tiling);
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if constexpr (FloorModDtypeMatch<D_T_X1, D_T_X2, D_T_Y, FLOOR_MOD_TPL_BF16>::value) {
        RunFloorMod<bfloat16_t, KERNEL_PATH>(x1, x2, y, tiling);
#endif
    }
}

} // namespace FloorModNs

#endif
