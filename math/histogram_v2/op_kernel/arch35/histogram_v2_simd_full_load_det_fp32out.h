/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_simd_full_load_det_fp32out.h
 * \brief Deterministic UB full-load SIMD histogram (dhistv2) with float32 output.
 */
#ifndef HISTOGRAM_V2_SIMD_FULL_LOAD_DET_FP32OUT_H
#define HISTOGRAM_V2_SIMD_FULL_LOAD_DET_FP32OUT_H

#include "kernel_operator.h"
#include "reg_compute/kernel_reg_compute_intf.h"

// Deterministic HistogramV2 built on the dav-3510 SIMD histogram instruction
// (dhistv2, exposed as Reg::Histograms<..., FREQUENCY>) instead of SIMT UB atomics.
//
// dhistv2 only accepts uint8 keys and only accumulates into a uint16 register, so
// the kernel is split into two vector passes over each UB tile:
//   pass 1: x (fp32/fp16) -> dense uint8 bin index
//   pass 2: uint8 keys -> per-bin uint16 counts, widened into an int32 accumulator
// Cross-core reduction has two variants, both deterministic:
//   HIST_SIMD_REDUCE 0: zero y, int32 GM atomic add, SyncAll, cast (2x SyncAll)
//   HIST_SIMD_REDUCE 1: per-core private workspace slot, SyncAll, core 0 sums the
//                       slots in a fixed order and writes fp32 (1x SyncAll)
#ifndef HIST_SIMD_REDUCE
#define HIST_SIMD_REDUCE 1
#endif

namespace HistogramV2SIMD {
using namespace AscendC;

// Elements per UB tile. Must be a multiple of HIST_SIMD_KEYS_PER_HIST and small
// enough that one bin cannot exceed int16 range within a single tile, which is what
// lets the uint16 counters be widened without an intermediate saturation check.
#ifndef HIST_SIMD_TILE
#define HIST_SIMD_TILE 4096
#endif
constexpr int32_t HIST_SIMD_TILE_ELEMS = HIST_SIMD_TILE;
constexpr int32_t HIST_SIMD_VL_F32 = 64;
constexpr int32_t HIST_SIMD_KEYS_PER_HIST = 256;
// BIN0 of dhistv2 covers keys [0, 127]; key 127 is reserved as the out-of-range sink.
constexpr int32_t HIST_SIMD_BIN_SLOTS = 128;
constexpr int32_t HIST_SIMD_TRASH_KEY = 127;
constexpr int64_t HIST_SIMD_MAX_BINS = 127;

static_assert(HIST_SIMD_TILE_ELEMS % HIST_SIMD_KEYS_PER_HIST == 0, "UB tile must be a whole number of dhistv2 groups");
static_assert(HIST_SIMD_TILE_ELEMS <= 32767, "one bin per tile must fit in int16");

constexpr AscendC::Reg::CastTrait kHistSimdB16ToF32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_NONE};
constexpr AscendC::Reg::CastTrait kHistSimdF32ToB16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait kHistSimdB16ToB8 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::CAST_TRUNC};

// Pass 1. Lanes that fall outside [min, max], are NaN, or sit past the end of the
// tile are all rewritten to HIST_SIMD_TRASH_KEY, so pass 2 needs neither a mask nor
// a tail branch and simply ignores bin 127.
template <typename X_TYPE>
__simd_vf__ inline void HistSimdMapKeysVf(__ubuf__ X_TYPE* xUb, __ubuf__ uint8_t* keyUb, uint32_t count,
                                          uint16_t repeat, float minValue, float maxValue, float minMaxLength,
                                          float binsF, float binsMinusOneF)
{
    AscendC::Reg::MaskReg full = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> minReg;
    AscendC::Reg::RegTensor<float> maxReg;
    AscendC::Reg::RegTensor<float> lenReg;
    AscendC::Reg::RegTensor<float> trashReg;
    AscendC::Reg::Duplicate(minReg, minValue, full);
    AscendC::Reg::Duplicate(maxReg, maxValue, full);
    AscendC::Reg::Duplicate(lenReg, minMaxLength, full);
    AscendC::Reg::Duplicate(trashReg, static_cast<float>(HIST_SIMD_TRASH_KEY), full);

    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * HIST_SIMD_VL_F32;
        AscendC::Reg::MaskReg inTile = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::RegTensor<float> xReg;
        if constexpr (sizeof(X_TYPE) == 4) {
            AscendC::Reg::LoadAlign(xReg, reinterpret_cast<__ubuf__ float*>(xUb) + off);
        } else {
            AscendC::Reg::RegTensor<half> halfReg;
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                halfReg, reinterpret_cast<__ubuf__ half*>(xUb) + off);
            AscendC::Reg::Cast<float, half, kHistSimdB16ToF32>(xReg, halfReg, full);
        }

        // (x - min) * bins / (max - min) with truncation, evaluated in the same order
        // and the same fp32 precision as the SIMT path, so results stay bit-exact
        // against torch.histc.
        AscendC::Reg::RegTensor<float> idxReg;
        AscendC::Reg::Sub(idxReg, xReg, minReg, full);
        AscendC::Reg::Muls(idxReg, idxReg, binsF, full);
        AscendC::Reg::Div(idxReg, idxReg, lenReg, full);
        AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_TRUNC>(idxReg, idxReg, full);
        AscendC::Reg::Mins(idxReg, idxReg, binsMinusOneF, full);

        AscendC::Reg::MaskReg geMin;
        AscendC::Reg::MaskReg leMax;
        AscendC::Reg::MaskReg keep;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(geMin, xReg, minReg, full);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::LE>(leMax, xReg, maxReg, full);
        AscendC::Reg::And(keep, geMin, leMax, full);
        AscendC::Reg::And(keep, keep, inTile, full);
        AscendC::Reg::Select(idxReg, idxReg, trashReg, keep);

        // Bin indices are integral and <= 127, so fp32 -> fp16 -> uint8 is exact.
        AscendC::Reg::RegTensor<half> keyHalf;
        AscendC::Reg::RegTensor<uint8_t> keyReg;
        AscendC::Reg::Cast<half, float, kHistSimdF32ToB16>(keyHalf, idxReg, full);
        AscendC::Reg::Cast<uint8_t, half, kHistSimdB16ToB8>(keyReg, keyHalf, full);
        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(keyUb + off, keyReg, full);
    }
}

// Pass 2. One dhistv2 per 256 keys; the uint16 counters cannot overflow because a
// tile holds at most HIST_SIMD_TILE_ELEMS elements.
__simd_vf__ inline void HistSimdCountKeysVf(__ubuf__ uint8_t* keyUb, __ubuf__ uint16_t* histUb, uint16_t groups)
{
    AscendC::Reg::MaskReg mask8 = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg mask16 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint16_t> histReg;
    AscendC::Reg::Duplicate(histReg, static_cast<uint16_t>(0), mask16);

    for (uint16_t g = 0; g < groups; ++g) {
        int32_t off = static_cast<int32_t>(g) * HIST_SIMD_KEYS_PER_HIST;
        AscendC::Reg::RegTensor<uint8_t> keyReg;
        AscendC::Reg::LoadAlign(keyReg, keyUb + off);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(histReg, keyReg, mask8);
    }

    AscendC::Reg::StoreAlign(histUb, histReg, mask16);
}

template <typename X_TYPE, typename COMPUTE_TYPE>
class HistogramV2SimdFullLoadDetFp32Out {
public:
    __aicore__ inline HistogramV2SimdFullLoadDetFp32Out(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace,
                                const HistogramV2SimtTilingData* __restrict tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearYInt(int64_t base, int64_t length);
    __aicore__ inline void AccumulateCore(int64_t xIndexBase, int64_t coreDataLength);
    __aicore__ inline void CastYToFloat(int64_t base, int64_t length);
    __aicore__ inline void ReduceSlotsToFloat();

    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<int32_t> yGmInt_;
    GlobalTensor<float> yGmFloat_;
    GlobalTensor<int32_t> wsGm_;

    TPipe* pipe_;
    TQue<TPosition::VECIN, 2> xQue_;
    TBuf<TPosition::VECCALC> keyBuf_;
    TBuf<TPosition::VECCALC> histBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> auxBuf_;
#if HIST_SIMD_REDUCE == 1
    TBuf<TPosition::VECCALC> slotBuf_;
#endif

    int32_t blockIdx_ = 0;
    int64_t bins_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;
    int64_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
    // Workspace row stride: bins rounded up to a 32B vector boundary rather than the
    // full 128 histogram slots, which keeps core 0's reduce read small.
    int64_t binsAlign_ = 0;
};

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimdFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace,
    const HistogramV2SimtTilingData* __restrict tilingData, TPipe* tPipe)
{
    this->pipe_ = tPipe;
    this->blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    this->bins_ = tilingData->bins;
    this->formerLength_ = tilingData->formerLength;
    this->tailLength_ = tilingData->tailLength;
    this->needXCoreNum_ = tilingData->needXCoreNum;
    this->clearYFactor_ = tilingData->clearYFactor;
    this->clearYCoreNum_ = tilingData->clearYCoreNum;
    this->clearYTail_ = tilingData->clearYTail;
    this->binsAlign_ = (this->bins_ + 7) & ~static_cast<int64_t>(7);

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
    this->maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
    this->yGmInt_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    this->yGmFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    this->wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));

    this->pipe_->InitBuffer(this->xQue_, 2, HIST_SIMD_TILE_ELEMS * sizeof(X_TYPE));
    this->pipe_->InitBuffer(this->keyBuf_, HIST_SIMD_TILE_ELEMS * sizeof(uint8_t));
    this->pipe_->InitBuffer(this->histBuf_, HIST_SIMD_BIN_SLOTS * sizeof(int16_t));
    this->pipe_->InitBuffer(this->accBuf_, HIST_SIMD_BIN_SLOTS * sizeof(int32_t));
    this->pipe_->InitBuffer(this->auxBuf_, HIST_SIMD_BIN_SLOTS * sizeof(int32_t));
#if HIST_SIMD_REDUCE == 1
    this->pipe_->InitBuffer(this->slotBuf_, this->needXCoreNum_ * this->binsAlign_ * sizeof(int32_t));
#endif
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimdFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::ClearYInt(int64_t base, int64_t length)
{
    LocalTensor<int32_t> zeroLocal = accBuf_.Get<int32_t>();
    Duplicate(zeroLocal, static_cast<int32_t>(0), length);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(yGmInt_[base], zeroLocal, params);
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimdFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::AccumulateCore(int64_t xIndexBase,
                                                                                               int64_t coreDataLength)
{
    COMPUTE_TYPE minValue = static_cast<COMPUTE_TYPE>(minGm_(0));
    COMPUTE_TYPE maxValue = static_cast<COMPUTE_TYPE>(maxGm_(0));
    if (minValue == maxValue) {
        minValue = minValue - 1;
        maxValue = maxValue + 1;
    }
    const float minF = static_cast<float>(minValue);
    const float maxF = static_cast<float>(maxValue);
    const float lenF = static_cast<float>(maxValue - minValue);
    const float binsF = static_cast<float>(bins_);
    const float binsMinusOneF = static_cast<float>(bins_ - 1);

    LocalTensor<uint8_t> keyLocal = keyBuf_.Get<uint8_t>();
    // dhistv2 counts into uint16, but the widening Cast below is the signed int16 one;
    // both views are safe because a tile can contribute at most HIST_SIMD_TILE_ELEMS
    // to a single bin, well inside int16 range.
    LocalTensor<int16_t> histLocal = histBuf_.Get<int16_t>();
    LocalTensor<int32_t> accLocal = accBuf_.Get<int32_t>();
    LocalTensor<int32_t> histIntLocal = auxBuf_.Get<int32_t>();

    Duplicate(accLocal, static_cast<int32_t>(0), HIST_SIMD_BIN_SLOTS);

    __ubuf__ uint8_t* keyAddr = (__ubuf__ uint8_t*)keyLocal.GetPhyAddr();
    __ubuf__ uint16_t* histAddr = (__ubuf__ uint16_t*)histLocal.GetPhyAddr();

    for (int64_t done = 0; done < coreDataLength; done += HIST_SIMD_TILE_ELEMS) {
        int64_t remain = coreDataLength - done;
        int32_t tileCount = static_cast<int32_t>(remain < HIST_SIMD_TILE_ELEMS ? remain : HIST_SIMD_TILE_ELEMS);
        uint16_t groups = static_cast<uint16_t>((tileCount + HIST_SIMD_KEYS_PER_HIST - 1) / HIST_SIMD_KEYS_PER_HIST);
        uint16_t repeat = static_cast<uint16_t>(groups * (HIST_SIMD_KEYS_PER_HIST / HIST_SIMD_VL_F32));

        LocalTensor<X_TYPE> xLocal = xQue_.template AllocTensor<X_TYPE>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(tileCount * sizeof(X_TYPE)), 0, 0,
                                     0};
        DataCopyPadExtParams<X_TYPE> padParams{false, 0, 0, static_cast<X_TYPE>(0)};
        DataCopyPad(xLocal, xGm_[xIndexBase + done], copyParams, padParams);
        xQue_.EnQue(xLocal);
        xLocal = xQue_.template DeQue<X_TYPE>();

        __ubuf__ X_TYPE* xAddr = (__ubuf__ X_TYPE*)xLocal.GetPhyAddr();
        asc_vf_call<HistSimdMapKeysVf<X_TYPE>>(xAddr, keyAddr, static_cast<uint32_t>(tileCount), repeat, minF, maxF,
                                               lenF, binsF, binsMinusOneF);
        PipeBarrier<PIPE_V>();
        asc_vf_call<HistSimdCountKeysVf>(keyAddr, histAddr, groups);
        PipeBarrier<PIPE_V>();

        Cast(histIntLocal, histLocal, RoundMode::CAST_NONE, HIST_SIMD_BIN_SLOTS);
        PipeBarrier<PIPE_V>();
        Add(accLocal, accLocal, histIntLocal, HIST_SIMD_BIN_SLOTS);
        PipeBarrier<PIPE_V>();

        xQue_.template FreeTensor<X_TYPE>(xLocal);
    }

#if HIST_SIMD_REDUCE == 1
    // With one active core there is nothing to reduce, so cast in place and write y
    // directly. This skips a workspace round trip and the SyncAll, which is the whole
    // cost of the kernel once the input is small enough to be pure launch overhead.
    if (needXCoreNum_ == 1) {
        LocalTensor<float> floatLocal = auxBuf_.Get<float>();
        Cast(floatLocal, accLocal, RoundMode::CAST_NONE, binsAlign_);

        event_t castToWrite = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(castToWrite);
        WaitFlag<HardEvent::V_MTE3>(castToWrite);

        DataCopyExtParams yParams{static_cast<uint16_t>(1), static_cast<uint32_t>(bins_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGmFloat_[0], floatLocal, yParams);
        return;
    }
#endif

    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

#if HIST_SIMD_REDUCE == 1
    // Each core owns its own row, so the write needs no atomics and no pre-zeroed target.
    DataCopyExtParams slotParams{static_cast<uint16_t>(1), static_cast<uint32_t>(binsAlign_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPad(wsGm_[static_cast<int64_t>(blockIdx_) * binsAlign_], accLocal, slotParams);
#else
    DataCopyExtParams addParams{static_cast<uint16_t>(1), static_cast<uint32_t>(bins_ * sizeof(int32_t)), 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(yGmInt_[0], accLocal, addParams);
    SetAtomicNone();
#endif
}

// Sums the per-core slots on a single core. Integer addition is associative, so the
// result is independent of the order the slots are combined in.
template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimdFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::ReduceSlotsToFloat()
{
#if HIST_SIMD_REDUCE == 1
    LocalTensor<int32_t> slotLocal = slotBuf_.Get<int32_t>();
    LocalTensor<float> floatLocal = auxBuf_.Get<float>();

    const int64_t width = binsAlign_;
    DataCopyExtParams readParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(needXCoreNum_ * width * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(slotLocal, wsGm_[0], readParams, padParams);

    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);

    // Folding the upper half of the rows onto the lower half is a single wide Add per
    // level, so the reduce costs ~log2(cores) issues instead of one Add per core. The
    // per-core form spent 1.6us of vec time on issue overhead alone.
    for (int64_t rows = needXCoreNum_; rows > 1;) {
        int64_t half = rows >> 1;
        Add(slotLocal, slotLocal, slotLocal[half * width], half * width);
        PipeBarrier<PIPE_V>();
        if ((rows & 1) != 0) {
            Add(slotLocal, slotLocal, slotLocal[(rows - 1) * width], width);
            PipeBarrier<PIPE_V>();
        }
        rows = half;
    }

    Cast(floatLocal, slotLocal, RoundMode::CAST_NONE, width);

    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);

    DataCopyExtParams writeParams{static_cast<uint16_t>(1), static_cast<uint32_t>(bins_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(yGmFloat_[0], floatLocal, writeParams);
#endif
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimdFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::CastYToFloat(int64_t base,
                                                                                             int64_t length)
{
    LocalTensor<int32_t> intLocal = accBuf_.Get<int32_t>();
    LocalTensor<float> floatLocal = auxBuf_.Get<float>();

    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(intLocal, yGmInt_[base], params, padParams);

    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);

    Cast(floatLocal, intLocal, RoundMode::CAST_NONE, length);

    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);

    DataCopyPad(yGmFloat_[base], floatLocal, params);
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimdFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }

#if HIST_SIMD_REDUCE == 0
    int64_t clearYIndexBase = blockIdx_ * clearYFactor_;
    int64_t clearYDataLength = (blockIdx_ == clearYCoreNum_ - 1) ? clearYTail_ : clearYFactor_;

    if (blockIdx_ < clearYCoreNum_) {
        ClearYInt(clearYIndexBase, clearYDataLength);
    }
#ifndef __CCE_UT_TEST__
    SyncAll();
#endif
#endif

    if (blockIdx_ < needXCoreNum_) {
        int64_t xIndexBase = blockIdx_ * formerLength_;
        int64_t coreDataLength = (blockIdx_ == needXCoreNum_ - 1) ? tailLength_ : formerLength_;
        AccumulateCore(xIndexBase, coreDataLength);
    }

#if HIST_SIMD_REDUCE == 1
    // Uniform across blocks, so no core is left waiting at a barrier the others skipped.
    if (needXCoreNum_ == 1) {
        return;
    }
#endif

#ifndef __CCE_UT_TEST__
    SyncAll();
#endif

#if HIST_SIMD_REDUCE == 1
    if (blockIdx_ == 0) {
        ReduceSlotsToFloat();
    }
#else
    if (blockIdx_ < clearYCoreNum_) {
        CastYToFloat(clearYIndexBase, clearYDataLength);
    }
#endif
}

} // namespace HistogramV2SIMD

#endif // HISTOGRAM_V2_SIMD_FULL_LOAD_DET_FP32OUT_H
