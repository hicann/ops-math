/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file population_count.h
 * \brief PopulationCount kernel implementation (arch35)
 *
 * Template parameters:
 *   - T:           input dtype (int16_t / uint16_t)
 *   - BUFFER_MODE: 0 = single buffer, 1 = double buffer
 *
 * Algorithm: 16-bit SWAR popcount (4 steps), executed as a register-resident
 * vector chain (AscendC::Reg, __VEC_SCOPE__):
 *   x = x - ((x >> 1) & 0x5555);
 *   x = (x & 0x3333) + ((x >> 2) & 0x3333);
 *   x = (x + (x >> 4)) & 0x0F0F;
 *   y = x + (x >> 8);            // low byte holds popcount [0..16]
 *   out_u8 = low byte of y;      // DIST_PACK_B16 store selects it per lane
 *
 * For int16 input: bits are reinterpreted as uint16 so all right-shifts are
 * logical (identical bit-level popcount semantics).
 *
 * Why register-resident (was: LocalTensor chain via VECCALC tmpBuf):
 *   the SWAR dependency chain is serial; in UB-resident form every op pays one
 *   UB read + one UB write, which adds avoidable latency and cold-run variance
 *   to launch-bound small shapes. The VF chain loads once, relays intermediates
 *   in vector registers, and stores once.
 *
 * Exit path note (replaces the historical 0x00FF mask + saturation Cast):
 *   arch35 `Cast<uint8_t, uint16_t, CAST_NONE>` performs numeric saturation
 *   (>255 -> 255), NOT low-byte truncation (verified by probe_uint16_small /
 *   probe_int16_dbuf / probe_unaligned_tail on NPU 2026-04-20), so the classic
 *   chain needed `Ands(u, u, 0x00FF)` before Cast. The VF exit avoids both ops:
 *   DIST_PACK_B16 stores the low byte of every 16-bit lane, which after SWAR
 *   step 4 is exactly the popcount (high byte holds only discarded residue).
 */

#ifndef POPULATION_COUNT_H
#define POPULATION_COUNT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "population_count_tiling_data.h"
#include "population_count_tiling_key.h"

namespace NsPopulationCount {

using namespace AscendC;

constexpr uint16_t SWAR_MASK_EVEN_BITS = 0x5555;
constexpr uint16_t SWAR_MASK_LOW_TWO_BITS = 0x3333;
constexpr uint16_t SWAR_MASK_LOW_NIBBLE = 0x0F0F;

// DAV_3510 vector register is 256B; u16 in packed 16-bit lanes processes 128
// elements per beat (LoadDist::DIST_NORM), and the output store dist
// DIST_PACK_B16 takes the low byte of every 16-bit lane, which is exactly the
// popcount after SWAR step 4 — no Cast / Pack / 0x00FF mask needed on exit.
constexpr uint32_t POPCOUNT_VF_TILE = 128U;

// Register-resident SWAR popcount chain.
// in/out are UB pointers derived from LocalTensor::GetPhyAddr() by the caller;
// `elements` is the number of u16 elements to process (tail beat narrows via mask).
__aicore__ inline void PopulationCountVF(__ubuf__ uint16_t* in, __ubuf__ uint8_t* out, uint32_t elements)
{
    const uint16_t tileCount = static_cast<uint16_t>((elements + POPCOUNT_VF_TILE - 1U) / POPCOUNT_VF_TILE);
    uint32_t remaining = elements;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> u;
        Reg::RegTensor<uint16_t> t;
        Reg::RegTensor<uint16_t> mEvenBits;
        Reg::RegTensor<uint16_t> mLowTwoBits;
        Reg::RegTensor<uint16_t> mLowNibble;
        Reg::MaskReg preg;
        // Loop-invariant SWAR constants: duplicated once, reused by every tile.
        Reg::Duplicate(mEvenBits, SWAR_MASK_EVEN_BITS);
        Reg::Duplicate(mLowTwoBits, SWAR_MASK_LOW_TWO_BITS);
        Reg::Duplicate(mLowNibble, SWAR_MASK_LOW_NIBBLE);
        for (uint16_t i = 0; i < tileCount; ++i) {
            // Per-beat mask count: maintain the remaining-element counter here
            // and feed UpdateMask each beat's own element count (<= 128) instead
            // of a running total. This keeps the mask input small and makes the
            // tail narrowing independent of the intrinsic's POST_UPDATE
            // decrement semantics.
            uint32_t tileElems = (remaining > POPCOUNT_VF_TILE) ? POPCOUNT_VF_TILE : remaining;
            remaining -= tileElems;
            preg = Reg::UpdateMask<uint16_t>(tileElems);
            const uint32_t offset = static_cast<uint32_t>(i) * POPCOUNT_VF_TILE;
            // Entry: packed B16 load, values stay in registers for the whole chain.
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(u, in + offset);

            // SWAR step 1: u = u - ((u >> 1) & EVEN_BITS)
            Reg::ShiftRights(t, u, static_cast<int16_t>(1), preg);
            Reg::And(t, t, mEvenBits, preg);
            Reg::Sub(u, u, t, preg);

            // SWAR step 2: u = (u & LOW_TWO_BITS) + ((u >> 2) & LOW_TWO_BITS)
            Reg::ShiftRights(t, u, static_cast<int16_t>(2), preg);
            Reg::And(t, t, mLowTwoBits, preg);
            Reg::And(u, u, mLowTwoBits, preg);
            Reg::Add(u, u, t, preg);

            // SWAR step 3: u = (u + (u >> 4)) & LOW_NIBBLE
            // (the LOW_NIBBLE mask is required: without it nibble residue from
            // the neighboring byte carries into bit 4 of the final low byte and
            // breaks popcount == 16 inputs)
            Reg::ShiftRights(t, u, static_cast<int16_t>(4), preg);
            Reg::Add(u, u, t, preg);
            Reg::And(u, u, mLowNibble, preg);

            // SWAR step 4: u = u + (u >> 8)  (low byte holds popcount [0..16],
            // high byte holds residue which the packed store discards)
            Reg::ShiftRights(t, u, static_cast<int16_t>(8), preg);
            Reg::Add(u, u, t, preg);

            // Exit: DIST_PACK_B16 stores the low byte of every 16-bit lane,
            // i.e. the popcount, contiguously as uint8.
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(reinterpret_cast<__ubuf__ uint16_t*>(out + offset),
                                                                     u, preg);
        }
    }
}

template <typename T, uint32_t BUFFER_MODE>
class PopulationCount {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline PopulationCount() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PopulationCountTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<uint8_t> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y,
                                                             const PopulationCountTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ uint8_t*)y + tilingData->blockFactor * blockIdx, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(uint8_t));
}

template <typename T, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<uint8_t> yLocal = outputQueueY.template DeQue<uint8_t>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(uint8_t));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<uint8_t> yLocal = outputQueueY.template AllocTensor<uint8_t>();

    // int16/uint16 share the same bit layout; the SWAR chain treats both as
    // uint16 so every right shift is logical (two's-complement bit count).
    PopulationCountVF(reinterpret_cast<__ubuf__ uint16_t*>(xLocal.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint8_t*>(yLocal.GetPhyAddr()), static_cast<uint32_t>(currentNum));

    outputQueueY.template EnQue<uint8_t>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsPopulationCount
#endif // POPULATION_COUNT_H
