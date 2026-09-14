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
 * \file hans_encode_simt.h
 * \brief Ascend 950 SIMD/SIMT implementation of the HANS encoder.
 */
#ifndef HANS_ENCODE_SIMT_H
#define HANS_ENCODE_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "../hans_format.h"

namespace HansEncodeArch35 {

using namespace AscendC;
using namespace HansFormat;

#ifndef HANS_ENCODE_DEBUG
#define HANS_ENCODE_DEBUG 0
#endif

#if HANS_ENCODE_DEBUG
#define HANS_ENC_DBG(fmt, ...)                                                                  \
    do {                                                                                        \
        printf("[HANS enc core=%d] " fmt "\n", static_cast<int>(GetBlockIdx()), ##__VA_ARGS__); \
    } while (0)
#else
#define HANS_ENC_DBG(fmt, ...) \
    do {                       \
    } while (0)
#endif

// Encoder pipeline overview:
//   - Group-major SIMT state update and record packing.
//   - BF16/FP32 register-based histogram paths.
//   - BF16 register split/map with fused per-group max-rank extraction.
//   - BF16/FP32 fused SIMD group metadata and overflow-prefix generation.
//   - Double-buffered software pipelines for histogram and var-suffix phases.

constexpr uint32_t UPDATE_THREADS_PER_GROUP = THREAD_COUNT / GROUP_COUNT;
constexpr uint32_t UPDATE_VALUES_PER_THREAD = BLOCK_SIZE / UPDATE_THREADS_PER_GROUP;
static_assert(THREAD_COUNT % GROUP_COUNT == 0, "state-update threads must divide evenly across groups");
static_assert(BLOCK_SIZE % UPDATE_THREADS_PER_GROUP == 0, "each state-update thread must own whole values");
static_assert(GROUP_COUNT == 64, "fused metadata path assumes exactly 64 groups");
static_assert(BLOCK_SIZE == 64, "state update and fused groupMax path assume 64 values per group");
static_assert(STATE_COUNT == GROUP_COUNT * BLOCK_SIZE, "STATE_COUNT must equal GROUP_COUNT * BLOCK_SIZE");
static_assert(THREAD_COUNT >= PDF_LENGTH, "BuildStableRank requires at least one SIMT thread per PDF symbol");
static_assert(PDF_LENGTH == 256, "HANS symbol/rank format and BIN0/BIN1 histogram paths assume 256 symbols");

__simt_vf__ __aicore__ LAUNCH_BOUND(PDF_LENGTH) inline void BuildStableRank(const __ubuf__ int32_t* pdf,
                                                                            __ubuf__ uint8_t* symbolToRank)
{
    const uint32_t symbol = Simt::GetThreadIdx();
    if (symbol >= PDF_LENGTH) {
        return;
    }
    const int32_t symbolCount = pdf[symbol];
    uint32_t rank = 0;
    for (uint32_t candidate = 0; candidate < PDF_LENGTH; ++candidate) {
        const int32_t candidateCount = pdf[candidate];
        if (candidateCount > symbolCount || (candidateCount == symbolCount && candidate < symbol)) {
            ++rank;
        }
    }
    symbolToRank[symbol] = static_cast<uint8_t>(rank);
}

template <int32_t DTYPE_BYTES, bool STORE_SYMBOLS, bool FULL_TILE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void SplitAndMapTile(
    const __ubuf__ uint8_t* raw, const __ubuf__ uint8_t* symbolToRank, __ubuf__ uint8_t* symbols,
    __ubuf__ uint8_t* mantissa, __ubuf__ uint8_t* ranks, uint32_t valueCount)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum<0>();
    const uint32_t end = FULL_TILE ? STATE_COUNT : valueCount;
    for (uint32_t index = threadIdx; index < end; index += threadCount) {
        const uint32_t rawOffset = index * DTYPE_BYTES;
        const uint32_t mantissaOffset = index * (DTYPE_BYTES - 1);
        const uint8_t symbol = raw[rawOffset + DTYPE_BYTES - 1];
        if constexpr (STORE_SYMBOLS) {
            symbols[index] = symbol;
        }
        ranks[index] = symbolToRank[symbol];
        for (uint32_t byte = 0; byte < static_cast<uint32_t>(DTYPE_BYTES - 1); ++byte) {
            mantissa[mantissaOffset + byte] = raw[rawOffset + byte];
        }
    }
}

// BF16 register-based split/map.  Process one 256-byte vector register at a
// time: extract the high-byte symbol, gather the rank LUT with native A5
// b8->b16 vgather2, then pack all byte outputs.
template <bool STORE_SYMBOLS, bool FULL_TILE>
__aicore__ inline void SplitAndMapTileBf16RegBase(__local_mem__ uint16_t* raw, __local_mem__ uint8_t* symbolToRank,
                                                  __local_mem__ uint8_t* symbols, __local_mem__ uint8_t* mantissa,
                                                  __local_mem__ uint8_t* ranks, __local_mem__ uint16_t* groupMax,
                                                  uint32_t valueCount)
{
    constexpr uint32_t VALUES_PER_REG = 128;
    const uint16_t repeatTimes = FULL_TILE ? STATE_COUNT / VALUES_PER_REG :
                                             (valueCount + VALUES_PER_REG - 1) / VALUES_PER_REG;
    uint32_t remaining = FULL_TILE ? STATE_COUNT : valueCount;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> rawReg;
        MicroAPI::RegTensor<uint16_t> symbolReg;
        MicroAPI::RegTensor<uint16_t> rankReg;
        MicroAPI::RegTensor<uint8_t> symbolPacked;
        MicroAPI::RegTensor<uint8_t> mantissaPacked;
        MicroAPI::RegTensor<uint8_t> rankPacked;
        MicroAPI::MaskReg allMaskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg lowHalfMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg highHalfMask;
        MicroAPI::MaskNot(highHalfMask, lowHalfMask, allMaskB16);
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            MicroAPI::MaskReg maskB16 = MicroAPI::UpdateMask<uint16_t>(remaining);
            MicroAPI::MaskReg maskB8;
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(maskB8, maskB16);
            MicroAPI::DataCopy(rawReg, raw + repeat * VALUES_PER_REG);
            MicroAPI::ShiftRights<uint16_t, int16_t>(symbolReg, rawReg, 8, maskB16);
            MicroAPI::DataCopyGather(rankReg, symbolToRank, symbolReg, maskB16);
            MicroAPI::Pack(mantissaPacked, rawReg);
            MicroAPI::Pack(symbolPacked, symbolReg);
            MicroAPI::Pack(rankPacked, rankReg);
            MicroAPI::DataCopy(mantissa + repeat * VALUES_PER_REG, mantissaPacked, maskB8);
            MicroAPI::DataCopy(ranks + repeat * VALUES_PER_REG, rankPacked, maskB8);
            if constexpr (STORE_SYMBOLS) {
                MicroAPI::DataCopy(symbols + repeat * VALUES_PER_REG, symbolPacked, maskB8);
            }
            if constexpr (!STORE_SYMBOLS) {
                // Pre-compute groupMax: each register = 128 values = 2 groups (64 each).
                // ReduceMax over the low 64 lanes -> groupMax[2*repeat], high 64 -> [2*repeat+1].
                //
                // StoreDist granularity MUST match the element type size:
                //   - uint16_t (2 bytes) -> DIST_FIRST_ELEMENT_B16 (2-byte store, 2-byte align)
                //   - float    (4 bytes) -> DIST_FIRST_ELEMENT_B32 (4-byte store, 4-byte align)
                // Using B32 with uint16_t causes 507035 (UB address not aligned) because
                // groupMax[2*repeat+1] is at byte offset repeat*4+2 (only 2-byte aligned).
                // Keep the explicit B16 StoreDist form.
                MicroAPI::RegTensor<uint16_t> maxRank;
                MicroAPI::MaskReg lowActiveMask;
                MicroAPI::MaskReg highActiveMask;
                MicroAPI::MaskAnd(lowActiveMask, maskB16, lowHalfMask, allMaskB16);
                MicroAPI::MaskAnd(highActiveMask, maskB16, highHalfMask, allMaskB16);
                MicroAPI::ReduceMax(maxRank, rankReg, lowActiveMask);
                MicroAPI::Duplicate(maxRank, maxRank, allMaskB16);
                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(groupMax + repeat * 2,
                                                                                          maxRank, allMaskB16);
                if (FULL_TILE || repeat * VALUES_PER_REG + BLOCK_SIZE < valueCount) {
                    MicroAPI::ReduceMax(maxRank, rankReg, highActiveMask);
                    MicroAPI::Duplicate(maxRank, maxRank, allMaskB16);
                    MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(groupMax + repeat * 2 + 1,
                                                                                              maxRank, allMaskB16);
                }
            }
        }
    }
}

// BF16 register histogram via A5 dhistv2.  Each dhistv2 accumulator holds 128
// uint16 bins; BIN0 + BIN1 cover all 256 symbols.  Two B16 registers (256
// values) are loaded, a single DeInterleave produces 256 B8 symbol lanes, and
// 2x dhistv2 counts all 256 lanes.  The B16 accumulators are widened to int32
// and added to the per-core PDF in UB.
//
// dhistv2's DataCopy always loads a full B16 register (128 uint16 = 256 B8
// lanes) per raw0/raw1 regardless of the UpdateMask.  For a partial repeat the
// stale lanes beyond the valid count would be counted, producing spurious bin
// entries.  FULL_TILE always has STATE_COUNT=4096 (a multiple of 256) so every
// repeat is a full 256-lane load.  For partial tiles the caller must pad the
// raw buffer tail to a 256-lane boundary with zeros (see ComputeHistogram).
template <bool FULL_TILE>
__aicore__ inline void AccumulateHistogramTileBf16RegBase(__local_mem__ uint16_t* raw, __local_mem__ int32_t* histogram,
                                                          uint32_t valueCount)
{
    constexpr uint32_t VALUES_PER_REG = 128;  // one B16 register = 128 uint16 values
    constexpr uint32_t B8_LANES = 256;        // dual-register DeInterleave produces 256 B8 lanes
    constexpr uint32_t BINS_PER_REG = 128;    // one Histograms accumulator holds 128 bins
    constexpr uint32_t BINS_PER_B32_REG = 64; // half of a uint16 register in uint32 elements
    const uint16_t repeatTimes = FULL_TILE ? STATE_COUNT / B8_LANES : (valueCount + B8_LANES - 1) / B8_LANES;
    uint32_t remaining = FULL_TILE ? STATE_COUNT : valueCount;
    __local_mem__ uint16_t* rawPtr = raw;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> raw0, raw1;
        MicroAPI::RegTensor<uint8_t> mantissaBytes, symbolBytes;
        MicroAPI::RegTensor<uint16_t> histogramLow, histogramHigh;
        MicroAPI::RegTensor<uint16_t> zeroB16;
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::Duplicate(histogramLow, 0, maskB16);
        MicroAPI::Duplicate(histogramHigh, 0, maskB16);
        MicroAPI::Duplicate(zeroB16, 0, maskB16);
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            uint32_t elemsThisRepeat = (remaining > B8_LANES ? B8_LANES : remaining);
            MicroAPI::MaskReg inputMaskB8 = MicroAPI::UpdateMask<uint8_t>(elemsThisRepeat);
            // Load TWO B16 registers (256 values), POST_MODE_UPDATE advances rawPtr
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(raw0, rawPtr, VALUES_PER_REG);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(raw1, rawPtr, VALUES_PER_REG);
            // Single DeInterleave: raw0/raw1 viewed as bytes.
            //   symbolBytes   = high byte of each of 256 B16 (BF16 symbol)  [SECOND output]
            //   mantissaBytes = low byte of each of 256 B16 (BF16 mantissa) [FIRST output]
            MicroAPI::DeInterleave(mantissaBytes, symbolBytes, (MicroAPI::RegTensor<uint8_t>&)raw0,
                                   (MicroAPI::RegTensor<uint8_t>&)raw1);
            // 2x dhistv2 over all 256 B8 lanes (BIN0 + BIN1)
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::FREQUENCY>(histogramLow, symbolBytes, inputMaskB8);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::FREQUENCY>(histogramHigh, symbolBytes, inputMaskB8);
            remaining -= elemsThisRepeat;
        }

        MicroAPI::RegTensor<int32_t> tileHistogram0;
        MicroAPI::RegTensor<int32_t> tileHistogram1;
        MicroAPI::RegTensor<int32_t> tileHistogram2;
        MicroAPI::RegTensor<int32_t> tileHistogram3;
        MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)tileHistogram0,
                             (MicroAPI::RegTensor<uint16_t>&)tileHistogram1, histogramLow, zeroB16);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)tileHistogram2,
                             (MicroAPI::RegTensor<uint16_t>&)tileHistogram3, histogramHigh, zeroB16);

        MicroAPI::RegTensor<int32_t> accumulated;
        MicroAPI::DataCopy(accumulated, histogram);
        MicroAPI::Add(accumulated, accumulated, tileHistogram0, maskB32);
        MicroAPI::DataCopy(histogram, accumulated, maskB32);
        MicroAPI::DataCopy(accumulated, histogram + BINS_PER_B32_REG);
        MicroAPI::Add(accumulated, accumulated, tileHistogram1, maskB32);
        MicroAPI::DataCopy(histogram + BINS_PER_B32_REG, accumulated, maskB32);
        MicroAPI::DataCopy(accumulated, histogram + BINS_PER_REG);
        MicroAPI::Add(accumulated, accumulated, tileHistogram2, maskB32);
        MicroAPI::DataCopy(histogram + BINS_PER_REG, accumulated, maskB32);
        MicroAPI::DataCopy(accumulated, histogram + BINS_PER_REG + BINS_PER_B32_REG);
        MicroAPI::Add(accumulated, accumulated, tileHistogram3, maskB32);
        MicroAPI::DataCopy(histogram + BINS_PER_REG + BINS_PER_B32_REG, accumulated, maskB32);
    }
}

// FP32 register histogram via A5 dhistv2.  dhistv2 always operates on 256 B8
// symbol lanes regardless of the original data width.  For FP32 the high byte
// (symbol) is extracted as follows:
//   1. Load one B32 register (64 uint32 values = 256 bytes).
//   2. ShiftRights by 24: move the high byte to the low 8 bits.
//   3. DeInterleave B32 -> two B16 (128 + 128 lanes).
//   4. DeInterleave B16 -> two B8  (256 + 256 lanes); the FIRST output holds
//      the valid symbol bytes in its low 64 lanes.
//   5. MaskPack: B32 mask (64 elements) -> B16 mask (128) -> B8 mask (256),
//      ensuring only the 64 valid symbol lanes are counted by dhistv2.
//   6. 2x Histograms (BIN0 + BIN1) cover all 256 symbol bins.
//
// Only 64 of 256 B8 lanes carry valid symbols (25% lane utilisation).  Despite
// the lower utilisation, dhistv2 is a single-cycle hardware instruction and
// avoids the UB-bank contention and thread synchronisation of the SIMT
// AtomicAdd path.  The B16 accumulators are widened to int32 and added to the
// per-core int32 histogram in UB, as in the BF16 path.
template <bool FULL_TILE>
__aicore__ inline void AccumulateHistogramTileFp32RegBase(__local_mem__ uint32_t* raw, __local_mem__ int32_t* histogram,
                                                          uint32_t valueCount)
{
    constexpr uint32_t VALUES_PER_REG = 64;   // one B32 register = 64 uint32 values
    constexpr uint32_t BINS_PER_REG = 128;    // one Histograms accumulator holds 128 bins
    constexpr uint32_t BINS_PER_B32_REG = 64; // half of a uint16 register in uint32 elements
    constexpr int16_t SYMBOL_SHIFT = 24;      // FP32 symbol = highest byte, right-shift 24

    const uint16_t repeatTimes = FULL_TILE ? STATE_COUNT / VALUES_PER_REG :
                                             (valueCount + VALUES_PER_REG - 1) / VALUES_PER_REG;
    uint32_t remaining = FULL_TILE ? STATE_COUNT : valueCount;
    __local_mem__ uint32_t* rawPtr = raw;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> rawReg, shiftReg;
        MicroAPI::RegTensor<uint16_t> histogramLow, histogramHigh;
        MicroAPI::RegTensor<uint16_t> zeroB16;
        MicroAPI::RegTensor<uint32_t> zeroB32;
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::Duplicate(histogramLow, 0, maskB16);
        MicroAPI::Duplicate(histogramHigh, 0, maskB16);
        MicroAPI::Duplicate(zeroB16, 0, maskB16);
        MicroAPI::Duplicate(zeroB32, 0, maskB32);

        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            uint32_t elemsThisRepeat = (remaining > VALUES_PER_REG ? VALUES_PER_REG : remaining);

            // B32 element mask (64 elements); MaskPack will expand to B8 (256).
            MicroAPI::MaskReg inputMaskB32 = MicroAPI::UpdateMask<uint32_t>(elemsThisRepeat);

            // 1. Load one B32 register (64 FP32 values), POST_MODE_UPDATE advances rawPtr.
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(rawReg, rawPtr, VALUES_PER_REG);

            // 2. Right-shift 24 bits: high byte (symbol) -> low 8 bits.
            MicroAPI::ShiftRights<uint32_t, int16_t>(shiftReg, rawReg, SYMBOL_SHIFT, maskB32);

            // 3. DeInterleave B32 -> two B16.
            MicroAPI::RegTensor<uint16_t> shiftU16Low, shiftU16High;
            MicroAPI::DeInterleave(shiftU16Low, shiftU16High, (MicroAPI::RegTensor<uint16_t>&)shiftReg,
                                   (MicroAPI::RegTensor<uint16_t>&)zeroB32);

            // 4. DeInterleave B16 -> two B8 (256 B8 lanes).  symbolBytes holds
            //    the valid symbols in its low 64 lanes; paddingBytes are zeros.
            MicroAPI::RegTensor<uint8_t> symbolBytes, paddingBytes;
            MicroAPI::DeInterleave(symbolBytes, paddingBytes, (MicroAPI::RegTensor<uint8_t>&)shiftU16Low,
                                   (MicroAPI::RegTensor<uint8_t>&)zeroB16);

            // 5. Mask expansion: B32(64) -> B16(128) -> B8(256).
            //    Ensures only the 64 valid symbol lanes are counted.
            MicroAPI::MaskReg maskU16, maskU8;
            MicroAPI::MaskPack(maskU16, inputMaskB32);
            MicroAPI::MaskPack(maskU8, maskU16);

            // 6. 2x dhistv2 (BIN0 + BIN1 cover all 256 symbol bins).
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::FREQUENCY>(histogramLow, symbolBytes, maskU8);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::FREQUENCY>(histogramHigh, symbolBytes, maskU8);

            remaining -= elemsThisRepeat;
        }

        // 7. Widen B16 histogram accumulators to int32 and add to the
        //    persistent per-core int32 histogram in UB (same as BF16 path).
        MicroAPI::RegTensor<int32_t> tileHist0, tileHist1, tileHist2, tileHist3;
        MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)tileHist0, (MicroAPI::RegTensor<uint16_t>&)tileHist1,
                             histogramLow, zeroB16);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)tileHist2, (MicroAPI::RegTensor<uint16_t>&)tileHist3,
                             histogramHigh, zeroB16);

        MicroAPI::RegTensor<int32_t> accumulated;
        MicroAPI::DataCopy(accumulated, histogram);
        MicroAPI::Add(accumulated, accumulated, tileHist0, maskB32);
        MicroAPI::DataCopy(histogram, accumulated, maskB32);
        MicroAPI::DataCopy(accumulated, histogram + BINS_PER_B32_REG);
        MicroAPI::Add(accumulated, accumulated, tileHist1, maskB32);
        MicroAPI::DataCopy(histogram + BINS_PER_B32_REG, accumulated, maskB32);
        MicroAPI::DataCopy(accumulated, histogram + BINS_PER_REG);
        MicroAPI::Add(accumulated, accumulated, tileHist2, maskB32);
        MicroAPI::DataCopy(histogram + BINS_PER_REG, accumulated, maskB32);
        MicroAPI::DataCopy(accumulated, histogram + BINS_PER_REG + BINS_PER_B32_REG);
        MicroAPI::Add(accumulated, accumulated, tileHist3, maskB32);
        MicroAPI::DataCopy(histogram + BINS_PER_REG + BINS_PER_B32_REG, accumulated, maskB32);
    }
}

// Fused SIMD group metadata generation.
//
// Data flow (1 B32 register = 64 lanes = 64 groups):
//   groupMax[64*u16] -> maxRankB32[64*u32]
//   maxRankB32 -> seven threshold compares -> groupBitsB32
//   groupBitsB32 -> groupBits UB
//   counters + groupBitsB32 -> overflowMask
//   overflowMask -> Unsqueeze -> exclusive overflowPrefix
//   overflowMask -> packed overflow flags
//
// Partial tiles mask inactive groups out of groupBits/overflow generation.
// Unsqueeze is an exclusive prefix over the complete mask register: trailing
// inactive (false) lanes keep the total count accumulated by the active lanes.
// Therefore the caller can recover overflowCount from the last metadata lane:
//   overflowPrefix[GROUP_COUNT - 1] + overflow[GROUP_COUNT - 1].
// Keep partial-tile coverage for this on-wire indexing contract.
template <bool FULL_TILE>
__aicore__ inline void PrepareGroupMetadataSimd(__local_mem__ uint16_t* groupMax, __local_mem__ int32_t* counters,
                                                __local_mem__ uint16_t* groupBits, __local_mem__ uint8_t* overflow,
                                                __local_mem__ int32_t* overflowPrefix, uint32_t valueCount)
{
    const uint32_t activeGroups = FULL_TILE ? GROUP_COUNT : valueCount / BLOCK_SIZE;
    uint32_t groupCountB16 = GROUP_COUNT;
    uint32_t activeCountB32 = activeGroups;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg groupMaskB16 = MicroAPI::UpdateMask<uint16_t>(groupCountB16);
        MicroAPI::MaskReg activeMaskB32 = MicroAPI::UpdateMask<int32_t>(activeCountB32);
        MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t>();

        // Load groupMax and unpack to B32.
        MicroAPI::RegTensor<uint16_t> maxRankB16;
        MicroAPI::RegTensor<uint32_t> maxRankB32;
        MicroAPI::DataCopy(maxRankB16, groupMax);
        MicroAPI::UnPack<uint32_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(maxRankB32, maxRankB16);

        // BitsForMaxRank(maxRank) = 1 + sum(maxRank >= threshold) for {2,4,8,16,32,64,128}.
        // Active groups: full computation.  Non-active groups: 0 (mask excludes them).
        MicroAPI::RegTensor<int32_t> zeroB32;
        MicroAPI::RegTensor<int32_t> oneB32;
        MicroAPI::Duplicate(zeroB32, 0, allMaskB32);
        MicroAPI::Duplicate(oneB32, 1, allMaskB32);

        // Start with 1 for active groups, 0 for non-active.
        MicroAPI::RegTensor<int32_t> bitsB32;
        MicroAPI::Select(bitsB32, oneB32, zeroB32, activeMaskB32);

        // 7 threshold comparisons, each adds 1 if maxRank >= threshold.
        // CompareScalar uses activeMaskB32 as predicate: non-active lanes produce
        // mask=0, so Select picks zeroB32 (0) and Add is a no-op for them.
        MicroAPI::MaskReg cmpMask;
        MicroAPI::RegTensor<int32_t> cmpResultB32;
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 2u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 4u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 8u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 16u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 32u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 64u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 128u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        // Store groupBits (B32 -> B16 -> UB).
        MicroAPI::RegTensor<uint16_t> bitsB16;
        MicroAPI::Pack(bitsB16, (MicroAPI::RegTensor<uint32_t>&)bitsB32);
        MicroAPI::DataCopy(groupBits, bitsB16, groupMaskB16);

        // Compute overflow flags (counters + groupBits > 16).
        MicroAPI::RegTensor<int32_t> counterB32;
        MicroAPI::RegTensor<int32_t> totalBitsB32;
        MicroAPI::RegTensor<int32_t> limitB32;
        MicroAPI::DataCopy(counterB32, counters);
        MicroAPI::Add(totalBitsB32, counterB32, bitsB32, allMaskB32);
        MicroAPI::Duplicate(limitB32, 16, allMaskB32);
        MicroAPI::MaskReg overflowMask;
        MicroAPI::Compare<int32_t, CMPMODE::GT>(overflowMask, totalBitsB32, limitB32, activeMaskB32);

        // Hardware prefix sum (Unsqueeze = exclusive prefix over mask bits).
        MicroAPI::RegTensor<int32_t> prefixB32;
        MicroAPI::Unsqueeze(prefixB32, overflowMask);
        MicroAPI::DataCopy(overflowPrefix, prefixB32, allMaskB32);

        // Store overflow flags (mask -> B32 -> B16 -> B8 -> UB).
        MicroAPI::RegTensor<int32_t> overflowB32;
        MicroAPI::RegTensor<uint16_t> overflowB16;
        MicroAPI::RegTensor<uint8_t> overflowB8;
        MicroAPI::Select(overflowB32, oneB32, zeroB32, overflowMask);
        MicroAPI::Pack(overflowB16, (MicroAPI::RegTensor<uint32_t>&)overflowB32);
        MicroAPI::Pack(overflowB8, overflowB16);
        MicroAPI::MaskReg storeMaskB16;
        MicroAPI::MaskReg storeMaskB8;
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(storeMaskB16, allMaskB32);
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(storeMaskB8, storeMaskB16);
        MicroAPI::DataCopy(overflow, overflowB8, storeMaskB8);
    }
}

// ---------------------------------------------------------------------------
// FP32 fused metadata path.
//
// FP32 SplitAndMapTile does not produce groupMax.  This function derives one
// max-rank value per 64-value group from ranks[] and stages the 64 results in
// groupMaxScratch before running the same metadata logic as the BF16 path.
//
// Per-group maxRank derivation (one group per iteration):
//   - load one full B8 register (256 bytes) starting at ranks[group * 64];
//   - consume only the low 64 rank lanes;
//   - unpack B8 -> B16 -> B32;
//   - ReduceMax over the low 64 B32 lanes;
//   - store one u16 max rank to groupMaxScratch[group].
//
// The full-register load is wider than the logical 64-byte group.  The UB
// allocation order keeps the backing addresses readable, but this is a
// physical-layout contract: ranks must be followed by at least 192 bytes of
// readable UB.
// ---------------------------------------------------------------------------
template <bool FULL_TILE>
__aicore__ inline void PrepareGroupMetadataSimdFromRanks(__local_mem__ uint8_t* ranks, __local_mem__ int32_t* counters,
                                                         __local_mem__ uint16_t* groupBits,
                                                         __local_mem__ uint8_t* overflow,
                                                         __local_mem__ int32_t* overflowPrefix,
                                                         __local_mem__ uint16_t* groupMaxScratch, uint32_t valueCount)
{
    const uint32_t activeGroups = FULL_TILE ? GROUP_COUNT : valueCount / BLOCK_SIZE;
    uint32_t groupCountB16 = GROUP_COUNT;
    uint32_t activeCountB32 = activeGroups;
    // Per-group maxRank from ranks, staged through groupMaxScratch (UB).
    // ranks is STATE_COUNT=4096 uint8 = 64 groups x 64 values.  Each iteration
    // loads one group's 64 B8 ranks, unpacks to B32, ReduceMax -> maxRank, and
    // stores the single B16 maxRank to groupMaxScratch[group].
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg groupMaskB16 = MicroAPI::UpdateMask<uint16_t>(groupCountB16);
        MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t>();
        constexpr uint32_t VALUES_PER_GROUP = BLOCK_SIZE; // 64
        MicroAPI::RegTensor<uint8_t> rankB8;
        MicroAPI::RegTensor<uint16_t> rankB16;
        MicroAPI::RegTensor<uint32_t> rankB32;
        MicroAPI::RegTensor<uint32_t> maxRankB32;
        MicroAPI::RegTensor<uint16_t> maxRankB16;
        for (uint16_t group = 0; group < GROUP_COUNT; ++group) {
            // Load 256 bytes; only the low 64 lanes (this group) are used.
            MicroAPI::DataCopy(rankB8, ranks + group * VALUES_PER_GROUP);
            // 64 B8 -> 64 B16 -> 64 B32 (each lane = one rank value).
            MicroAPI::UnPack<uint16_t, uint8_t, MicroAPI::HighLowPart::LOWEST>(rankB16, rankB8);
            MicroAPI::UnPack<uint32_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(rankB32, rankB16);
            // ReduceMax over all 64 lanes = max rank of this group.
            MicroAPI::ReduceMax(maxRankB32, rankB32, allMaskB32);
            // Store the single B16 maxRank to groupMaxScratch[group].
            MicroAPI::Pack(maxRankB16, (MicroAPI::RegTensor<uint32_t>&)maxRankB32);
            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(groupMaxScratch + group,
                                                                                      maxRankB16, groupMaskB16);
        }
    }

    // The per-group loop stores maxRank to groupMaxScratch through the V pipe;
    // the next vector scope reloads the same UB region through the V pipe.  This
    // is a same-pipe store->load RAW dependency, so cross-pipe events do not
    // order it.  Drain PIPE_V explicitly before the reload.
    PipeBarrier<PIPE_V>();

    // Metadata generation: single __VEC_SCOPE__ block, identical to
    // PrepareGroupMetadataSimd.  groupBits is stored to UB for the caller, but
    // the overflow computation uses the register bitsB32 directly (no reload of
    // groupBits), so there is no store-to-load hazard within this block.
    __VEC_SCOPE__
    {
        // UpdateMask consumes its count. Do not reuse the count from the
        // max-rank scope, which is already zero in CPU execution.
        uint32_t metadataCountB16 = GROUP_COUNT;
        MicroAPI::MaskReg groupMaskB16 = MicroAPI::UpdateMask<uint16_t>(metadataCountB16);
        MicroAPI::MaskReg activeMaskB32 = MicroAPI::UpdateMask<int32_t>(activeCountB32);
        MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t>();

        // Load maxRank from scratch and unpack to B32.
        MicroAPI::RegTensor<uint16_t> maxRankB16;
        MicroAPI::RegTensor<uint32_t> maxRankB32;
        MicroAPI::DataCopy(maxRankB16, groupMaxScratch);
        MicroAPI::UnPack<uint32_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(maxRankB32, maxRankB16);

        // BitsForMaxRank(maxRank) = 1 + sum(maxRank >= threshold) for {2,4,8,16,32,64,128}.
        // Active groups: full computation.  Non-active groups: 0 (mask excludes them).
        MicroAPI::RegTensor<int32_t> zeroB32;
        MicroAPI::RegTensor<int32_t> oneB32;
        MicroAPI::Duplicate(zeroB32, 0, allMaskB32);
        MicroAPI::Duplicate(oneB32, 1, allMaskB32);
        MicroAPI::RegTensor<int32_t> bitsB32;
        MicroAPI::Select(bitsB32, oneB32, zeroB32, activeMaskB32);
        MicroAPI::MaskReg cmpMask;
        MicroAPI::RegTensor<int32_t> cmpResultB32;
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 2u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 4u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 8u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 16u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 32u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 64u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(cmpMask, maxRankB32, 128u, activeMaskB32);
        MicroAPI::Select(cmpResultB32, oneB32, zeroB32, cmpMask);
        MicroAPI::Add(bitsB32, bitsB32, cmpResultB32, allMaskB32);

        // Store groupBits (B32 -> B16 -> UB).
        MicroAPI::RegTensor<uint16_t> bitsB16;
        MicroAPI::Pack(bitsB16, (MicroAPI::RegTensor<uint32_t>&)bitsB32);
        MicroAPI::DataCopy(groupBits, bitsB16, groupMaskB16);

        // Compute overflow flags (counters + groupBits > 16).
        // Uses the register bitsB32 directly (no reload of groupBits from UB).
        MicroAPI::RegTensor<int32_t> counterB32;
        MicroAPI::RegTensor<int32_t> totalBitsB32;
        MicroAPI::RegTensor<int32_t> limitB32;
        MicroAPI::DataCopy(counterB32, counters);
        MicroAPI::Add(totalBitsB32, counterB32, bitsB32, allMaskB32);
        MicroAPI::Duplicate(limitB32, 16, allMaskB32);
        MicroAPI::MaskReg overflowMask;
        MicroAPI::Compare<int32_t, CMPMODE::GT>(overflowMask, totalBitsB32, limitB32, activeMaskB32);

        // Hardware prefix sum (Unsqueeze = exclusive prefix over mask bits).
        MicroAPI::RegTensor<int32_t> prefixB32;
        MicroAPI::Unsqueeze(prefixB32, overflowMask);
        MicroAPI::DataCopy(overflowPrefix, prefixB32, allMaskB32);

        // Store overflow flags (mask -> B32 -> B16 -> B8 -> UB).
        MicroAPI::RegTensor<int32_t> overflowB32;
        MicroAPI::RegTensor<uint16_t> overflowB16;
        MicroAPI::RegTensor<uint8_t> overflowB8;
        MicroAPI::Select(overflowB32, oneB32, zeroB32, overflowMask);
        MicroAPI::Pack(overflowB16, (MicroAPI::RegTensor<uint32_t>&)overflowB32);
        MicroAPI::Pack(overflowB8, overflowB16);
        MicroAPI::MaskReg storeMaskB16;
        MicroAPI::MaskReg storeMaskB8;
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(storeMaskB16, allMaskB32);
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(storeMaskB8, storeMaskB16);
        MicroAPI::DataCopy(overflow, overflowB8, storeMaskB8);
    }
}

template <bool FULL_TILE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void UpdateStatesAndPackRecord(
    const __ubuf__ uint8_t* ranks, __ubuf__ uint32_t* states, __ubuf__ int32_t* counters,
    const __ubuf__ uint16_t* groupBits, const __ubuf__ uint8_t* overflow, const __ubuf__ int32_t* overflowPrefix,
    __ubuf__ uint16_t* record, uint32_t valueCount, uint32_t overflowCount)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    // Assign sixteen threads to each 64-value group and four values to each
    // thread.  bits/overflow/prefix are consequently loaded once per four
    // values instead of once per value.  At every unrolled step the sixteen
    // threads touch consecutive lanes, keeping state/rank/record UB accesses
    // coalesced instead of producing a stride-4 bank pattern.
    const uint32_t group = threadIdx / UPDATE_THREADS_PER_GROUP;
    const uint32_t laneBlock = threadIdx % UPDATE_THREADS_PER_GROUP;
    const uint32_t activeGroups = FULL_TILE ? GROUP_COUNT : valueCount / BLOCK_SIZE;
    if (FULL_TILE || group < activeGroups) {
        const uint32_t bits = static_cast<uint32_t>(groupBits[group]);
        const bool doesOverflow = overflow[group] != 0;
        const uint32_t outputBase = static_cast<uint32_t>(overflowPrefix[group]) * BLOCK_SIZE;
        const uint32_t valueStart = group * BLOCK_SIZE;
        for (uint32_t value = 0; value < UPDATE_VALUES_PER_THREAD; ++value) {
            const uint32_t lane = laneBlock + value * UPDATE_THREADS_PER_GROUP;
            const uint32_t index = valueStart + lane;
            uint32_t state = (states[index] << bits) + static_cast<uint32_t>(ranks[index]);
            if (doesOverflow) {
                record[outputBase + lane] = static_cast<uint16_t>(state & STATE_LOW_MASK);
                state >>= 16;
            }
            states[index] = state;
        }
        if (laneBlock == 0) {
            counters[group] += static_cast<int32_t>(bits);
            if (doesOverflow) {
                counters[group] -= 16;
            }
        }
    }
    if (threadIdx < GROUP_COUNT) {
        record[overflowCount * BLOCK_SIZE + threadIdx] = groupBits[threadIdx];
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void PackStateTail(const __ubuf__ uint32_t* states,
                                                                            __ubuf__ uint16_t* stateTail)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum<0>();
    for (uint32_t index = threadIdx; index < STATE_COUNT; index += threadCount) {
        stateTail[index] = static_cast<uint16_t>(states[index] & STATE_LOW_MASK);
    }
}

// Var-suffix split.  Unlike the fixed path, the var path stores only the
// original symbol byte and mantissa bytes; rank/groupMax have no consumer.
// Keeping this path specialized avoids LUT-gather/rank-pack/rank-store
// overhead for data that never enters the fixed HANS state machine.
template <int32_t BYTES, bool FULL_TILE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void SplitVarTileSimt(const __ubuf__ uint8_t* raw,
                                                                               __ubuf__ uint8_t* symbols,
                                                                               __ubuf__ uint8_t* mantissa,
                                                                               uint32_t valueCount)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum<0>();
    const uint32_t end = FULL_TILE ? STATE_COUNT : valueCount;
    for (uint32_t index = threadIdx; index < end; index += threadCount) {
        const uint32_t rawOffset = index * BYTES;
        const uint32_t mantissaOffset = index * (BYTES - 1);
        symbols[index] = raw[rawOffset + BYTES - 1];
        for (uint32_t byte = 0; byte < static_cast<uint32_t>(BYTES - 1); ++byte) {
            mantissa[mantissaOffset + byte] = raw[rawOffset + byte];
        }
    }
}

template <bool FULL_TILE>
__aicore__ inline void SplitVarTileBf16RegBase(__local_mem__ uint16_t* raw, __local_mem__ uint8_t* symbols,
                                               __local_mem__ uint8_t* mantissa, uint32_t valueCount)
{
    constexpr uint32_t VALUES_PER_REG = 128;
    const uint16_t repeatTimes = FULL_TILE ? STATE_COUNT / VALUES_PER_REG :
                                             (valueCount + VALUES_PER_REG - 1) / VALUES_PER_REG;
    uint32_t remaining = FULL_TILE ? STATE_COUNT : valueCount;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> rawReg;
        MicroAPI::RegTensor<uint16_t> symbolReg;
        MicroAPI::RegTensor<uint8_t> symbolPacked;
        MicroAPI::RegTensor<uint8_t> mantissaPacked;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            MicroAPI::MaskReg maskB16 = MicroAPI::UpdateMask<uint16_t>(remaining);
            MicroAPI::MaskReg maskB8;
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(maskB8, maskB16);
            MicroAPI::DataCopy(rawReg, raw + repeat * VALUES_PER_REG);
            MicroAPI::ShiftRights<uint16_t, int16_t>(symbolReg, rawReg, 8, maskB16);
            MicroAPI::Pack(mantissaPacked, rawReg);
            MicroAPI::Pack(symbolPacked, symbolReg);
            MicroAPI::DataCopy(mantissa + repeat * VALUES_PER_REG, mantissaPacked, maskB8);
            MicroAPI::DataCopy(symbols + repeat * VALUES_PER_REG, symbolPacked, maskB8);
        }
    }
}

template <int32_t DTYPE_BYTES>
class HansEncodeSimt {
    // HANS arch35 supports floating-point layouts with one low-entropy symbol byte
    // and (N-1) mantissa bytes. Production entry points instantiate only <2> and <4>.
    static_assert(DTYPE_BYTES == 2 || DTYPE_BYTES == 4,
                  "HANS arch35 supports only 2-byte (BF16/FP16) and 4-byte (FP32) inputs");

    // LocalMemAllocator is a bump allocator, so enforce the usable UB budget at
    // compile time. USABLE_UB_BYTES already excludes the SIMT DCache reservation.
    static constexpr int32_t kUbBudgetBytes = STATE_COUNT * DTYPE_BYTES * 3 // rawLocal + rawLocalE5b + rawLocalB
                                              + STATE_COUNT * 3             // symbols x2 + ranks
                                              + STATE_COUNT * (DTYPE_BYTES - 1) * 2 // mantissa x2
                                              + STATE_COUNT                         // e5LayoutReserve
                                              + STATE_COUNT * 4                     // states
                                              + GROUP_COUNT * 4                     // counters
                                              + PDF_LENGTH * 4 * 2                  // pdf + histogram
                                              + PDF_LENGTH                          // symbolToRank
                                              + (STATE_COUNT + GROUP_COUNT) * 2     // record
                                              + GROUP_COUNT *
                                                    (2 + 2 + 1 + 4)     // groupMax/groupBits/overflow/overflowPrefix
                                              + HEADER_INT32_COUNT * 4; // header
    static_assert(kUbBudgetBytes <= USABLE_UB_BYTES,
                  "HANS encode UB allocation exceeds the usable budget after SIMT DCache reservation");

public:
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR pdf, GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var,
                                   GM_ADDR workspace, const HansEncodeTilingData* tiling)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= tiling->processCoreDim) {
            return;
        }

        tiling_ = tiling;
        InitGlobalTensors(input, pdf, mantissa, fixed, var);
        HANS_ENC_DBG("Process entry: processCoreDim=%d reshuff=%d statistic=%d",
                     static_cast<int>(tiling->processCoreDim), static_cast<int>(tiling->reshuff),
                     static_cast<int>(tiling->statistic));

        // Keep all UB allocations centralized here. Allocation order is part of
        // the validated physical layout and must not change during phase refactoring.
        LocalMemAllocator<Hardware::UB> allocator;
        LocalTensor<uint8_t> rawLocal = allocator.Alloc<uint8_t>(STATE_COUNT * DTYPE_BYTES);
        LocalTensor<uint8_t> symbolsLocal = allocator.Alloc<uint8_t>(STATE_COUNT);
        LocalTensor<uint8_t> mantissaLocal = allocator.Alloc<uint8_t>(STATE_COUNT * (DTYPE_BYTES - 1));
        LocalTensor<uint8_t> ranksLocal = allocator.Alloc<uint8_t>(STATE_COUNT);

        // Second buffer set for var-suffix double buffering.
        LocalTensor<uint8_t> rawLocalE5b = allocator.Alloc<uint8_t>(STATE_COUNT * DTYPE_BYTES);
        LocalTensor<uint8_t> symbolsLocalE5b = allocator.Alloc<uint8_t>(STATE_COUNT);
        LocalTensor<uint8_t> mantissaLocalE5b = allocator.Alloc<uint8_t>(STATE_COUNT * (DTYPE_BYTES - 1));

        // Preserve the validated UB layout even though the var path no longer
        // needs a second ranks buffer.
        LocalTensor<uint8_t> e5LayoutReserve = allocator.Alloc<uint8_t>(STATE_COUNT);

        // Second raw buffer used only by histogram double buffering.
        LocalTensor<uint8_t> rawLocalB = allocator.Alloc<uint8_t>(STATE_COUNT * DTYPE_BYTES);
        LocalTensor<uint32_t> statesLocal = allocator.Alloc<uint32_t>(STATE_COUNT);
        LocalTensor<int32_t> countersLocal = allocator.Alloc<int32_t>(GROUP_COUNT);
        LocalTensor<int32_t> pdfLocal = allocator.Alloc<int32_t>(PDF_LENGTH);

        // pdfLocal occupies 1024 bytes, so histogramLocal naturally remains
        // 32-byte aligned for RegBase UB DataCopy operations.
        LocalTensor<int32_t> histogramLocal = allocator.Alloc<int32_t>(PDF_LENGTH);
        LocalTensor<uint8_t> symbolToRankLocal = allocator.Alloc<uint8_t>(PDF_LENGTH);
        LocalTensor<uint16_t> recordLocal = allocator.Alloc<uint16_t>(STATE_COUNT + GROUP_COUNT);
        LocalTensor<uint16_t> groupMaxLocal = allocator.Alloc<uint16_t>(GROUP_COUNT);
        LocalTensor<uint16_t> groupBitsLocal = allocator.Alloc<uint16_t>(GROUP_COUNT);
        LocalTensor<uint8_t> overflowLocal = allocator.Alloc<uint8_t>(GROUP_COUNT);
        LocalTensor<int32_t> overflowPrefixLocal = allocator.Alloc<int32_t>(GROUP_COUNT);
        LocalTensor<int32_t> headerLocal = allocator.Alloc<int32_t>(HEADER_INT32_COUNT);

        InitializeLocalState(statesLocal, countersLocal, histogramLocal, tiling->statistic);

        const int64_t loopsCurrentCore = blockIdx < tiling->processCoreDim - 1 ? tiling->processLoopPerCore :
                                                                                 tiling->processLoopLastCore;
        const int64_t valuesCurrentCore = loopsCurrentCore * BLOCK_SIZE;
        const int64_t valueStart = tiling->processLoopPerCore * BLOCK_SIZE * blockIdx;
        const int64_t slotCapacity = blockIdx < tiling->processCoreDim - 1 ? tiling->fixedLengthPerCore :
                                                                             tiling->fixedLengthLastCore;
        HANS_ENC_DBG(
            "tiling: loopsCurrentCore=%lld valuesCurrentCore=%lld valueStart=%lld slotCapacity=%lld "
            "processLoopPerCore=%lld processLoopLastCore=%lld fixedLengthPerCore=%lld fixedLengthLastCore=%lld",
            static_cast<long long>(loopsCurrentCore), static_cast<long long>(valuesCurrentCore),
            static_cast<long long>(valueStart), static_cast<long long>(slotCapacity),
            static_cast<long long>(tiling->processLoopPerCore), static_cast<long long>(tiling->processLoopLastCore),
            static_cast<long long>(tiling->fixedLengthPerCore), static_cast<long long>(tiling->fixedLengthLastCore));

        InitializeSharedOutputs(headerLocal, pdfLocal, tiling->statistic);
        HANS_ENC_DBG("InitializeSharedOutputs done");

        RunHistogramPhase(rawLocal, rawLocalB, histogramLocal, valueStart, valuesCurrentCore, tiling->statistic);
        HANS_ENC_DBG("histogram phase done (statistic=%d)", static_cast<int>(tiling->statistic));

        BuildRankTable(pdfLocal, symbolToRankLocal);
        HANS_ENC_DBG("BuildStableRank done");

        GlobalTensor<uint8_t> payloadGm;
        __gm__ uint8_t* payloadBase = tiling->reshuff ? reinterpret_cast<__gm__ uint8_t*>(workspace) :
                                                        reinterpret_cast<__gm__ uint8_t*>(fixed);
        payloadGm.SetGlobalBuffer(payloadBase + HEADER_BYTES + tiling->fixedLengthPerCore * blockIdx);
        HANS_ENC_DBG("payloadGm base offset=%lld (reshuff=%d)",
                     static_cast<long long>(HEADER_BYTES + tiling->fixedLengthPerCore * blockIdx),
                     static_cast<int>(tiling->reshuff));

        int64_t processedValues = 0;
        int64_t fixedUsedBytes = 0;
        RunFixedEncodePhase(rawLocal, symbolsLocal, mantissaLocal, ranksLocal, symbolToRankLocal, statesLocal,
                            countersLocal, groupMaxLocal, groupBitsLocal, overflowLocal, overflowPrefixLocal,
                            recordLocal, payloadGm, valueStart, valuesCurrentCore, slotCapacity, processedValues,
                            fixedUsedBytes);

        const int64_t varValues = valuesCurrentCore - processedValues;
        PublishAndAggregateHeader(headerLocal, fixedUsedBytes, varValues, blockIdx, tiling);
        RefreshFinalHeader(headerLocal);

        CopyVarAndRemainingMantissa(rawLocal, symbolsLocal, mantissaLocal, valueStart, processedValues,
                                    valuesCurrentCore, rawLocalE5b, symbolsLocalE5b, mantissaLocalE5b, headerLocal,
                                    blockIdx);
        HANS_ENC_DBG("CopyVarAndRemainingMantissa done");

        if (tiling->reshuff) {
            CompactPayload(rawLocal, payloadGm, fixedUsedBytes, blockIdx, headerLocal);
            HANS_ENC_DBG("CompactPayload done");
        }
        HANS_ENC_DBG("Process exit");
    }

private:
    GlobalTensor<uint8_t> inputGm_;
    GlobalTensor<int32_t> pdfGm_;
    GlobalTensor<uint8_t> mantissaGm_;
    GlobalTensor<uint8_t> fixedGm_;
    GlobalTensor<uint8_t> varGm_;
    GlobalTensor<int32_t> headerGm_;
    const HansEncodeTilingData* tiling_ = nullptr;

    __aicore__ inline void InitGlobalTensors(GM_ADDR input, GM_ADDR pdf, GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input));
        pdfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pdf), PDF_LENGTH);
        mantissaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mantissa));
        fixedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(fixed));
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(var));
        headerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(fixed), HEADER_INT32_COUNT);
    }

    __aicore__ inline void InitializeLocalState(LocalTensor<uint32_t> statesLocal, LocalTensor<int32_t> countersLocal,
                                                LocalTensor<int32_t> histogramLocal, bool statistic)
    {
        Duplicate<uint32_t>(statesLocal, 0, STATE_COUNT);
        Duplicate<int32_t>(countersLocal, 0, GROUP_COUNT);
        // histogramLocal is accumulated across tiles by the RegBase histogram
        // kernels, so it must start from zero whenever statistics are rebuilt.
        if (statistic) {
            Duplicate<int32_t>(histogramLocal, 0, PDF_LENGTH);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunHistogramPhase(LocalTensor<uint8_t> rawLocal, LocalTensor<uint8_t> rawLocalB,
                                             LocalTensor<int32_t> histogramLocal, int64_t valueStart,
                                             int64_t valuesCurrentCore, bool statistic)
    {
        if (!statistic) {
            return;
        }

        ComputeHistogram(rawLocal, histogramLocal, valueStart, valuesCurrentCore, rawLocalB);
        SetAtomicAdd<int32_t>();
        DataCopy(pdfGm_, histogramLocal, PDF_LENGTH);
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
        SyncAll();
    }

    __aicore__ inline void BuildRankTable(LocalTensor<int32_t> pdfLocal, LocalTensor<uint8_t> symbolToRankLocal)
    {
        DataCopy(pdfLocal, pdfGm_, PDF_LENGTH);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Simt::VF_CALL<BuildStableRank>(Simt::Dim3{PDF_LENGTH}, (__ubuf__ int32_t*)pdfLocal.GetPhyAddr(),
                                       (__ubuf__ uint8_t*)symbolToRankLocal.GetPhyAddr());
    }

    __aicore__ inline void RunFixedEncodePhase(
        LocalTensor<uint8_t> rawLocal, LocalTensor<uint8_t> symbolsLocal, LocalTensor<uint8_t> mantissaLocal,
        LocalTensor<uint8_t> ranksLocal, LocalTensor<uint8_t> symbolToRankLocal, LocalTensor<uint32_t> statesLocal,
        LocalTensor<int32_t> countersLocal, LocalTensor<uint16_t> groupMaxLocal, LocalTensor<uint16_t> groupBitsLocal,
        LocalTensor<uint8_t> overflowLocal, LocalTensor<int32_t> overflowPrefixLocal, LocalTensor<uint16_t> recordLocal,
        GlobalTensor<uint8_t> payloadGm, int64_t valueStart, int64_t valuesCurrentCore, int64_t slotCapacity,
        int64_t& processedValues, int64_t& fixedUsedBytes)
    {
        HANS_ENC_DBG("entering fixed loop: slotCapacity=%lld TAIL_BYTES=%lld slotCapacity>=TAIL_BYTES=%d",
                     static_cast<long long>(slotCapacity), static_cast<long long>(TAIL_BYTES),
                     static_cast<int>(slotCapacity >= TAIL_BYTES));

        if (slotCapacity >= TAIL_BYTES) {
            while (processedValues < valuesCurrentCore) {
                const uint32_t tileValues = static_cast<uint32_t>(valuesCurrentCore - processedValues > STATE_COUNT ?
                                                                      STATE_COUNT :
                                                                      valuesCurrentCore - processedValues);

                if (tileValues == STATE_COUNT) {
                    CopyAndSplitTile<false, true>(rawLocal, symbolsLocal, mantissaLocal, ranksLocal, symbolToRankLocal,
                                                  groupMaxLocal, valueStart + processedValues, tileValues);
                } else {
                    CopyAndSplitTile<false, false>(rawLocal, symbolsLocal, mantissaLocal, ranksLocal, symbolToRankLocal,
                                                   groupMaxLocal, valueStart + processedValues, tileValues);
                }

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(mantissaGm_[(valueStart + processedValues) * (DTYPE_BYTES - 1)], mantissaLocal,
                         tileValues * (DTYPE_BYTES - 1));
                // Metadata and state update do not read mantissaLocal. Keep this
                // MTE3 store in flight while those phases execute on V.

                if (tileValues == STATE_COUNT) {
                    if constexpr (DTYPE_BYTES == 2) {
                        PrepareGroupMetadataSimd<true>((__local_mem__ uint16_t*)groupMaxLocal.GetPhyAddr(),
                                                       (__local_mem__ int32_t*)countersLocal.GetPhyAddr(),
                                                       (__local_mem__ uint16_t*)groupBitsLocal.GetPhyAddr(),
                                                       (__local_mem__ uint8_t*)overflowLocal.GetPhyAddr(),
                                                       (__local_mem__ int32_t*)overflowPrefixLocal.GetPhyAddr(),
                                                       tileValues);
                    } else {
                        PrepareGroupMetadataSimdFromRanks<true>(
                            (__local_mem__ uint8_t*)ranksLocal.GetPhyAddr(),
                            (__local_mem__ int32_t*)countersLocal.GetPhyAddr(),
                            (__local_mem__ uint16_t*)groupBitsLocal.GetPhyAddr(),
                            (__local_mem__ uint8_t*)overflowLocal.GetPhyAddr(),
                            (__local_mem__ int32_t*)overflowPrefixLocal.GetPhyAddr(),
                            (__local_mem__ uint16_t*)groupMaxLocal.GetPhyAddr(), tileValues);
                    }
                } else {
                    if constexpr (DTYPE_BYTES == 2) {
                        PrepareGroupMetadataSimd<false>((__local_mem__ uint16_t*)groupMaxLocal.GetPhyAddr(),
                                                        (__local_mem__ int32_t*)countersLocal.GetPhyAddr(),
                                                        (__local_mem__ uint16_t*)groupBitsLocal.GetPhyAddr(),
                                                        (__local_mem__ uint8_t*)overflowLocal.GetPhyAddr(),
                                                        (__local_mem__ int32_t*)overflowPrefixLocal.GetPhyAddr(),
                                                        tileValues);
                    } else {
                        PrepareGroupMetadataSimdFromRanks<false>(
                            (__local_mem__ uint8_t*)ranksLocal.GetPhyAddr(),
                            (__local_mem__ int32_t*)countersLocal.GetPhyAddr(),
                            (__local_mem__ uint16_t*)groupBitsLocal.GetPhyAddr(),
                            (__local_mem__ uint8_t*)overflowLocal.GetPhyAddr(),
                            (__local_mem__ int32_t*)overflowPrefixLocal.GetPhyAddr(),
                            (__local_mem__ uint16_t*)groupMaxLocal.GetPhyAddr(), tileValues);
                    }
                }

                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                const uint32_t overflowCount = static_cast<uint32_t>(overflowPrefixLocal.GetValue(GROUP_COUNT - 1) +
                                                                     overflowLocal.GetValue(GROUP_COUNT - 1));
                HANS_ENC_DBG(
                    "metadata: processed=%lld rank0=%u max0=%u bits0=%u counter0=%d prefixLast=%d flagLast=%u count=%u",
                    static_cast<long long>(processedValues), static_cast<unsigned>(ranksLocal.GetValue(0)),
                    static_cast<unsigned>(groupMaxLocal.GetValue(0)), static_cast<unsigned>(groupBitsLocal.GetValue(0)),
                    countersLocal.GetValue(0), overflowPrefixLocal.GetValue(GROUP_COUNT - 1),
                    static_cast<unsigned>(overflowLocal.GetValue(GROUP_COUNT - 1)), overflowCount);
#if HANS_ENCODE_DEBUG
                if (processedValues <= 2 * STATE_COUNT) {
                    uint32_t expectedPrefix = 0;
                    uint32_t prefixMismatches = 0;
                    uint32_t flagMismatches = 0;
                    for (uint32_t group = 0; group < GROUP_COUNT; ++group) {
                        prefixMismatches += overflowPrefixLocal.GetValue(group) != static_cast<int32_t>(expectedPrefix);
                        const uint32_t expectedFlag = group < tileValues / BLOCK_SIZE &&
                                                      countersLocal.GetValue(group) + groupBitsLocal.GetValue(group) >
                                                          16;
                        flagMismatches += overflowLocal.GetValue(group) != expectedFlag;
                        expectedPrefix += expectedFlag;
                    }
                    HANS_ENC_DBG("prefix check: expectedCount=%u prefixMismatches=%u flagMismatches=%u "
                                 "prefix0=%d prefix1=%d prefix62=%d bits63=%u counter63=%d",
                                 expectedPrefix, prefixMismatches, flagMismatches, overflowPrefixLocal.GetValue(0),
                                 overflowPrefixLocal.GetValue(1), overflowPrefixLocal.GetValue(GROUP_COUNT - 2),
                                 static_cast<unsigned>(groupBitsLocal.GetValue(GROUP_COUNT - 1)),
                                 countersLocal.GetValue(GROUP_COUNT - 1));
                }
#endif
                const int64_t recordBytes = GROUP_BITS_BYTES + overflowCount * GROUP_OVERFLOW_BYTES;
                if (fixedUsedBytes + recordBytes + TAIL_BYTES > slotCapacity) {
                    break;
                }

                if (tileValues == STATE_COUNT) {
                    Simt::VF_CALL<UpdateStatesAndPackRecord<true>>(
                        Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint8_t*)ranksLocal.GetPhyAddr(),
                        (__ubuf__ uint32_t*)statesLocal.GetPhyAddr(), (__ubuf__ int32_t*)countersLocal.GetPhyAddr(),
                        (__ubuf__ uint16_t*)groupBitsLocal.GetPhyAddr(), (__ubuf__ uint8_t*)overflowLocal.GetPhyAddr(),
                        (__ubuf__ int32_t*)overflowPrefixLocal.GetPhyAddr(),
                        (__ubuf__ uint16_t*)recordLocal.GetPhyAddr(), tileValues, overflowCount);
                } else {
                    Simt::VF_CALL<UpdateStatesAndPackRecord<false>>(
                        Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint8_t*)ranksLocal.GetPhyAddr(),
                        (__ubuf__ uint32_t*)statesLocal.GetPhyAddr(), (__ubuf__ int32_t*)countersLocal.GetPhyAddr(),
                        (__ubuf__ uint16_t*)groupBitsLocal.GetPhyAddr(), (__ubuf__ uint8_t*)overflowLocal.GetPhyAddr(),
                        (__ubuf__ int32_t*)overflowPrefixLocal.GetPhyAddr(),
                        (__ubuf__ uint16_t*)recordLocal.GetPhyAddr(), tileValues, overflowCount);
                }

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(payloadGm[fixedUsedBytes], recordLocal.ReinterpretCast<uint8_t>(), recordBytes);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);

                fixedUsedBytes += recordBytes;
                processedValues += tileValues;
                HANS_ENC_DBG("tile done: tileValues=%u overflowCount=%u recordBytes=%lld fixedUsedBytes=%lld "
                             "processedValues=%lld",
                             tileValues, overflowCount, static_cast<long long>(recordBytes),
                             static_cast<long long>(fixedUsedBytes), static_cast<long long>(processedValues));
            }

            Simt::VF_CALL<PackStateTail>(Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint32_t*)statesLocal.GetPhyAddr(),
                                         (__ubuf__ uint16_t*)recordLocal.GetPhyAddr());
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(payloadGm[fixedUsedBytes], recordLocal.ReinterpretCast<uint8_t>(), STATE_TAIL_BYTES);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            DataCopy(payloadGm[fixedUsedBytes + STATE_TAIL_BYTES], countersLocal.ReinterpretCast<uint8_t>(),
                     COUNTER_TAIL_BYTES);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
            fixedUsedBytes += TAIL_BYTES;
        }

        HANS_ENC_DBG("fixed loop done: fixedUsedBytes=%lld processedValues=%lld (out of valuesCurrentCore=%lld)",
                     static_cast<long long>(fixedUsedBytes), static_cast<long long>(processedValues),
                     static_cast<long long>(valuesCurrentCore));
    }

    __aicore__ inline void PublishAndAggregateHeader(LocalTensor<int32_t> headerLocal, int64_t fixedUsedBytes,
                                                     int64_t varValues, int64_t blockIdx,
                                                     const HansEncodeTilingData* tiling)
    {
        HANS_ENC_DBG("writing per-core header via DMA: DEVICE_START_IDX+%d=%lld HOST_START_IDX+%d=%lld",
                     static_cast<int>(blockIdx), static_cast<long long>(fixedUsedBytes), static_cast<int>(blockIdx),
                     static_cast<long long>(varValues));

        // Each core publishes only its own per-core fields through atomic DMA.
        Duplicate<int32_t>(headerLocal, 0, HEADER_INT32_COUNT);
        // SetValue uses S: a V-only barrier does not order these scalar writes.
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        headerLocal.SetValue(DEVICE_START_IDX + blockIdx, static_cast<int32_t>(fixedUsedBytes));
        headerLocal.SetValue(HOST_START_IDX + blockIdx, static_cast<int32_t>(varValues));
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        SetAtomicAdd<int32_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(HEADER_BYTES), 0, 0};
        DataCopyPad(headerGm_, headerLocal, copyParams);
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);

        SyncAll();
        HANS_ENC_DBG("after per-core DMA header write + SyncAll");

        // Core 0 computes aggregate fields after all per-core slots are visible.
        if (blockIdx == 0) {
            DataCopy(headerLocal, headerGm_, HEADER_INT32_COUNT);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);

            int32_t totalVarValues = 0;
            for (int32_t core = 0; core < tiling->processCoreDim; ++core) {
                const int32_t hostVal = headerLocal.GetValue(HOST_START_IDX + core);
                const int32_t devVal = headerLocal.GetValue(DEVICE_START_IDX + core);
                HANS_ENC_DBG("core0 readback core=%d: HOST[%d]=%d DEVICE[%d]=%d", core, HOST_START_IDX + core, hostVal,
                             DEVICE_START_IDX + core, devVal);
                totalVarValues += hostVal;
            }

            const int32_t totalLoops = static_cast<int32_t>(tiling->processLoopPerCore * (tiling->processCoreDim - 1) +
                                                            tiling->processLoopLastCore);
            headerLocal.SetValue(HEADER_MAGIC_IDX, MAGIC);
            headerLocal.SetValue(HEADER_CORE_COUNT_IDX, static_cast<int32_t>(tiling->processCoreDim));
            headerLocal.SetValue(HEADER_TOTAL_LOOPS_IDX, totalLoops);
            headerLocal.SetValue(HEADER_VAR_LOOPS_IDX, totalVarValues / BLOCK_SIZE);
            headerLocal.SetValue(HEADER_FIXED_LOOPS_IDX, totalLoops - totalVarValues / BLOCK_SIZE);
            HANS_ENC_DBG("core0 final header: magic=%d coreCount=%d totalLoops=%d varLoops=%d fixedLoops=%d "
                         "totalVarValues=%d",
                         MAGIC, static_cast<int>(tiling->processCoreDim), totalLoops, totalVarValues / BLOCK_SIZE,
                         totalLoops - totalVarValues / BLOCK_SIZE, totalVarValues);

            // Scalar updates in headerLocal must become visible before MTE3
            // overwrites the finalized header in GM.
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(headerGm_, headerLocal, copyParams);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }

        SyncAll();
        HANS_ENC_DBG("after core0 aggregation SyncAll");
    }

    __aicore__ inline void RefreshFinalHeader(LocalTensor<int32_t> headerLocal)
    {
        // Downstream prefix calculations use one coherent UB snapshot rather
        // than scalar GM reads of fields written by other cores.
        DataCopy(headerLocal, headerGm_, HEADER_INT32_COUNT);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }

    __aicore__ inline void InitializeSharedOutputs(LocalTensor<int32_t> headerLocal, LocalTensor<int32_t> pdfLocal,
                                                   bool statistic)
    {
        if (GetBlockIdx() == 0) {
            Duplicate<int32_t>(headerLocal, 0, HEADER_INT32_COUNT);
            if (statistic) {
                Duplicate<int32_t>(pdfLocal, 0, PDF_LENGTH);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(headerGm_, headerLocal, HEADER_INT32_COUNT);
            if (statistic) {
                DataCopy(pdfGm_, pdfLocal, PDF_LENGTH);
            }
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }
        SyncAll();
    }

    __aicore__ inline void ComputeHistogram(LocalTensor<uint8_t> rawLocal, LocalTensor<int32_t> histogramLocal,
                                            int64_t valueStart, int64_t valueCount, LocalTensor<uint8_t> rawLocalB)
    {
        if constexpr (DTYPE_BYTES == 2) {
            // Double-buffer: two rawLocal buffers (A=rawLocal, B=rawLocalB)
            // with distinct eventIds (0/1).  Even tiles use A, odd tiles use B.
            // The next tile's MTE2 prefetch overlaps the current tile's V dhistv2.
            // Only full tiles (STATE_COUNT) are double-buffered; the final partial
            // tile (if any) is handled serially with the padding logic.
            //
            // Pipeline per iteration:
            //   Wait(MTE2_V, cur)          -- current buffer's raw load done
            //   DataCopy(next buffer) + SetFlag(MTE2_V, next)  -- prefetch next tile
            //   AccumulateHistogramTileBf16RegBase(cur)  -- V dhistv2 on current
            //   Set/Wait(V_MTE2, cur)     -- V done before next even tile reuses cur
            {
                const int64_t fullTiles = valueCount / STATE_COUNT;
                const int64_t partialStart = fullTiles * STATE_COUNT;
                const uint32_t partialValues = static_cast<uint32_t>(valueCount - partialStart);

                if (fullTiles > 0) {
                    // Prefetch tile 0 into buffer A (eventId 0).
                    DataCopy(rawLocal, inputGm_[valueStart * DTYPE_BYTES], STATE_COUNT * DTYPE_BYTES);
                    SetFlag<HardEvent::MTE2_V>(0);

                    int64_t offset = 0;
                    uint32_t curBuf = 0; // 0 = A (rawLocal), 1 = B (rawLocalB)
                    while (offset < partialStart) {
                        const int64_t nextOffset = offset + STATE_COUNT;
                        // Wait for current buffer's MTE2 to complete.
                        WaitFlag<HardEvent::MTE2_V>(curBuf);
                        // Prefetch next full tile into the other buffer.
                        if (nextOffset < partialStart) {
                            const auto& prefetchBuf = (curBuf == 0) ? rawLocalB : rawLocal;
                            DataCopy(prefetchBuf, inputGm_[(valueStart + nextOffset) * DTYPE_BYTES],
                                     STATE_COUNT * DTYPE_BYTES);
                            SetFlag<HardEvent::MTE2_V>(curBuf ^ 1);
                        }
                        // V: dhistv2 on current buffer.
                        PipeBarrier<PIPE_V>();
                        const auto& curRaw = (curBuf == 0) ? rawLocal : rawLocalB;
                        AccumulateHistogramTileBf16RegBase<true>((__local_mem__ uint16_t*)curRaw.GetPhyAddr(),
                                                                 (__local_mem__ int32_t*)histogramLocal.GetPhyAddr(),
                                                                 STATE_COUNT);
                        // Ensure V done before the next same-parity tile's MTE2
                        // reuses this buffer.
                        SetFlag<HardEvent::V_MTE2>(curBuf);
                        WaitFlag<HardEvent::V_MTE2>(curBuf);
                        offset = nextOffset;
                        curBuf ^= 1;
                    }
                }

                // Handle the final partial tile serially (if any).
                if (partialValues > 0) {
                    const int64_t partialOffset = partialStart;
                    DataCopy(rawLocal, inputGm_[(valueStart + partialOffset) * DTYPE_BYTES],
                             partialValues * DTYPE_BYTES);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    PipeBarrier<PIPE_V>();
                    // dhistv2 always loads a full B16 register (256 B8 lanes)
                    // per raw0/raw1 regardless of the UpdateMask.  Pad the raw
                    // tail to a 256-lane boundary with zeros so every repeat is
                    // a complete load.  The zero-padded lanes map to symbol 0x00
                    // and are counted in bin 0; subtract (paddedValues -
                    // partialValues) from bin 0 after the accumulate.
                    constexpr uint32_t SAFE_LANE_ALIGN = 256; // == B8_LANES
                    const uint32_t paddedValues = ((partialValues + SAFE_LANE_ALIGN - 1) / SAFE_LANE_ALIGN) *
                                                  SAFE_LANE_ALIGN;
                    if (paddedValues > partialValues) {
                        const uint32_t padStartBytes = partialValues * DTYPE_BYTES;
                        const uint32_t padEndBytes = paddedValues * DTYPE_BYTES;
                        const uint32_t alignedStart = (padStartBytes + 31u) & ~31u;
                        for (uint32_t b = padStartBytes; b < alignedStart && b < padEndBytes; ++b) {
                            rawLocal.SetValue(b, static_cast<uint8_t>(0));
                        }
                        // SetValue is PIPE_S.  Order any scalar prefix bytes before
                        // the following PIPE_V Duplicate/Histogram operations.
                        if (padStartBytes < alignedStart && padStartBytes < padEndBytes) {
                            SetFlag<HardEvent::S_V>(0);
                            WaitFlag<HardEvent::S_V>(0);
                        }
                        if (alignedStart < padEndBytes) {
                            Duplicate<uint8_t>(rawLocal[alignedStart], 0, padEndBytes - alignedStart);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    AccumulateHistogramTileBf16RegBase<false>((__local_mem__ uint16_t*)rawLocal.GetPhyAddr(),
                                                              (__local_mem__ int32_t*)histogramLocal.GetPhyAddr(),
                                                              paddedValues);
                    if (paddedValues > partialValues) {
                        SetFlag<HardEvent::V_S>(0);
                        WaitFlag<HardEvent::V_S>(0);
                        histogramLocal.SetValue(
                            0, histogramLocal.GetValue(0) - static_cast<int32_t>(paddedValues - partialValues));
                        SetFlag<HardEvent::S_V>(0);
                        WaitFlag<HardEvent::S_V>(0);
                    }
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                }
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            return;
        }
        if constexpr (DTYPE_BYTES == 4) {
            // FP32 double-buffer (same pattern as BF16 above).
            {
                const int64_t fullTiles = valueCount / STATE_COUNT;
                const int64_t partialStart = fullTiles * STATE_COUNT;
                const uint32_t partialValues = static_cast<uint32_t>(valueCount - partialStart);

                if (fullTiles > 0) {
                    // Prefetch tile 0 into buffer A (eventId 0).
                    DataCopy(rawLocal, inputGm_[valueStart * DTYPE_BYTES], STATE_COUNT * DTYPE_BYTES);
                    SetFlag<HardEvent::MTE2_V>(0);

                    int64_t offset = 0;
                    uint32_t curBuf = 0;
                    while (offset < partialStart) {
                        const int64_t nextOffset = offset + STATE_COUNT;
                        WaitFlag<HardEvent::MTE2_V>(curBuf);
                        if (nextOffset < partialStart) {
                            const auto& prefetchBuf = (curBuf == 0) ? rawLocalB : rawLocal;
                            DataCopy(prefetchBuf, inputGm_[(valueStart + nextOffset) * DTYPE_BYTES],
                                     STATE_COUNT * DTYPE_BYTES);
                            SetFlag<HardEvent::MTE2_V>(curBuf ^ 1);
                        }
                        PipeBarrier<PIPE_V>();
                        const auto& curRaw = (curBuf == 0) ? rawLocal : rawLocalB;
                        AccumulateHistogramTileFp32RegBase<true>((__local_mem__ uint32_t*)curRaw.GetPhyAddr(),
                                                                 (__local_mem__ int32_t*)histogramLocal.GetPhyAddr(),
                                                                 STATE_COUNT);
                        SetFlag<HardEvent::V_MTE2>(curBuf);
                        WaitFlag<HardEvent::V_MTE2>(curBuf);
                        offset = nextOffset;
                        curBuf ^= 1;
                    }
                }

                // Handle the final partial tile serially (if any).
                if (partialValues > 0) {
                    const int64_t partialOffset = partialStart;
                    DataCopy(rawLocal, inputGm_[(valueStart + partialOffset) * DTYPE_BYTES],
                             partialValues * DTYPE_BYTES);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    PipeBarrier<PIPE_V>();
                    // dhistv2 always loads a full B32 register (64 uint32) per
                    // repeat.  Pad the raw tail to a 64-value boundary with zeros.
                    constexpr uint32_t SAFE_LANE_ALIGN = 64; // == VALUES_PER_REG (B32)
                    const uint32_t paddedValues = ((partialValues + SAFE_LANE_ALIGN - 1) / SAFE_LANE_ALIGN) *
                                                  SAFE_LANE_ALIGN;
                    if (paddedValues > partialValues) {
                        const uint32_t padStartBytes = partialValues * DTYPE_BYTES;
                        const uint32_t padEndBytes = paddedValues * DTYPE_BYTES;
                        const uint32_t alignedStart = (padStartBytes + 31u) & ~31u;
                        for (uint32_t b = padStartBytes; b < alignedStart && b < padEndBytes; ++b) {
                            rawLocal.SetValue(b, static_cast<uint8_t>(0));
                        }
                        // SetValue is PIPE_S.  Order any scalar prefix bytes before
                        // the following PIPE_V Duplicate/Histogram operations.
                        if (padStartBytes < alignedStart && padStartBytes < padEndBytes) {
                            SetFlag<HardEvent::S_V>(0);
                            WaitFlag<HardEvent::S_V>(0);
                        }
                        if (alignedStart < padEndBytes) {
                            Duplicate<uint8_t>(rawLocal[alignedStart], 0, padEndBytes - alignedStart);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    AccumulateHistogramTileFp32RegBase<false>((__local_mem__ uint32_t*)rawLocal.GetPhyAddr(),
                                                              (__local_mem__ int32_t*)histogramLocal.GetPhyAddr(),
                                                              paddedValues);
                    if (paddedValues > partialValues) {
                        SetFlag<HardEvent::V_S>(0);
                        WaitFlag<HardEvent::V_S>(0);
                        histogramLocal.SetValue(
                            0, histogramLocal.GetValue(0) - static_cast<int32_t>(paddedValues - partialValues));
                        SetFlag<HardEvent::S_V>(0);
                        WaitFlag<HardEvent::S_V>(0);
                    }
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                }
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            return;
        }
    }

    template <bool STORE_SYMBOLS, bool FULL_TILE>
    __aicore__ inline void CopyAndSplitTile(LocalTensor<uint8_t> rawLocal, LocalTensor<uint8_t> symbolsLocal,
                                            LocalTensor<uint8_t> mantissaLocal, LocalTensor<uint8_t> ranksLocal,
                                            LocalTensor<uint8_t> symbolToRankLocal, LocalTensor<uint16_t> groupMaxLocal,
                                            int64_t globalValueOffset, uint32_t tileValues)
    {
        DataCopy(rawLocal, inputGm_[globalValueOffset * DTYPE_BYTES], tileValues * DTYPE_BYTES);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (DTYPE_BYTES == 2) {
            SplitAndMapTileBf16RegBase<STORE_SYMBOLS, FULL_TILE>(
                (__local_mem__ uint16_t*)rawLocal.GetPhyAddr(), (__local_mem__ uint8_t*)symbolToRankLocal.GetPhyAddr(),
                (__local_mem__ uint8_t*)symbolsLocal.GetPhyAddr(), (__local_mem__ uint8_t*)mantissaLocal.GetPhyAddr(),
                (__local_mem__ uint8_t*)ranksLocal.GetPhyAddr(), (__local_mem__ uint16_t*)groupMaxLocal.GetPhyAddr(),
                tileValues);
            return;
        }
        Simt::VF_CALL<SplitAndMapTile<DTYPE_BYTES, STORE_SYMBOLS, FULL_TILE>>(
            Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint8_t*)rawLocal.GetPhyAddr(),
            (__ubuf__ uint8_t*)symbolToRankLocal.GetPhyAddr(), (__ubuf__ uint8_t*)symbolsLocal.GetPhyAddr(),
            (__ubuf__ uint8_t*)mantissaLocal.GetPhyAddr(), (__ubuf__ uint8_t*)ranksLocal.GetPhyAddr(), tileValues);
    }
    template <bool FULL_TILE>
    __aicore__ inline void SplitTileE5(LocalTensor<uint8_t> rawLocal, LocalTensor<uint8_t> symbolsLocal,
                                       LocalTensor<uint8_t> mantissaLocal, uint32_t tileValues)
    {
        if constexpr (DTYPE_BYTES == 2) {
            SplitVarTileBf16RegBase<FULL_TILE>((__local_mem__ uint16_t*)rawLocal.GetPhyAddr(),
                                               (__local_mem__ uint8_t*)symbolsLocal.GetPhyAddr(),
                                               (__local_mem__ uint8_t*)mantissaLocal.GetPhyAddr(), tileValues);
            return;
        }
        Simt::VF_CALL<SplitVarTileSimt<DTYPE_BYTES, FULL_TILE>>(
            Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint8_t*)rawLocal.GetPhyAddr(),
            (__ubuf__ uint8_t*)symbolsLocal.GetPhyAddr(), (__ubuf__ uint8_t*)mantissaLocal.GetPhyAddr(), tileValues);
    }

    __aicore__ inline void CopyVarAndRemainingMantissa(LocalTensor<uint8_t> rawLocal, LocalTensor<uint8_t> symbolsLocal,
                                                       LocalTensor<uint8_t> mantissaLocal, int64_t valueStart,
                                                       int64_t processedValues, int64_t valuesCurrentCore,
                                                       LocalTensor<uint8_t> rawLocalE5b,
                                                       LocalTensor<uint8_t> symbolsLocalE5b,
                                                       LocalTensor<uint8_t> mantissaLocalE5b,
                                                       LocalTensor<int32_t> headerLocal, int64_t blockIdx)
    {
        int64_t varOffset = 0;
        for (int32_t core = 0; core < blockIdx; ++core) {
            varOffset += headerLocal.GetValue(HOST_START_IDX + core);
        }
        const int64_t remainingValues = valuesCurrentCore - processedValues;
        HANS_ENC_DBG("CopyVar: varOffset=%lld remainingValues=%lld varLength=%lld blockIdx=%d",
                     static_cast<long long>(varOffset), static_cast<long long>(remainingValues),
                     static_cast<long long>(tiling_->varLength), static_cast<int>(GetBlockIdx()));
        if (varOffset + remainingValues > tiling_->varLength) {
            HANS_ENC_DBG("CopyVar: SKIP (varOffset+remaining > varLength)");
            return;
        }

        // Var-suffix double-buffer: event 0 owns buffer set A, event 1 owns B.
        // The current tile is split on V, its symbol/mantissa are stored through
        // MTE3, and the next raw tile is prefetched through MTE2.  MTE3_MTE2
        // participates in the transitive buffer-reuse ordering:
        //   MTE3 store -> next MTE2 load -> next V split/write.
        int64_t offset = processedValues;
        if (offset >= valuesCurrentCore) {
            return;
        }

        // Prefetch first tile into buffer set A.
        {
            const uint32_t firstTileValues = static_cast<uint32_t>(
                valuesCurrentCore - offset > STATE_COUNT ? STATE_COUNT : valuesCurrentCore - offset);
            DataCopy(rawLocal, inputGm_[(valueStart + offset) * DTYPE_BYTES], firstTileValues * DTYPE_BYTES);
            SetFlag<HardEvent::MTE2_V>(0);
        }

        while (offset < valuesCurrentCore) {
            // --- Even tile: buffer set A, event 0 ---
            {
                const uint32_t tileValues = static_cast<uint32_t>(
                    valuesCurrentCore - offset > STATE_COUNT ? STATE_COUNT : valuesCurrentCore - offset);
                WaitFlag<HardEvent::MTE2_V>(0);
                const int64_t nextOffset = offset + tileValues;
                if (tileValues == STATE_COUNT) {
                    SplitTileE5<true>(rawLocal, symbolsLocal, mantissaLocal, tileValues);
                } else {
                    SplitTileE5<false>(rawLocal, symbolsLocal, mantissaLocal, tileValues);
                }
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(mantissaGm_[(valueStart + offset) * (DTYPE_BYTES - 1)], mantissaLocal,
                         tileValues * (DTYPE_BYTES - 1));
                DataCopy(varGm_[varOffset + offset - processedValues], symbolsLocal, tileValues);

                if (nextOffset < valuesCurrentCore) {
                    const uint32_t nextTileValues = static_cast<uint32_t>(
                        valuesCurrentCore - nextOffset > STATE_COUNT ? STATE_COUNT : valuesCurrentCore - nextOffset);
                    DataCopy(rawLocalE5b, inputGm_[(valueStart + nextOffset) * DTYPE_BYTES],
                             nextTileValues * DTYPE_BYTES);
                    SetFlag<HardEvent::MTE2_V>(1);
                }
                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                offset = nextOffset;
            }
            if (offset >= valuesCurrentCore) {
                break;
            }

            // --- Odd tile: buffer set B, event 1 ---
            {
                const uint32_t tileValues = static_cast<uint32_t>(
                    valuesCurrentCore - offset > STATE_COUNT ? STATE_COUNT : valuesCurrentCore - offset);
                WaitFlag<HardEvent::MTE2_V>(1);
                const int64_t nextOffset = offset + tileValues;
                if (tileValues == STATE_COUNT) {
                    SplitTileE5<true>(rawLocalE5b, symbolsLocalE5b, mantissaLocalE5b, tileValues);
                } else {
                    SplitTileE5<false>(rawLocalE5b, symbolsLocalE5b, mantissaLocalE5b, tileValues);
                }
                SetFlag<HardEvent::V_MTE3>(1);
                WaitFlag<HardEvent::V_MTE3>(1);
                DataCopy(mantissaGm_[(valueStart + offset) * (DTYPE_BYTES - 1)], mantissaLocalE5b,
                         tileValues * (DTYPE_BYTES - 1));
                DataCopy(varGm_[varOffset + offset - processedValues], symbolsLocalE5b, tileValues);

                if (nextOffset < valuesCurrentCore) {
                    const uint32_t nextTileValues = static_cast<uint32_t>(
                        valuesCurrentCore - nextOffset > STATE_COUNT ? STATE_COUNT : valuesCurrentCore - nextOffset);
                    DataCopy(rawLocal, inputGm_[(valueStart + nextOffset) * DTYPE_BYTES], nextTileValues * DTYPE_BYTES);
                    SetFlag<HardEvent::MTE2_V>(0);
                }
                SetFlag<HardEvent::MTE3_MTE2>(1);
                WaitFlag<HardEvent::MTE3_MTE2>(1);
                offset = nextOffset;
            }
        }
    }

    __aicore__ inline void CompactPayload(LocalTensor<uint8_t> copyLocal, GlobalTensor<uint8_t> payloadGm,
                                          int64_t payloadBytes, int64_t blockIdx, LocalTensor<int32_t> headerLocal)
    {
        int64_t compactOffset = HEADER_BYTES;
        for (int32_t core = 0; core < blockIdx; ++core) {
            compactOffset += headerLocal.GetValue(DEVICE_START_IDX + core);
        }
        HANS_ENC_DBG("Compact: compactOffset=%lld payloadBytes=%lld blockIdx=%d", static_cast<long long>(compactOffset),
                     static_cast<long long>(payloadBytes), static_cast<int>(blockIdx));
        int64_t copied = 0;
        while (copied < payloadBytes) {
            const uint32_t copyBytes = static_cast<uint32_t>(
                payloadBytes - copied > STATE_COUNT * DTYPE_BYTES ? STATE_COUNT * DTYPE_BYTES : payloadBytes - copied);
            DataCopy(copyLocal, payloadGm[copied], copyBytes);
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
            DataCopy(fixedGm_[compactOffset + copied], copyLocal, copyBytes);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            copied += copyBytes;
        }
    }
};

} // namespace HansEncodeArch35

#undef HANS_ENC_DBG

#endif // HANS_ENCODE_SIMT_H
