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
 * \file hans_decode_simt.h
 * \brief Ascend 950 SIMD/SIMT implementation of the HANS decoder.
 */
#ifndef HANS_DECODE_SIMT_H
#define HANS_DECODE_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace HansDecodeArch35 {

using namespace AscendC;
using namespace HansFormat;

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t REGISTER_BYTES = 256;
// cpudebug requires the tensor allocation size to be a multiple of 32 bytes.
// Only slots [0..2] carry data; the remaining slots provide aligned storage.
constexpr uint32_t CONTROL_SLOTS = 8;
constexpr uint32_t OVERFLOW_COUNT_IDX = 0;
constexpr uint32_t WARP_TOTAL_IDX = 1;
constexpr uint32_t VALIDATION_ERROR_IDX = 2;

static_assert(GROUP_COUNT == 2 * WARP_SIZE, "The overflow prefix scan requires exactly two full warps.");
static_assert(CONTROL_SLOTS * sizeof(int32_t) % 32 == 0, "Control storage must be 32-byte aligned for cpudebug.");

// Build the FP32 stride-3 byte-gather indices in int32, then narrow to uint8.
constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_S32_TO_U8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void BuildRankToSymbol(const __ubuf__ int32_t* pdf,
                                                                                __ubuf__ uint8_t* rankToSymbol)
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
    rankToSymbol[rank] = static_cast<uint8_t>(symbol);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void UnpackStateTail(const __ubuf__ uint16_t* stateTail,
                                                                              __ubuf__ uint32_t* states)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum<0>();
    for (uint32_t index = threadIdx; index < STATE_COUNT; index += threadCount) {
        states[index] = static_cast<uint32_t>(stateTail[index]);
    }
}

// Counters are updated by ComputeDecodeOverflow before this per-value decode.
// Threads within each warp use the same group metadata.
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void DecodeTile(
    __ubuf__ uint32_t* states, const __ubuf__ uint16_t* groupBits, const __ubuf__ uint8_t* overflow,
    const __ubuf__ int32_t* overflowPrefix, const __ubuf__ uint16_t* lowWords, const __ubuf__ uint8_t* rankToSymbol,
    __ubuf__ uint8_t* symbols, uint32_t valueCount)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum<0>();
    for (uint32_t index = threadIdx; index < valueCount; index += threadCount) {
        const uint32_t group = index / BLOCK_SIZE;
        const uint32_t lane = index % BLOCK_SIZE;
        uint32_t state = states[index];
        if (overflow[group] != 0) {
            const uint32_t lowIndex = static_cast<uint32_t>(overflowPrefix[group]) * BLOCK_SIZE + lane;
            state = (state << RECORD_BITS) | static_cast<uint32_t>(lowWords[lowIndex]);
        }
        const uint32_t bits = static_cast<uint32_t>(groupBits[group]);
        const uint32_t rankMask = (static_cast<uint32_t>(1) << bits) - 1U;
        const uint32_t rank = state & rankMask;
        symbols[index] = rankToSymbol[rank];
        states[index] = state >> bits;
    }
    // Complete every SIMT symbol write before the following vector interleave.
    Simt::ThreadBarrier();
}

// Scan 64 groups as two complete warps. Phase 1 publishes each warp's
// inclusive prefix and updates its counters; a block barrier precedes phase 2,
// which stitches the warps and converts the prefix to an exclusive offset.
// All launched threads reach the barrier in the wrapper, including inactive
// groups. Inactive groups contribute zero but still participate in shuffles.
//
// control: [0] total published by the last active group, [1] warp-0 total,
//          [2] sticky validation error (atomic updates), [3..7] padding.
__simt_callee__ __aicore__ inline void ComputeDecodeOverflowPhase1(
    __ubuf__ int32_t* counters, const __ubuf__ uint16_t* groupBits, __ubuf__ uint8_t* overflow,
    __ubuf__ int32_t* overflowPrefix, __ubuf__ int32_t* control, uint32_t activeGroups, uint32_t tileIndex)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    if (threadIdx >= GROUP_COUNT) {
        return;
    }
    const uint32_t lane = threadIdx % WARP_SIZE;
    const uint32_t warpId = threadIdx / WARP_SIZE;

    const int32_t bits = static_cast<int32_t>(groupBits[threadIdx]);
    // Validation: active groups must have bits in [1,8]; inactive must be 0.
    const bool invalid = (threadIdx < activeGroups) ? (bits < 1 || bits > 8) : (bits != 0);
    if (invalid) {
        Simt::AtomicOr<int32_t>(control + VALIDATION_ERROR_IDX, 1);
    }

    // Decode overflow semantics: only tiles after the first (tileIndex > 0)
    // can overflow, and a group overflows when its counter is <= its bits.
    const uint32_t myFlag = (tileIndex > 0 && threadIdx < activeGroups && counters[threadIdx] <= bits) ? 1u : 0u;
    overflow[threadIdx] = static_cast<uint8_t>(myFlag);

    // Counter updates are independent of symbol decoding. The first encoded
    // tile must rewind all counters to zero, including inactive groups.
    if (threadIdx < activeGroups) {
        const int32_t nextCounter = counters[threadIdx] + (myFlag != 0 ? RECORD_BITS : 0) - bits;
        counters[threadIdx] = nextCounter;
        if (tileIndex == 0 && nextCounter != 0) {
            Simt::AtomicOr<int32_t>(control + VALIDATION_ERROR_IDX, 1);
        }
    } else if (tileIndex == 0 && counters[threadIdx] != 0) {
        // Preserve the old final validation over all GROUP_COUNT counters.
        Simt::AtomicOr<int32_t>(control + VALIDATION_ERROR_IDX, 1);
    }

    // Underflow lanes receive their own value from asc_shfl_up. Guard each
    // addition so those lanes are not counted twice.
    uint32_t prefix = myFlag;
    uint32_t shifted = asc_shfl_up(prefix, 1U);
    if (lane >= 1U) {
        prefix += shifted;
    }
    shifted = asc_shfl_up(prefix, 2U);
    if (lane >= 2U) {
        prefix += shifted;
    }
    shifted = asc_shfl_up(prefix, 4U);
    if (lane >= 4U) {
        prefix += shifted;
    }
    shifted = asc_shfl_up(prefix, 8U);
    if (lane >= 8U) {
        prefix += shifted;
    }
    shifted = asc_shfl_up(prefix, 16U);
    if (lane >= 16U) {
        prefix += shifted;
    }

    // Temporarily store the per-warp inclusive prefix in overflowPrefix[].
    overflowPrefix[threadIdx] = static_cast<int32_t>(prefix);

    // Publish warp0's inclusive total to control[WARP_TOTAL_IDX] for the cross-warp stitch.
    if (warpId == 0 && lane == WARP_SIZE - 1) {
        control[WARP_TOTAL_IDX] = static_cast<int32_t>(prefix);
    }
}

__simt_callee__ __aicore__ inline void ComputeDecodeOverflowPhase2(const __ubuf__ uint8_t* overflow,
                                                                   __ubuf__ int32_t* overflowPrefix,
                                                                   __ubuf__ int32_t* control, uint32_t activeGroups)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    if (threadIdx >= GROUP_COUNT) {
        return;
    }
    const uint32_t warpId = threadIdx / WARP_SIZE;

    // Cross-warp stitch + convert inclusive prefix to exclusive.
    uint32_t stitched = static_cast<uint32_t>(overflowPrefix[threadIdx]);
    if (warpId == 1) {
        stitched += static_cast<uint32_t>(control[WARP_TOTAL_IDX]);
    }

    // The last active group owns the final inclusive prefix and publishes
    // overflowCount without a second cross-warp reduction.
    if (activeGroups > 0 && threadIdx == activeGroups - 1u) {
        control[OVERFLOW_COUNT_IDX] = static_cast<int32_t>(stitched);
    }

    const uint32_t myFlag = static_cast<uint32_t>(overflow[threadIdx]);
    overflowPrefix[threadIdx] = static_cast<int32_t>(stitched - myFlag);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void ComputeDecodeOverflow(
    __ubuf__ int32_t* counters, const __ubuf__ uint16_t* groupBits, __ubuf__ uint8_t* overflow,
    __ubuf__ int32_t* overflowPrefix, __ubuf__ int32_t* control, uint32_t activeGroups, uint32_t tileIndex)
{
    ComputeDecodeOverflowPhase1(counters, groupBits, overflow, overflowPrefix, control, activeGroups, tileIndex);
    Simt::ThreadBarrier(); // Publish the first warp's total to the second warp.
    ComputeDecodeOverflowPhase2(overflow, overflowPrefix, control, activeGroups);
}

template <int32_t DTYPE_BYTES>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void JoinRawBytes(const __ubuf__ uint8_t* symbols,
                                                                           const __ubuf__ uint8_t* mantissa,
                                                                           __ubuf__ uint8_t* raw, uint32_t valueCount)
{
    const uint32_t threadIdx = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum<0>();
    for (uint32_t index = threadIdx; index < valueCount; index += threadCount) {
        const uint32_t rawOffset = index * DTYPE_BYTES;
        const uint32_t mantissaOffset = index * (DTYPE_BYTES - 1);
        for (uint32_t byte = 0; byte < static_cast<uint32_t>(DTYPE_BYTES - 1); ++byte) {
            raw[rawOffset + byte] = mantissa[mantissaOffset + byte];
        }
        raw[rawOffset + DTYPE_BYTES - 1] = symbols[index];
    }
}

template <int32_t DTYPE_BYTES>
class HansDecodeSimt {
    static_assert(DTYPE_BYTES == 2 || DTYPE_BYTES == 4, "HANS supports only 2-byte and 4-byte values.");
    // All buffer allocation sizes are multiples of 32 bytes.
    static constexpr uint32_t REQUIRED_UB_BYTES = HEADER_BYTES + STATE_COUNT * sizeof(uint32_t) + STATE_TAIL_BYTES +
                                                  COUNTER_TAIL_BYTES + GROUP_BITS_BYTES +
                                                  GROUP_COUNT * sizeof(uint8_t) + GROUP_COUNT * sizeof(int32_t) +
                                                  PDF_LENGTH * sizeof(int32_t) + PDF_LENGTH * sizeof(uint8_t) +
                                                  STATE_COUNT + REGISTER_BYTES + STATE_COUNT * (DTYPE_BYTES - 1) +
                                                  REGISTER_BYTES + STATE_COUNT * DTYPE_BYTES +
                                                  CONTROL_SLOTS * sizeof(int32_t);
    static_assert(REQUIRED_UB_BYTES <= USABLE_UB_BYTES, "Decoder buffers exceed the available UB budget.");

public:
    __aicore__ inline void Process(GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf, GM_ADDR output,
                                   const HansDecodeTilingData* tiling)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= tiling->launchCoreDim || tiling->fixedByteSize < HEADER_BYTES) {
            return;
        }
        InitGlobalTensors(mantissa, fixed, var, pdf, output);

        LocalMemAllocator<Hardware::UB> allocator;
        LocalTensor<int32_t> headerLocal = allocator.Alloc<int32_t>(HEADER_INT32_COUNT);
        DataCopy(headerLocal, headerGm_, HEADER_INT32_COUNT);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);

        const int32_t magic = headerLocal.GetValue(HEADER_MAGIC_IDX);
        const int32_t actualCoreCount = headerLocal.GetValue(HEADER_CORE_COUNT_IDX);
        const int32_t totalLoops = headerLocal.GetValue(HEADER_TOTAL_LOOPS_IDX);
        if (magic != MAGIC || actualCoreCount <= 0 || actualCoreCount > MAX_CORE_COUNT ||
            actualCoreCount > tiling->launchCoreDim || blockIdx >= actualCoreCount || totalLoops <= 0 ||
            static_cast<int64_t>(totalLoops) * BLOCK_SIZE != tiling->outputValueCount) {
            return;
        }

        const int64_t loopsPerCore = totalLoops / actualCoreCount;
        const int64_t loopsCurrentCore = blockIdx < actualCoreCount - 1 ? loopsPerCore :
                                                                          loopsPerCore + totalLoops % actualCoreCount;
        const int64_t valuesCurrentCore = loopsCurrentCore * BLOCK_SIZE;
        const int64_t valueStart = loopsPerCore * BLOCK_SIZE * blockIdx;
        const int64_t hostValues = headerLocal.GetValue(HOST_START_IDX + blockIdx);
        const int64_t devicePayloadBytes = headerLocal.GetValue(DEVICE_START_IDX + blockIdx);
        if (hostValues < 0 || hostValues > valuesCurrentCore || hostValues % BLOCK_SIZE != 0 ||
            devicePayloadBytes < 0) {
            return;
        }
        const int64_t encodedValues = valuesCurrentCore - hostValues;
        if ((encodedValues > 0 && devicePayloadBytes < TAIL_BYTES) ||
            (encodedValues == 0 && devicePayloadBytes != 0 && devicePayloadBytes < TAIL_BYTES)) {
            return;
        }

        int64_t totalHostValues = 0;
        int64_t totalEncodedValues = 0;
        for (int32_t core = 0; core < actualCoreCount; ++core) {
            const int64_t coreHost = headerLocal.GetValue(HOST_START_IDX + core);
            if (coreHost < 0) {
                return;
            }
            totalHostValues += coreHost;
        }
        totalEncodedValues = static_cast<int64_t>(totalLoops) * BLOCK_SIZE - totalHostValues;
        if (totalHostValues > tiling->varByteSize ||
            totalHostValues / BLOCK_SIZE != headerLocal.GetValue(HEADER_VAR_LOOPS_IDX) ||
            totalEncodedValues / BLOCK_SIZE != headerLocal.GetValue(HEADER_FIXED_LOOPS_IDX)) {
            return;
        }

        const int64_t payloadStart = GetPayloadStart(tiling, actualCoreCount, blockIdx, devicePayloadBytes,
                                                     headerLocal);
        if (payloadStart < 0 || payloadStart + devicePayloadBytes > tiling->fixedByteSize) {
            return;
        }

        LocalTensor<uint32_t> statesLocal = allocator.Alloc<uint32_t>(STATE_COUNT);
        LocalTensor<uint16_t> lowWordsLocal = allocator.Alloc<uint16_t>(STATE_COUNT);
        LocalTensor<int32_t> countersLocal = allocator.Alloc<int32_t>(GROUP_COUNT);
        LocalTensor<uint16_t> groupBitsLocal = allocator.Alloc<uint16_t>(GROUP_COUNT);
        LocalTensor<uint8_t> overflowLocal = allocator.Alloc<uint8_t>(GROUP_COUNT);
        LocalTensor<int32_t> overflowPrefixLocal = allocator.Alloc<int32_t>(GROUP_COUNT);
        LocalTensor<int32_t> pdfLocal = allocator.Alloc<int32_t>(PDF_LENGTH);
        LocalTensor<uint8_t> rankToSymbolLocal = allocator.Alloc<uint8_t>(PDF_LENGTH);
        // Full-register loads can read 192 bytes past the final symbol chunk
        // and 64 bytes past the final mantissa chunk. Padding keeps those
        // unused lanes within allocated UB; gather indices select valid bytes.
        LocalTensor<uint8_t> symbolsLocal = allocator.Alloc<uint8_t>(STATE_COUNT + REGISTER_BYTES);
        LocalTensor<uint8_t> mantissaLocal = allocator.Alloc<uint8_t>(STATE_COUNT * (DTYPE_BYTES - 1) + REGISTER_BYTES);
        LocalTensor<uint8_t> rawLocal = allocator.Alloc<uint8_t>(STATE_COUNT * DTYPE_BYTES);
        // Prefix totals, sticky error flag and padding for cpudebug alignment.
        LocalTensor<int32_t> controlLocal = allocator.Alloc<int32_t>(CONTROL_SLOTS);

        DataCopy(pdfLocal, pdfGm_, PDF_LENGTH);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Simt::VF_CALL<BuildRankToSymbol>(Simt::Dim3{THREAD_COUNT}, (__ubuf__ int32_t*)pdfLocal.GetPhyAddr(),
                                         (__ubuf__ uint8_t*)rankToSymbolLocal.GetPhyAddr());

        if (encodedValues > 0) {
            if (!DecodeFixedPayload(statesLocal, lowWordsLocal, countersLocal, groupBitsLocal, overflowLocal,
                                    overflowPrefixLocal, controlLocal, rankToSymbolLocal, symbolsLocal, mantissaLocal,
                                    rawLocal, payloadStart, devicePayloadBytes, valueStart, encodedValues)) {
                return;
            }
        }
        DecodeVarSuffix(symbolsLocal, mantissaLocal, rawLocal, valueStart, encodedValues, hostValues, blockIdx,
                        headerLocal);
    }

private:
    GlobalTensor<uint8_t> mantissaGm_;
    GlobalTensor<uint8_t> fixedGm_;
    GlobalTensor<uint8_t> varGm_;
    GlobalTensor<int32_t> pdfGm_;
    GlobalTensor<uint8_t> outputGm_;
    GlobalTensor<int32_t> headerGm_;

    __aicore__ inline void InitGlobalTensors(GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf, GM_ADDR output)
    {
        mantissaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mantissa));
        fixedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(fixed));
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(var));
        pdfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pdf), PDF_LENGTH);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        headerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(fixed), HEADER_INT32_COUNT);
    }

    __aicore__ inline int64_t GetPayloadStart(const HansDecodeTilingData* tiling, int32_t actualCoreCount,
                                              int64_t blockIdx, int64_t payloadBytes, LocalTensor<int32_t> headerLocal)
    {
        if (tiling->reshuff) {
            int64_t compactOffset = HEADER_BYTES;
            for (int32_t core = 0; core < blockIdx; ++core) {
                const int64_t coreBytes = headerLocal.GetValue(DEVICE_START_IDX + core);
                if (coreBytes < 0) {
                    return -1;
                }
                compactOffset += coreBytes;
            }
            return compactOffset;
        }
        const int64_t fixedAreaBytes = tiling->fixedByteSize - HEADER_BYTES;
        if (fixedAreaBytes < 0) {
            return -1;
        }
        const int64_t slotBytes = fixedAreaBytes / actualCoreCount;
        const int64_t slotCapacity = blockIdx < actualCoreCount - 1 ? slotBytes :
                                                                      slotBytes + fixedAreaBytes % actualCoreCount;
        if (payloadBytes > slotCapacity) {
            return -1;
        }
        return HEADER_BYTES + slotBytes * blockIdx;
    }

    __aicore__ inline bool DecodeFixedPayload(LocalTensor<uint32_t> statesLocal, LocalTensor<uint16_t> lowWordsLocal,
                                              LocalTensor<int32_t> countersLocal, LocalTensor<uint16_t> groupBitsLocal,
                                              LocalTensor<uint8_t> overflowLocal,
                                              LocalTensor<int32_t> overflowPrefixLocal,
                                              LocalTensor<int32_t> controlLocal, LocalTensor<uint8_t> rankToSymbolLocal,
                                              LocalTensor<uint8_t> symbolsLocal, LocalTensor<uint8_t> mantissaLocal,
                                              LocalTensor<uint8_t> rawLocal, int64_t payloadStart, int64_t payloadBytes,
                                              int64_t valueStart, int64_t encodedValues)
    {
        const int64_t recordEnd = payloadStart + payloadBytes - TAIL_BYTES;
        DataCopy(lowWordsLocal.ReinterpretCast<uint8_t>(), fixedGm_[recordEnd], STATE_TAIL_BYTES);
        DataCopy(countersLocal.ReinterpretCast<uint8_t>(), fixedGm_[recordEnd + STATE_TAIL_BYTES], COUNTER_TAIL_BYTES);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Simt::VF_CALL<UnpackStateTail>(Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint16_t*)lowWordsLocal.GetPhyAddr(),
                                       (__ubuf__ uint32_t*)statesLocal.GetPhyAddr());

        // The sticky error flag is initialized once and any error ends decoding.
        controlLocal.SetValue(VALIDATION_ERROR_IDX, 0);
        // Publish the initialized flag to the SIMT threads.
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);

        int64_t cursor = recordEnd;
        const int64_t tileCount = (encodedValues + STATE_COUNT - 1) / STATE_COUNT;
        for (int64_t tileIndex = tileCount - 1; tileIndex >= 0; --tileIndex) {
            const int64_t tileStart = tileIndex * STATE_COUNT;
            const uint32_t tileValues = static_cast<uint32_t>(
                encodedValues - tileStart > STATE_COUNT ? STATE_COUNT : encodedValues - tileStart);
            const uint32_t activeGroups = tileValues / BLOCK_SIZE;
            const int64_t bitsStart = cursor - GROUP_BITS_BYTES;
            if (bitsStart < payloadStart) {
                return false;
            }
            DataCopy(groupBitsLocal.ReinterpretCast<uint8_t>(), fixedGm_[bitsStart], GROUP_BITS_BYTES);
            // The overflow VF consumes group metadata on the Vector pipe.
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Simt::VF_CALL<ComputeDecodeOverflow>(
                Simt::Dim3{GROUP_COUNT}, (__ubuf__ int32_t*)countersLocal.GetPhyAddr(),
                (__ubuf__ uint16_t*)groupBitsLocal.GetPhyAddr(), (__ubuf__ uint8_t*)overflowLocal.GetPhyAddr(),
                (__ubuf__ int32_t*)overflowPrefixLocal.GetPhyAddr(), (__ubuf__ int32_t*)controlLocal.GetPhyAddr(),
                activeGroups, static_cast<uint32_t>(tileIndex));
            // Scalar reads the prefix total and validation flag after the VF.
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            if (controlLocal.GetValue(VALIDATION_ERROR_IDX) != 0) {
                return false; // validation failed
            }
            const int32_t overflowCount = controlLocal.GetValue(OVERFLOW_COUNT_IDX);
            const int64_t lowBytes = static_cast<int64_t>(overflowCount) * GROUP_OVERFLOW_BYTES;
            const int64_t lowStart = bitsStart - lowBytes;
            if (lowStart < payloadStart) {
                return false;
            }
            if (lowBytes > 0) {
                DataCopy(lowWordsLocal.ReinterpretCast<uint8_t>(), fixedGm_[lowStart], lowBytes);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
            }

            // Prefetch mantissa while DecodeTile updates states and symbols.
            // JoinAndCopyOut consumes this event immediately before interleaving.
            DataCopy(mantissaLocal, mantissaGm_[(valueStart + tileStart) * (DTYPE_BYTES - 1)],
                     tileValues * (DTYPE_BYTES - 1));
            SetFlag<HardEvent::MTE2_V>(0);

            Simt::VF_CALL<DecodeTile>(
                Simt::Dim3{THREAD_COUNT}, (__ubuf__ uint32_t*)statesLocal.GetPhyAddr(),
                (__ubuf__ uint16_t*)groupBitsLocal.GetPhyAddr(), (__ubuf__ uint8_t*)overflowLocal.GetPhyAddr(),
                (__ubuf__ int32_t*)overflowPrefixLocal.GetPhyAddr(), (__ubuf__ uint16_t*)lowWordsLocal.GetPhyAddr(),
                (__ubuf__ uint8_t*)rankToSymbolLocal.GetPhyAddr(), (__ubuf__ uint8_t*)symbolsLocal.GetPhyAddr(),
                tileValues);
            JoinAndCopyOut(mantissaLocal, rawLocal, symbolsLocal, valueStart + tileStart, tileValues);
            cursor = lowStart;
        }
        if (cursor != payloadStart) {
            return false;
        }
        // The tileIndex == 0 overflow phase validates the final counters.
        return true;
    }

    // Both fixed and var paths provide a pending MTE2_V event for mantissa.
    // Keep the register operations and V_S synchronization together.
    __aicore__ inline void JoinAndCopyOut(LocalTensor<uint8_t> mantissaLocal, LocalTensor<uint8_t> rawLocal,
                                          LocalTensor<uint8_t> symbolsLocal, int64_t globalValueOffset,
                                          uint32_t valueCount)
    {
        // Wait for the pending mantissa transfer.
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (DTYPE_BYTES == 2) {
            // Interleave [mantissa, symbol] for each 2-byte value. A partial
            // register is handled by the SIMT tail below.
            constexpr uint32_t VEC_BYTES = REGISTER_BYTES;
            const uint16_t fullChunks = static_cast<uint16_t>(valueCount / VEC_BYTES);
            const uint32_t tail = valueCount % VEC_BYTES;
            __ubuf__ uint8_t* mantPtr = (__ubuf__ uint8_t*)mantissaLocal.GetPhyAddr();
            __ubuf__ uint8_t* symPtr = (__ubuf__ uint8_t*)symbolsLocal.GetPhyAddr();
            __ubuf__ uint8_t* rawPtr = (__ubuf__ uint8_t*)rawLocal.GetPhyAddr();
            if (fullChunks > 0) {
                __VEC_SCOPE__
                {
                    using namespace AscendC::MicroAPI;
                    MaskReg maskAll = CreateMask<uint8_t, MaskPattern::ALL>();
                    for (uint16_t c = 0; c < fullChunks; ++c) {
                        RegTensor<uint8_t> mantReg, symReg, dstReg0, dstReg1;
                        DataCopy<uint8_t>(mantReg, mantPtr + c * VEC_BYTES);
                        DataCopy<uint8_t>(symReg, symPtr + c * VEC_BYTES);
                        Interleave<uint8_t>(dstReg0, dstReg1, mantReg, symReg);
                        DataCopy<uint8_t>(rawPtr + c * 2u * VEC_BYTES, dstReg0, maskAll);
                        DataCopy<uint8_t>(rawPtr + c * 2u * VEC_BYTES + VEC_BYTES, dstReg1, maskAll);
                    }
                }
                // This immediately paired event is shared by NPU and cpudebug.
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
            }
            if (tail > 0) {
                Simt::VF_CALL<JoinRawBytes<DTYPE_BYTES>>(Simt::Dim3{THREAD_COUNT}, symPtr + fullChunks * VEC_BYTES,
                                                         mantPtr + fullChunks * VEC_BYTES,
                                                         rawPtr + fullChunks * 2u * VEC_BYTES, tail);
            }
        } else if constexpr (DTYPE_BYTES == 4) {
            // Gather the three stride-3 mantissa streams, then interleave them
            // with the symbol stream into [m0, m1, m2, symbol] for each value.
            constexpr uint32_t VEC_BYTES = REGISTER_BYTES;
            constexpr uint32_t CHUNK_VALUES = 64u;
            const uint16_t fullChunks = static_cast<uint16_t>(valueCount / CHUNK_VALUES);
            const uint32_t tail = valueCount % CHUNK_VALUES;
            __ubuf__ uint8_t* mantPtr = (__ubuf__ uint8_t*)mantissaLocal.GetPhyAddr();
            __ubuf__ uint8_t* symPtr = (__ubuf__ uint8_t*)symbolsLocal.GetPhyAddr();
            __ubuf__ uint8_t* rawPtr = (__ubuf__ uint8_t*)rawLocal.GetPhyAddr();
            if (fullChunks > 0) {
                __VEC_SCOPE__
                {
                    using namespace AscendC::MicroAPI;
                    MaskReg maskAll = CreateMask<uint8_t, MaskPattern::ALL>();
                    RegTensor<int32_t> idx32;
                    Arange<int32_t>(idx32, 0);
                    Muls<int32_t, int32_t, MaskMergeMode::ZEROING>(idx32, idx32, 3, maskAll);
                    RegTensor<int32_t> idx32B, idx32C;
                    Adds<int32_t, int32_t, MaskMergeMode::ZEROING>(idx32B, idx32, 1, maskAll);
                    Adds<int32_t, int32_t, MaskMergeMode::ZEROING>(idx32C, idx32, 2, maskAll);
                    RegTensor<uint8_t> idxA, idxB, idxC;
                    Cast<uint8_t, int32_t, CAST_TRAIT_S32_TO_U8>(idxA, idx32, maskAll);
                    Cast<uint8_t, int32_t, CAST_TRAIT_S32_TO_U8>(idxB, idx32B, maskAll);
                    Cast<uint8_t, int32_t, CAST_TRAIT_S32_TO_U8>(idxC, idx32C, maskAll);
                    // Narrowing writes at byte offsets 0, 4, 8, ... . Two
                    // deinterleaves pack the 64 gather indices into low lanes.
                    RegTensor<uint8_t> tmpA0, tmpA1, tmpB0, tmpB1, tmpC0, tmpC1;
                    DeInterleave<uint8_t>(tmpA0, tmpA1, idxA, idxA);
                    DeInterleave<uint8_t>(idxA, tmpA1, tmpA0, tmpA0);
                    DeInterleave<uint8_t>(tmpB0, tmpB1, idxB, idxB);
                    DeInterleave<uint8_t>(idxB, tmpB1, tmpB0, tmpB0);
                    DeInterleave<uint8_t>(tmpC0, tmpC1, idxC, idxC);
                    DeInterleave<uint8_t>(idxC, tmpC1, tmpC0, tmpC0);
                    for (uint16_t c = 0; c < fullChunks; ++c) {
                        RegTensor<uint8_t> mantReg, symReg;
                        DataCopy<uint8_t>(mantReg, mantPtr + c * (CHUNK_VALUES * 3u));
                        DataCopy<uint8_t>(symReg, symPtr + c * CHUNK_VALUES);
                        RegTensor<uint8_t> A, B, C;
                        Gather<uint8_t, uint8_t>(A, mantReg, idxA);
                        Gather<uint8_t, uint8_t>(B, mantReg, idxB);
                        Gather<uint8_t, uint8_t>(C, mantReg, idxC);
                        RegTensor<uint8_t> ac0, ac1, bd0, bd1, out0, out1;
                        Interleave<uint8_t>(ac0, ac1, A, C);
                        Interleave<uint8_t>(bd0, bd1, B, symReg);
                        Interleave<uint8_t>(out0, out1, ac0, bd0);
                        DataCopy<uint8_t>(rawPtr + c * VEC_BYTES, out0, maskAll);
                    }
                }
                // This immediately paired event is shared by NPU and cpudebug.
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
            }
            if (tail > 0) {
                Simt::VF_CALL<JoinRawBytes<DTYPE_BYTES>>(Simt::Dim3{THREAD_COUNT}, symPtr + fullChunks * CHUNK_VALUES,
                                                         mantPtr + fullChunks * CHUNK_VALUES * 3u,
                                                         rawPtr + fullChunks * VEC_BYTES, tail);
            }
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(outputGm_[globalValueOffset * DTYPE_BYTES], rawLocal, valueCount * DTYPE_BYTES);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }

    __aicore__ inline void DecodeVarSuffix(LocalTensor<uint8_t> symbolsLocal, LocalTensor<uint8_t> mantissaLocal,
                                           LocalTensor<uint8_t> rawLocal, int64_t valueStart, int64_t encodedValues,
                                           int64_t hostValues, int64_t blockIdx, LocalTensor<int32_t> headerLocal)
    {
        int64_t varStart = 0;
        for (int32_t core = 0; core < blockIdx; ++core) {
            varStart += headerLocal.GetValue(HOST_START_IDX + core);
        }
        for (int64_t offset = 0; offset < hostValues; offset += STATE_COUNT) {
            const uint32_t tileValues = static_cast<uint32_t>(hostValues - offset > STATE_COUNT ? STATE_COUNT :
                                                                                                  hostValues - offset);
            DataCopy(symbolsLocal, varGm_[varStart + offset], tileValues);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            DataCopy(mantissaLocal, mantissaGm_[(valueStart + encodedValues + offset) * (DTYPE_BYTES - 1)],
                     tileValues * (DTYPE_BYTES - 1));
            SetFlag<HardEvent::MTE2_V>(0);
            JoinAndCopyOut(mantissaLocal, rawLocal, symbolsLocal, valueStart + encodedValues + offset, tileValues);
        }
    }
};

} // namespace HansDecodeArch35

#endif // HANS_DECODE_SIMT_H
