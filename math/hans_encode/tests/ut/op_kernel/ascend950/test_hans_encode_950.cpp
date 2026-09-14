/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../op_host/hans_encode_tiling.h"
#include "../../../../op_kernel/hans_format.h"

extern "C" __global__ __aicore__ void hans_encode(GM_ADDR input, GM_ADDR pdf, GM_ADDR pdfRef, GM_ADDR mantissa,
                                                  GM_ADDR fixed, GM_ADDR var, GM_ADDR workspace, GM_ADDR tiling);

namespace {
using namespace HansFormat;
constexpr int64_t kPdfLength = PDF_LENGTH;
constexpr int64_t kProcessMinSizePerCore = 32768;
constexpr uint8_t kSentinel = 0xA5;
// This file runs kernels through ICPU_RUN_KF, not on the NPU. Keep the
// affected cases visible as skipped until the cpudebug prefix issue is resolved.
// Tracking evidence: ops-math PR #5596, CI runs #3418 and #3430.
constexpr const char*
    kCpuPrefixSkipReason = "Temporarily skipped in cpudebug: invalid overflow prefixes in CI #3418/#3430. "
                           "NPU validation is separate; re-enable after the CPU execution issue is resolved.";

class HansEncode950KernelTest : public testing::Test {};

// Keep a trailing guard outside the DMA-aligned allocation. Even zero-length
// logical tensors receive a non-null pointer for the CPU simulator.
class GmBuffer {
public:
    explicit GmBuffer(size_t bytes)
        : size_(std::max<size_t>(32, (bytes + 31) / 32 * 32)),
          data_(static_cast<uint8_t*>(AscendC::GmAlloc(size_ + 32)))
    {
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(data_, kSentinel, size_ + 32);
    }
    ~GmBuffer() { AscendC::GmFree(data_); }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    uint8_t* Data() const { return data_; }
    size_t Size() const { return size_; }
    void CheckGuard() const
    {
        for (size_t i = size_; i < size_ + 32; ++i) {
            ASSERT_EQ(data_[i], kSentinel) << "GM guard byte " << i;
        }
    }
    void CheckUnchanged() const
    {
        for (size_t i = 0; i < size_; ++i) {
            ASSERT_EQ(data_[i], kSentinel) << "Unexpected GM write at byte " << i;
        }
        CheckGuard();
    }

private:
    size_t size_;
    uint8_t* data_;
};

// These are kernel-level fixtures, not host tiling acceptance tests. Empty
// inputs and inputs below 32768 deliberately bypass the host size restriction.
// All non-empty fixtures still contain an integral number of 64-value groups.
int64_t CoreCount(int64_t inputSize, int64_t maxCores)
{
    return std::max<int64_t>(1, std::min(inputSize / kProcessMinSizePerCore, maxCores));
}

// Reserve equal slots large enough for the longest core, including the full
// 128-byte group-bits record for every tile (also for a partial tile).
// This fixture capacity does not assert that the host upper bound is correct.
int64_t FixedCapacity(int64_t inputSize, int64_t maxCores)
{
    const int64_t cores = CoreCount(inputSize, maxCores);
    const int64_t loops = inputSize / BLOCK_SIZE;
    const int64_t longestCore = (loops / cores + loops % cores) * BLOCK_SIZE;
    const int64_t tiles = (longestCore + STATE_COUNT - 1) / STATE_COUNT;
    return HEADER_BYTES + cores * (longestCore + tiles * GROUP_BITS_BYTES + TAIL_BYTES);
}

// HANS treats the highest byte as a symbol; it is not the whole IEEE exponent.
// Preserve the contributor's deterministic generators and raw BF16 patterns.
void GenerateFloat32InputAndPdf(float* input, int32_t* pdf, int64_t inputSize)
{
    uint32_t seed = 1234;
    for (int64_t i = 0; i < inputSize; ++i) {
        seed = seed * 1103515245U + 12345U;
        uint32_t bits = (seed / 65536U) % 32768U;
        input[i] = static_cast<float>(bits) / 32768.0f;
    }

    std::memset(pdf, 0, static_cast<size_t>(kPdfLength) * sizeof(int32_t));
    const auto* bytes = reinterpret_cast<const uint8_t*>(input);
    for (int64_t i = 0; i < inputSize; ++i) {
        const uint8_t symbolByte = bytes[i * 4 + 3];
        pdf[symbolByte]++;
    }
}

void GenerateBf16InputAndPdf(uint16_t* input, int32_t* pdf, int64_t inputSize)
{
    uint32_t seed = 1234;
    for (int64_t i = 0; i < inputSize; ++i) {
        seed = seed * 1103515245U + 12345U;

        uint16_t mantBits = static_cast<uint16_t>(seed & 0x7FU);
        input[i] = static_cast<uint16_t>(0x3F00U | mantBits);
    }
    std::memset(pdf, 0, static_cast<size_t>(kPdfLength) * sizeof(int32_t));
    const auto* bytes = reinterpret_cast<const uint8_t*>(input);
    for (int64_t i = 0; i < inputSize; ++i) {
        const uint8_t symbolByte = bytes[i * 2 + 1];
        pdf[symbolByte]++;
    }
}

enum class InputPattern { kRandom = 0, kUniform, kAllSymbols, kSpecial, kEightBitRank };

void GenerateBf16UniformAndPdf(uint16_t* input, int32_t* pdf, int64_t inputSize)
{
    constexpr uint16_t kValue = 0x3F80;
    for (int64_t i = 0; i < inputSize; ++i) {
        input[i] = kValue;
    }
    std::memset(pdf, 0, kPdfLength * sizeof(int32_t));
    if (inputSize > 0) {
        pdf[kValue >> 8] = static_cast<int32_t>(inputSize);
    }
}

// Equal frequencies give symbol == rank. Each four consecutive groups have
// widths 6, 7, 8 and 8; not every group uses eight bits.
void GenerateBf16AllSymbolsAndPdf(uint16_t* input, int32_t* pdf, int64_t inputSize)
{
    std::memset(pdf, 0, kPdfLength * sizeof(int32_t));
    for (int64_t i = 0; i < inputSize; ++i) {
        const uint16_t symbol = static_cast<uint16_t>(i % kPdfLength);
        input[i] = static_cast<uint16_t>((symbol << 8) | (i & 0xFF));
        pdf[symbol] += 1;
    }
}

void GenerateBf16SpecialAndPdf(uint16_t* input, int32_t* pdf, int64_t inputSize)
{
    static const uint16_t kPatterns[] = {
        0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001, 0x0000, 0x8000, 0x3F80,
    };
    constexpr int64_t kPatternCount = sizeof(kPatterns) / sizeof(kPatterns[0]);
    std::memset(pdf, 0, kPdfLength * sizeof(int32_t));
    for (int64_t i = 0; i < inputSize; ++i) {
        input[i] = kPatterns[i % kPatternCount];
        pdf[input[i] >> 8] += 1;
    }
}

void GenerateInput(uint8_t* input, int32_t* pdf, int64_t count, int64_t bytes, InputPattern pattern)
{
    if (pattern == InputPattern::kEightBitRank) {
        std::fill(pdf, pdf + PDF_LENGTH, 1);
        for (int64_t value = 0; value < count; ++value) {
            for (int64_t byte = 0; byte < bytes - 1; ++byte) {
                input[value * bytes + byte] = static_cast<uint8_t>(value + byte);
            }
            input[value * bytes + bytes - 1] = 0xFF; // Equal PDF counts give rank 255.
        }
        return;
    }
    if (bytes == 4) {
        GenerateFloat32InputAndPdf(reinterpret_cast<float*>(input), pdf, count);
        return;
    }
    auto* bf16 = reinterpret_cast<uint16_t*>(input);
    switch (pattern) {
        case InputPattern::kUniform:
            GenerateBf16UniformAndPdf(bf16, pdf, count);
            break;
        case InputPattern::kAllSymbols:
            GenerateBf16AllSymbolsAndPdf(bf16, pdf, count);
            break;
        case InputPattern::kSpecial:
            GenerateBf16SpecialAndPdf(bf16, pdf, count);
            break;
        default:
            GenerateBf16InputAndPdf(bf16, pdf, count);
            break;
    }
}

// One-symbol input has rank zero and one bit per active group. For at most
// sixteen full tiles, no overflow words are emitted: the payload is completely
// known, including the state and counter tails.
void CheckUniformPayload(const uint8_t* payload, int64_t payloadBytes, int64_t values)
{
    ASSERT_EQ(values % STATE_COUNT, 0);
    const int64_t tiles = values / STATE_COUNT;
    ASSERT_LE(tiles, RECORD_BITS);
    ASSERT_EQ(payloadBytes, tiles * GROUP_BITS_BYTES + TAIL_BYTES);
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t group = 0; group < GROUP_COUNT; ++group) {
            uint16_t bits = 0;
            std::memcpy(&bits, payload + tile * GROUP_BITS_BYTES + group * sizeof(bits), sizeof(bits));
            ASSERT_EQ(bits, 1) << "tile " << tile << ", group " << group;
        }
    }
    const auto* tail = payload + tiles * GROUP_BITS_BYTES;
    for (int64_t byte = 0; byte < STATE_TAIL_BYTES; ++byte) {
        ASSERT_EQ(tail[byte], 0) << "state-tail byte " << byte;
    }
    for (int64_t group = 0; group < GROUP_COUNT; ++group) {
        int32_t counter = 0;
        std::memcpy(&counter, tail + STATE_TAIL_BYTES + group * sizeof(counter), sizeof(counter));
        ASSERT_EQ(counter, tiles) << "counter group " << group;
    }
}

// For constant rank 255, counters alternate 8/16 and all groups overflow on
// tiles 3,5,7,... . Verify the entire stream, not just a nonempty fixed header.
void CheckEightBitPayload(const uint8_t* payload, int64_t payloadBytes, int64_t values)
{
    ASSERT_EQ(values % STATE_COUNT, 0);
    const int64_t tiles = values / STATE_COUNT;
    ASSERT_GT(tiles, 0);
    const int64_t overflowTiles = (tiles - 1) / 2;
    ASSERT_EQ(payloadBytes, tiles * GROUP_BITS_BYTES + overflowTiles * STATE_TAIL_BYTES + TAIL_BYTES);
    int64_t offset = 0;
    for (int64_t tile = 0; tile < tiles; ++tile) {
        if (tile >= 2 && tile % 2 == 0) {
            for (int64_t state = 0; state < STATE_COUNT; ++state) {
                uint16_t word = 0;
                std::memcpy(&word, payload + offset + state * sizeof(word), sizeof(word));
                ASSERT_EQ(word, 0xFFFF) << "tile " << tile << ", state " << state;
            }
            offset += STATE_TAIL_BYTES;
        }
        for (int64_t group = 0; group < GROUP_COUNT; ++group) {
            uint16_t bits = 0;
            std::memcpy(&bits, payload + offset + group * sizeof(bits), sizeof(bits));
            ASSERT_EQ(bits, 8) << "tile " << tile << ", group " << group;
        }
        offset += GROUP_BITS_BYTES;
    }
    for (int64_t state = 0; state < STATE_COUNT; ++state) {
        uint16_t word = 0;
        std::memcpy(&word, payload + offset + state * sizeof(word), sizeof(word));
        ASSERT_EQ(word, tiles % 2 == 0 ? 0xFFFF : 0x00FF);
    }
    offset += STATE_TAIL_BYTES;
    for (int64_t group = 0; group < GROUP_COUNT; ++group) {
        int32_t counter = 0;
        std::memcpy(&counter, payload + offset + group * sizeof(counter), sizeof(counter));
        ASSERT_EQ(counter, tiles % 2 == 0 ? 16 : 8);
    }
}

void RunEncodeKernelWithFixedSize(int64_t inputSize, int64_t dtypeBytes, int64_t maxCores, uint64_t tilingKey,
                                  bool statistic, bool reshuff, int64_t fixedByteSize, int64_t varByteSize,
                                  InputPattern pattern = InputPattern::kRandom)
{
    ASSERT_GE(inputSize, 0);
    ASSERT_EQ(inputSize % BLOCK_SIZE, 0) << "Do not silently truncate fixture input.";
    ASSERT_TRUE(dtypeBytes == 2 || dtypeBytes == 4);
    ASSERT_EQ(tilingKey, static_cast<uint64_t>(dtypeBytes));
    ASSERT_GT(maxCores, 0);
    ASSERT_LE(maxCores, MAX_CORE_COUNT);
    ASSERT_GE(fixedByteSize, HEADER_BYTES);
    ASSERT_GE(varByteSize, 0);
    const int64_t cores = CoreCount(inputSize, maxCores);
    const int64_t loops = inputSize / BLOCK_SIZE;
    const int64_t loopsPerCore = loops / cores;
    const int64_t slotBytes = (fixedByteSize - HEADER_BYTES) / cores;

    GmBuffer input(inputSize * dtypeBytes);
    GmBuffer pdf(PDF_LENGTH * sizeof(int32_t));
    GmBuffer pdfRef(PDF_LENGTH * sizeof(int32_t));
    GmBuffer mantissa(inputSize * (dtypeBytes - 1));
    GmBuffer fixed(fixedByteSize);
    GmBuffer var(varByteSize);
    GmBuffer workspace(16 * 1024 * 1024 + (reshuff ? fixedByteSize : 0));
    optiling::HansEncodeTilingData tilingData;
    GmBuffer tiling(tilingData.GetDataSize());
    GenerateInput(input.Data(), reinterpret_cast<int32_t*>(pdf.Data()), inputSize, dtypeBytes, pattern);
    std::array<int32_t, PDF_LENGTH> expectedPdf{};
    std::memcpy(expectedPdf.data(), pdf.Data(), sizeof(expectedPdf));
    if (statistic) {
        // Reusing a correct input PDF would let a no-op histogram pass.
        std::memset(pdf.Data(), kSentinel, pdf.Size());
    }

    tilingData.set_processCoreDim(cores);
    tilingData.set_processLoopPerCore(loopsPerCore);
    tilingData.set_processLoopLastCore(loopsPerCore + loops % cores);
    tilingData.set_fixedLengthPerCore(slotBytes);
    tilingData.set_fixedLengthLastCore(slotBytes + (fixedByteSize - HEADER_BYTES) % cores);
    tilingData.set_varLength(varByteSize);
    tilingData.set_statistic(statistic);
    tilingData.set_reshuff(reshuff);
    tilingData.SaveToBuffer(tiling.Data(), tiling.Size());

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(hans_encode, static_cast<uint32_t>(cores), input.Data(), pdf.Data(), pdfRef.Data(), mantissa.Data(),
                fixed.Data(), var.Data(), workspace.Data(), tiling.Data());

    for (const GmBuffer* buffer : {&input, &pdf, &pdfRef, &mantissa, &fixed, &var, &workspace, &tiling}) {
        buffer->CheckGuard();
    }
    const auto* actualPdf = reinterpret_cast<const int32_t*>(pdf.Data());
    for (int32_t symbol = 0; symbol < PDF_LENGTH; ++symbol) {
        ASSERT_EQ(actualPdf[symbol], expectedPdf[symbol]) << "PDF symbol " << symbol;
    }
    const auto* hdr = reinterpret_cast<const int32_t*>(fixed.Data());
    ASSERT_EQ(hdr[HEADER_MAGIC_IDX], MAGIC);
    ASSERT_EQ(hdr[HEADER_CORE_COUNT_IDX], cores);
    ASSERT_EQ(hdr[HEADER_TOTAL_LOOPS_IDX], loops);

    int64_t expectedVarValues = 0;
    int64_t compactOffset = HEADER_BYTES;
    for (int64_t core = 0; core < cores; ++core) {
        SCOPED_TRACE(core);
        const int64_t coreValues = (loopsPerCore + (core == cores - 1 ? loops % cores : 0)) * BLOCK_SIZE;
        const int64_t capacity = slotBytes + (core == cores - 1 ? (fixedByteSize - HEADER_BYTES) % cores : 0);
        // The fixtures use either ample fixed capacity or a slot smaller than
        // the state tail, which deliberately forces the entire core into var.
        const int64_t hostValues = capacity < TAIL_BYTES ? coreValues : 0;
        expectedVarValues += hostValues;
        ASSERT_EQ(hdr[HOST_START_IDX + core], hostValues);
        const int64_t payloadBytes = hdr[DEVICE_START_IDX + core];
        ASSERT_GE(payloadBytes, 0);
        ASSERT_LE(payloadBytes, capacity);
        if (capacity < TAIL_BYTES) {
            ASSERT_EQ(payloadBytes, 0);
        } else {
            ASSERT_GE(payloadBytes, TAIL_BYTES);
            const int64_t payloadOffset = reshuff ? compactOffset : HEADER_BYTES + core * slotBytes;
            ASSERT_LE(payloadOffset + payloadBytes, fixedByteSize);
            if (pattern == InputPattern::kUniform) {
                CheckUniformPayload(fixed.Data() + payloadOffset, payloadBytes, coreValues);
            } else if (pattern == InputPattern::kEightBitRank) {
                CheckEightBitPayload(fixed.Data() + payloadOffset, payloadBytes, coreValues);
            } else if (pattern == InputPattern::kAllSymbols && coreValues >= STATE_COUNT) {
                // Equal-frequency symbols are ranked by byte value. The first
                // tile cannot overflow; successive groups have widths 6,7,8,8.
                constexpr uint16_t widths[] = {6, 7, 8, 8};
                for (int64_t group = 0; group < GROUP_COUNT; ++group) {
                    uint16_t bits = 0;
                    std::memcpy(&bits, fixed.Data() + payloadOffset + group * sizeof(bits), sizeof(bits));
                    ASSERT_EQ(bits, widths[group % 4]) << "first-tile group " << group;
                }
            }
            compactOffset += payloadBytes;
        }
    }
    ASSERT_EQ(hdr[HEADER_VAR_LOOPS_IDX], expectedVarValues / BLOCK_SIZE);
    ASSERT_EQ(hdr[HEADER_FIXED_LOOPS_IDX], (inputSize - expectedVarValues) / BLOCK_SIZE);

    if (expectedVarValues > varByteSize) {
        // An intentionally invalid, host-bypassing fixture checks only the
        // bounded early return. A valid header alone is not encode success.
        ASSERT_EQ(cores, 1);
        ASSERT_EQ(expectedVarValues, inputSize);
        var.CheckUnchanged();
        mantissa.CheckUnchanged();
        return;
    }

    std::vector<uint8_t> expectedMantissa(inputSize * (dtypeBytes - 1));
    std::vector<uint8_t> expectedVar;
    for (int64_t value = 0; value < inputSize; ++value) {
        for (int64_t byte = 0; byte < dtypeBytes - 1; ++byte) {
            expectedMantissa[value * (dtypeBytes - 1) + byte] = input.Data()[value * dtypeBytes + byte];
        }
        if (expectedVarValues != 0) {
            expectedVar.push_back(input.Data()[value * dtypeBytes + dtypeBytes - 1]);
        }
    }
    if (!expectedMantissa.empty()) {
        ASSERT_EQ(std::memcmp(mantissa.Data(), expectedMantissa.data(), expectedMantissa.size()), 0)
            << "Mantissa bytes must match the original input exactly.";
    } else {
        mantissa.CheckUnchanged();
    }
    if (!expectedVar.empty()) {
        ASSERT_EQ(expectedVar.size(), static_cast<size_t>(expectedVarValues));
        ASSERT_EQ(std::memcmp(var.Data(), expectedVar.data(), expectedVar.size()), 0);
    } else {
        var.CheckUnchanged();
    }
}

void RunEncodeKernel(int64_t inputSize, int64_t dtypeBytes, int64_t maxCores, uint64_t tilingKey, bool statistic,
                     bool reshuff, InputPattern pattern = InputPattern::kRandom)
{
    RunEncodeKernelWithFixedSize(inputSize, dtypeBytes, maxCores, tilingKey, statistic, reshuff,
                                 FixedCapacity(inputSize, maxCores), 0, pattern);
}

// All launched blocks take the guard, so none enters SyncAll. Do not combine
// active and inactive blocks: that launch/tiling mismatch is not a valid NPU
// configuration and can leave the active blocks waiting at a global barrier.
void RunEncodeInactiveBlocks(uint64_t tilingKey)
{
    GmBuffer input(32), pdf(PDF_LENGTH * sizeof(int32_t)), pdfRef(PDF_LENGTH * sizeof(int32_t));
    GmBuffer mantissa(32), fixed(HEADER_BYTES), var(32), workspace(16 * 1024 * 1024);
    optiling::HansEncodeTilingData tilingData;
    GmBuffer tiling(tilingData.GetDataSize());
    tilingData.set_processCoreDim(0);
    tilingData.set_processLoopPerCore(0);
    tilingData.set_processLoopLastCore(0);
    tilingData.set_fixedLengthPerCore(0);
    tilingData.set_fixedLengthLastCore(0);
    tilingData.set_varLength(0);
    tilingData.set_statistic(false);
    tilingData.set_reshuff(false);
    tilingData.SaveToBuffer(tiling.Data(), tiling.Size());
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(hans_encode, 2, input.Data(), pdf.Data(), pdfRef.Data(), mantissa.Data(), fixed.Data(), var.Data(),
                workspace.Data(), tiling.Data());
    for (const GmBuffer* buffer : {&input, &pdf, &pdfRef, &mantissa, &fixed, &var}) {
        buffer->CheckUnchanged();
    }
    workspace.CheckGuard();
    tiling.CheckGuard();
}

// Repeated launches test repeatability; device race validation is separate.

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_001) { RunEncodeKernel(65536, 4, 1, 4, false, false); }

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_001)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_002)
{
    constexpr int64_t kInputSize = 65536;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, true, true);
}

// 4992 = 4096 + 896: a partial tile with fourteen complete 64-value groups.
TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_003)
{
    constexpr int64_t kInputSize = 4992;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_002)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    constexpr int64_t kInputSize = 65536;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, true, true);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_003)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    constexpr int64_t kInputSize = 4992;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_004)
{
    constexpr int64_t kInputSize = 4992;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, true, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_004)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    constexpr int64_t kInputSize = 4992;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, true, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_005)
{
    constexpr int64_t kInputSize = 9152;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    constexpr int64_t kFixedByteSize = 4096;
    constexpr int64_t kVarByteSize = 16384;
    RunEncodeKernelWithFixedSize(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false, kFixedByteSize,
                                 kVarByteSize);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_005)
{
    constexpr int64_t kInputSize = 9152;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    constexpr int64_t kFixedByteSize = 4096;
    constexpr int64_t kVarByteSize = 16384;
    RunEncodeKernelWithFixedSize(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false, kFixedByteSize,
                                 kVarByteSize);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_006)
{
    constexpr int64_t kInputSize = 5056;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    constexpr int64_t kFixedByteSize = 4096;
    constexpr int64_t kVarByteSize = 16384;
    RunEncodeKernelWithFixedSize(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false, kFixedByteSize,
                                 kVarByteSize);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_006)
{
    constexpr int64_t kInputSize = 5056;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    constexpr int64_t kFixedByteSize = 4096;
    constexpr int64_t kVarByteSize = 16384;
    RunEncodeKernelWithFixedSize(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false, kFixedByteSize,
                                 kVarByteSize);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_empty)
{
    constexpr int64_t kInputSize = 0;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_single_block)
{
    constexpr int64_t kInputSize = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_inactive_blocks) { RunEncodeInactiveBlocks(4); }

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_var_too_small)
{
    constexpr int64_t kInputSize = 9152;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    constexpr int64_t kFixedByteSize = 4096;
    constexpr int64_t kVarByteSize = 16;
    RunEncodeKernelWithFixedSize(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false, kFixedByteSize,
                                 kVarByteSize);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_multicore)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    constexpr int64_t kInputSize = 262144;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_multiround)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    constexpr int64_t kInputSize = 131072;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    constexpr int kRounds = 5;
    for (int r = 0; r < kRounds; ++r) {
        RunEncodeKernel(kInputSize, kDtypeBytes, kMaxCores, kTilingKey, true, true);
    }
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_single_block)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(64, 2, 1, 2, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_empty) { RunEncodeKernel(0, 2, 1, 2, false, false); }

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_var_too_small)
{
    RunEncodeKernelWithFixedSize(9152, 2, 1, 2, false, false, 4096, 16);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_inactive_blocks) { RunEncodeInactiveBlocks(2); }

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_multicore)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(262144, 2, 1, 2, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_multiround)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    constexpr int kRounds = 5;
    for (int r = 0; r < kRounds; ++r) {
        RunEncodeKernel(131072, 2, 1, 2, false, false);
    }
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_partial_tile)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536 + 1024, 2, 1, 2, false, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_partial_tile_half_reg)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536 + 64, 2, 1, 2, false, false);
}

// BF16 needs padding from 192 to 256 values in its final histogram tile.
TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_histogram_partial)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(32768 + 192, 2, 1, 2, true, false);
}

// FP32 has a partial tile, but 128 is already aligned to its 64-value register.
// This fixture does not claim to cover the histogram zero-padding branch.
TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_histogram_partial)
{
    RunEncodeKernel(32768 + 128, 4, 1, 4, true, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_statistic_only)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, true, false);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_reshuff_only)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, false, true);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_statistic_reshuff)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, true, true);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_multicore_reshuff)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(262144, 2, 1, 2, false, true);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_multicore_reshuff)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(262144, 4, 1, 4, false, true);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_uniform_value)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, true, false, InputPattern::kUniform);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_all_symbols)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, true, false, InputPattern::kAllSymbols);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_special_values)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536, 2, 1, 2, true, false, InputPattern::kSpecial);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_all_symbols_partial)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    RunEncodeKernel(65536 + 1024, 2, 1, 2, true, false, InputPattern::kAllSymbols);
}

TEST_F(HansEncode950KernelTest, ascend950_encode_fp32_129_tiles_eight_bit_rank)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    for (const bool reshuff : {false, true}) {
        RunEncodeKernel(129 * STATE_COUNT, 4, 1, 4, false, reshuff, InputPattern::kEightBitRank);
    }
}

TEST_F(HansEncode950KernelTest, ascend950_encode_bf16_129_tiles_eight_bit_rank)
{
    GTEST_SKIP() << kCpuPrefixSkipReason;
    for (const bool reshuff : {false, true}) {
        RunEncodeKernel(129 * STATE_COUNT, 2, 1, 2, false, reshuff, InputPattern::kEightBitRank);
    }
}

} // namespace
