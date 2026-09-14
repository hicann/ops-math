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
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../op_host/hans_decode_tiling.h"
#include "../../../../../hans_encode/op_kernel/hans_format.h"

extern "C" __global__ __aicore__ void hans_decode(GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf,
                                                  GM_ADDR recover, GM_ADDR workspace, GM_ADDR tiling);

namespace {
using namespace HansFormat;
constexpr uint8_t kSentinel = 0xA5;
class HansDecode950KernelTest : public testing::Test {};

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

using HeaderMutation = std::function<void(int32_t*, optiling::HansDecodeTilingData&)>;

// Construct fixtures independently of the encoder. Fixed cases use one tile
// per core with rank 1, one bit per active group, and no overflow words.
// Equal PDF frequencies map rank 1 to symbol byte 1. Inactive states are zero.
class DecodeFixture {
public:
    DecodeFixture(int64_t count, int64_t bytes, int64_t cores, bool varOnly, bool reshuff)
        : bytes_(bytes),
          cores_(cores),
          fixedBytes_(varOnly ? HEADER_BYTES : HEADER_BYTES + cores * (GROUP_BITS_BYTES + TAIL_BYTES)),
          varBytes_(varOnly ? count : 32),
          mantissa_(count * (bytes - 1)),
          fixed_(fixedBytes_),
          var_(varBytes_),
          pdf_(PDF_LENGTH * sizeof(int32_t)),
          recover_(count * bytes),
          workspace_(16 * 1024 * 1024),
          tiling_(tilingData_.GetDataSize()),
          expected_(count * bytes)
    {
        const int64_t loops = count / BLOCK_SIZE;
        const int64_t loopsPerCore = loops / cores;
        std::memset(fixed_.Data(), 0, fixed_.Size());
        auto* hdr = Header();
        hdr[HEADER_MAGIC_IDX] = MAGIC;
        hdr[HEADER_CORE_COUNT_IDX] = static_cast<int32_t>(cores);
        hdr[HEADER_TOTAL_LOOPS_IDX] = static_cast<int32_t>(loops);
        hdr[HEADER_FIXED_LOOPS_IDX] = varOnly ? 0 : static_cast<int32_t>(loops);
        hdr[HEADER_VAR_LOOPS_IDX] = varOnly ? static_cast<int32_t>(loops) : 0;

        for (int64_t core = 0; core < cores; ++core) {
            const int64_t values = (loopsPerCore + (core == cores - 1 ? loops % cores : 0)) * BLOCK_SIZE;
            hdr[HOST_START_IDX + core] = varOnly ? static_cast<int32_t>(values) : 0;
            hdr[DEVICE_START_IDX + core] = varOnly ? 0 : GROUP_BITS_BYTES + TAIL_BYTES;
            if (!varOnly) {
                auto* payload = fixed_.Data() + HEADER_BYTES + core * (GROUP_BITS_BYTES + TAIL_BYTES);
                auto* bits = reinterpret_cast<uint16_t*>(payload);
                auto* states = reinterpret_cast<uint16_t*>(payload + GROUP_BITS_BYTES);
                auto* counters = reinterpret_cast<int32_t*>(payload + GROUP_BITS_BYTES + STATE_TAIL_BYTES);
                for (int64_t group = 0; group < values / BLOCK_SIZE; ++group) {
                    bits[group] = 1;
                    counters[group] = 1;
                }
                for (int64_t value = 0; value < values; ++value) {
                    states[value] = 1;
                }
            }
        }

        uint32_t seed = 1234;
        for (int64_t i = 0; i < count * (bytes - 1); ++i) {
            seed = seed * 1103515245U + 12345U;
            mantissa_.Data()[i] = static_cast<uint8_t>((seed >> 16) & 0xFFU);
        }
        auto* histogram = reinterpret_cast<int32_t*>(pdf_.Data());
        std::fill_n(histogram, PDF_LENGTH, varOnly ? 0 : 1);
        for (int64_t value = 0; value < count; ++value) {
            uint8_t symbol = 1;
            if (varOnly) {
                seed = seed * 1103515245U + 12345U;
                symbol = static_cast<uint8_t>((seed >> 16) & 0xFFU);
                var_.Data()[value] = symbol;
                ++histogram[symbol];
            }
            std::memcpy(expected_.data() + value * bytes, mantissa_.Data() + value * (bytes - 1), bytes - 1);
            expected_[value * bytes + bytes - 1] = symbol;
        }

        tilingData_.set_mantissaByteSize(count * (bytes - 1));
        tilingData_.set_fixedByteSize(fixedBytes_);
        tilingData_.set_recoverExpByteSize(count * bytes);
        tilingData_.set_recoverByteSize(count * bytes);
        tilingData_.set_varByteSize(varBytes_);
        tilingData_.set_outputValueCount(count);
        tilingData_.set_launchCoreDim(cores);
        tilingData_.set_reshuff(reshuff);
    }

    int32_t* Header() { return reinterpret_cast<int32_t*>(fixed_.Data()); }
    optiling::HansDecodeTilingData& Tiling() { return tilingData_; }

    void InvalidateFirstGroupBits()
    {
        auto* bits = reinterpret_cast<uint16_t*>(fixed_.Data() + HEADER_BYTES);
        bits[0] = 0;
    }

    void Run(uint64_t key, bool expectOutput)
    {
        ASSERT_EQ(key, static_cast<uint64_t>(bytes_));
        tilingData_.SaveToBuffer(tiling_.Data(), tiling_.Size());
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(key);
        ICPU_RUN_KF(hans_decode, static_cast<uint32_t>(cores_), mantissa_.Data(), fixed_.Data(), var_.Data(),
                    pdf_.Data(), recover_.Data(), workspace_.Data(), tiling_.Data());
        for (const GmBuffer* buffer : {&mantissa_, &fixed_, &var_, &pdf_, &recover_, &workspace_, &tiling_}) {
            buffer->CheckGuard();
        }
        if (expectOutput) {
            ASSERT_EQ(std::memcmp(recover_.Data(), expected_.data(), expected_.size()), 0)
                << "Decoded bytes must exactly match the constructed fixture.";
        } else {
            // All invalid fixtures fail before the first output write. This
            // does not promise transactional output for arbitrary late errors.
            recover_.CheckUnchanged();
        }
    }

private:
    int64_t bytes_;
    int64_t cores_;
    int64_t fixedBytes_;
    int64_t varBytes_;
    GmBuffer mantissa_;
    GmBuffer fixed_;
    GmBuffer var_;
    GmBuffer pdf_;
    GmBuffer recover_;
    GmBuffer workspace_;
    optiling::HansDecodeTilingData tilingData_;
    GmBuffer tiling_;
    std::vector<uint8_t> expected_;
};

void CheckParameters(int64_t count, int64_t bytes, int64_t cores, bool varOnly)
{
    ASSERT_GT(count, 0);
    ASSERT_EQ(count % BLOCK_SIZE, 0);
    ASSERT_TRUE(bytes == 2 || bytes == 4);
    ASSERT_GT(cores, 0);
    ASSERT_LE(cores, MAX_CORE_COUNT);
    ASSERT_GE(count / BLOCK_SIZE, cores);
    if (!varOnly) {
        ASSERT_EQ((count / BLOCK_SIZE) % cores, 0);
        ASSERT_LE(count / cores, STATE_COUNT) << "The fixed fixture models one tile per core.";
    }
}

void RunVarOnlyDecodeKernel(int64_t count, int64_t bytes, int64_t cores, uint64_t key, bool reshuff)
{
    ASSERT_NO_FATAL_FAILURE(CheckParameters(count, bytes, cores, true));
    DecodeFixture fixture(count, bytes, cores, true, reshuff);
    fixture.Run(key, true);
}

void RunFixedPayloadDecodeTest(int64_t count, int64_t bytes, int64_t cores, uint64_t key, bool reshuff)
{
    ASSERT_NO_FATAL_FAILURE(CheckParameters(count, bytes, cores, false));
    DecodeFixture fixture(count, bytes, cores, false, reshuff);
    fixture.Run(key, true);
}

void RunInvalidHeaderDecodeTest(int64_t count, int64_t bytes, int64_t cores, uint64_t key, bool reshuff,
                                const HeaderMutation& mutate)
{
    ASSERT_NO_FATAL_FAILURE(CheckParameters(count, bytes, cores, false));
    DecodeFixture fixture(count, bytes, cores, false, reshuff);
    // Start with a complete, decodable payload. A zero/invalid baseline would
    // allow rejection for the wrong reason even when the tested guard regresses.
    mutate(fixture.Header(), fixture.Tiling());
    fixture.Run(key, false);
}

void RunInvalidGroupBits(int64_t bytes, uint64_t key)
{
    DecodeFixture fixture(64, bytes, 1, false, false);
    fixture.InvalidateFirstGroupBits();
    fixture.Run(key, false);
}

// CPU-debug repeatability is not a substitute for device concurrency testing.

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_001) { RunVarOnlyDecodeKernel(65536, 4, 2, 4, false); }

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_001) { RunVarOnlyDecodeKernel(65536, 2, 2, 2, false); }

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_002)
{
    constexpr int64_t kRecoverElements = 65536;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 4;
    RunVarOnlyDecodeKernel(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_003)
{
    RunInvalidHeaderDecodeTest(4096, 4, 1, 4, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_MAGIC_IDX] = 99999; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_002)
{
    constexpr int64_t kRecoverElements = 65536;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 2;
    RunVarOnlyDecodeKernel(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_004)
{
    constexpr int64_t kRecoverElements = 4096;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunVarOnlyDecodeKernel(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_003)
{
    constexpr int64_t kRecoverElements = 4096;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunVarOnlyDecodeKernel(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_fixed_001)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_fixed_001)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_fixed_002)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_fixed_003)
{
    constexpr int64_t kRecoverElements = 128;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 4;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_fixed_004)
{
    constexpr int64_t kRecoverElements = 4096;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_fixed_002)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_invalid_magic)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_MAGIC_IDX] = 0; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_invalid_core_count)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_CORE_COUNT_IDX] = 0; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_invalid_total_loops)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_TOTAL_LOOPS_IDX] = 0; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_negative_host_values)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HOST_START_IDX] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_negative_device_payload)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_payload_too_small)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = 100; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_cross_core_fail)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) {
                                   hdr[HOST_START_IDX] = 64;
                                   hdr[HEADER_VAR_LOOPS_IDX] = 1;
                                   hdr[HEADER_FIXED_LOOPS_IDX] = 0;
                               });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_payload_start_oob)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = 100000; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bits_start_oob)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) {
                                   hdr[DEVICE_START_IDX] = static_cast<int32_t>(TAIL_BYTES);
                               });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_invalid_group_bits) { RunInvalidGroupBits(4, 4); }

TEST_F(HansDecode950KernelTest, ascend950_decode_cross_core_negative_host)
{
    constexpr int64_t kRecoverElements = 128;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HOST_START_IDX + 1] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_reshuff_negative_core_bytes)
{
    constexpr int64_t kRecoverElements = 128;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fixed_area_negative)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData& tiling) {
                                   (void)hdr;
                                   tiling.set_fixedByteSize(100);
                               });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_fixed_003)
{
    constexpr int64_t kRecoverElements = 256;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_fixed_004)
{
    constexpr int64_t kRecoverElements = 512;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_fixed_005)
{
    constexpr int64_t kRecoverElements = 128;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 4;
    RunFixedPayloadDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_invalid_magic)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_MAGIC_IDX] = 0; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_invalid_core_count)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_CORE_COUNT_IDX] = 0; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_invalid_total_loops)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HEADER_TOTAL_LOOPS_IDX] = 0; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_negative_host_values)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HOST_START_IDX] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_negative_device_payload)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_payload_too_small)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = 100; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_cross_core_fail)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) {
                                   hdr[HOST_START_IDX] = 64;
                                   hdr[HEADER_VAR_LOOPS_IDX] = 1;
                                   hdr[HEADER_FIXED_LOOPS_IDX] = 0;
                               });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_payload_start_oob)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = 100000; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_bits_start_oob)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) {
                                   hdr[DEVICE_START_IDX] = static_cast<int32_t>(TAIL_BYTES);
                               });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_cross_core_negative_host)
{
    constexpr int64_t kRecoverElements = 128;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[HOST_START_IDX + 1] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_reshuff_negative_core_bytes)
{
    constexpr int64_t kRecoverElements = 128;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 2;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true,
                               [](int32_t* hdr, optiling::HansDecodeTilingData&) { hdr[DEVICE_START_IDX] = -1; });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_fixed_area_negative)
{
    constexpr int64_t kRecoverElements = 64;
    constexpr int64_t kDtypeBytes = 2;
    constexpr int64_t kMaxCores = 1;
    constexpr uint64_t kTilingKey = 2;
    RunInvalidHeaderDecodeTest(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false,
                               [](int32_t* hdr, optiling::HansDecodeTilingData& tiling) {
                                   (void)hdr;
                                   tiling.set_fixedByteSize(100);
                               });
}

TEST_F(HansDecode950KernelTest, ascend950_decode_bf16_invalid_group_bits) { RunInvalidGroupBits(2, 2); }

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_multicore)
{
    constexpr int64_t kRecoverElements = 65536;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 8;
    constexpr uint64_t kTilingKey = 4;
    RunVarOnlyDecodeKernel(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, false);
}

TEST_F(HansDecode950KernelTest, ascend950_decode_fp32_multiround)
{
    constexpr int64_t kRecoverElements = 65536;
    constexpr int64_t kDtypeBytes = 4;
    constexpr int64_t kMaxCores = 4;
    constexpr uint64_t kTilingKey = 4;
    constexpr int kRounds = 5;
    for (int r = 0; r < kRounds; ++r) {
        RunVarOnlyDecodeKernel(kRecoverElements, kDtypeBytes, kMaxCores, kTilingKey, true);
    }
}

} // namespace
