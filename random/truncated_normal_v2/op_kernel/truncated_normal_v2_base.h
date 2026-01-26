/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncated_normal_v2_base.h
 * \brief truncated_normal_v2_base
 */

#ifndef RANDOM_TRUNCATED_NORMAL_V2_BASE_H_
#define RANDOM_TRUNCATED_NORMAL_V2_BASE_H_

#include "kernel_operator.h"
#include "truncated_normal_v2_tiling_data.h"

namespace TruncatedNormalV2 {
using namespace AscendC;

constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr uint16_t ALG_RGIHT_BIT = 32;
constexpr float M_PI = 3.14159265358979323846f;
constexpr uint64_t RESERVED_SAMPLES_PER_OUTPUT = 256;
constexpr uint64_t GROUP_SIZE = 4;
constexpr uint32_t PHILOX_W32_A = 0x9E3779B9;
constexpr uint32_t PHILOX_W32_B = 0xBB67AE85;
constexpr uint32_t PHILOX_M4X32_A = 0xD2511F53;
constexpr uint32_t PHILOX_M4X32_B = 0xCD9E8D57;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr float RANDOM_THREAD_R = 2.0f;
constexpr float RANDOM_THREAD_L = -2.0f;
constexpr uint32_t MANTISSA_BIT = 23;

// Skip the specified number of samples of 128-bits in the current stream.
__aicore__ inline void Skip(const uint64_t count, uint32_t* counter)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> ALG_RGIHT_BIT);

    counter[0] += countLo;
    if (counter[0] < countLo) {
        ++countHi;
    }
    counter[1] += countHi;
    if (counter[1] < countHi) {
        if (++counter[IDX_2] == 0) {
            ++counter[IDX_3];
        }
    }
}

__aicore__ inline void CopyArray4(uint32_t* dst, const uint32_t* src)
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[IDX_2] = src[IDX_2];
    dst[IDX_3] = src[IDX_3];
}

// Helper function to skip the next sample of 128-bits in the current stream.
__aicore__ inline void SkipOne(uint32_t* counter)
{
    if (++counter[0] == 0) {
        if (++counter[1] == 0) {
            if (++counter[IDX_2] == 0) {
                ++counter[IDX_3];
            }
        }
    }
}

/*
 * Helper function to return the lower and higher 32-bits from two 32-bit
 * integer multiplications.
 */
__aicore__ inline void MultiplyHighLow(uint32_t a, uint32_t b, uint32_t* result_low, uint32_t* result_high)
{
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> ALG_RGIHT_BIT);
}

// Helper function for a single round of the underlying Philox algorithm.
__aicore__ inline void ComputeSingleRound(uint32_t* counter, const uint32_t* key)
{
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(PHILOX_M4X32_A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(PHILOX_M4X32_B, counter[IDX_2], &lo1, &hi1);

    uint32_t result[ALG_COUNTER_SIZE];
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[IDX_2] = hi0 ^ counter[IDX_3] ^ key[1];
    result[IDX_3] = lo0;

    CopyArray4(counter, result);
}

__aicore__ inline void RaiseKey(uint32_t* key)
{
    key[0] += PHILOX_W32_A;
    key[1] += PHILOX_W32_B;
}

/*
 * Returns counter: a group of four random numbers using the underlying Philox
 * algorithm.
 */
__aicore__ inline void PhiloxRandom(const uint32_t* keyCst, uint32_t* counter, uint32_t* counterOrig)
{
    uint32_t key[ALG_KEY_SIZE];
    CopyArray4(key, keyCst);
    /*
     * Run the single rounds for ten times. Manually unrolling the loop
     * for better performance.
     */
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    RaiseKey(key);
    ComputeSingleRound(counter, key);
    SkipOne(counterOrig);
}

// Helper function to convert an 32-bit integer to a float between [0..1).
__aicore__ inline float Uint32ToFloat(const uint32_t& x)
{
    const uint32_t man = x & 0x7fffffu; // 23 bit mantissa
    const uint32_t exp = static_cast<uint32_t>(127);
    const uint32_t val = (exp << MANTISSA_BIT) | man;
    float result = *reinterpret_cast<const float*>(&val);
    result = result - 1.0f;
    return result;
}

// This function implements the Box-Muller transform:
__aicore__ inline void BoxMullerFloat(const uint32_t& x0, const uint32_t& x1, float* f0, float* f1)
{
    const float eps = 1.0e-7f;
    float u1 = Uint32ToFloat(x0);
    if (u1 < eps) {
        u1 = eps;
    }
    float v1 = static_cast<float>(RANDOM_THREAD_R * M_PI * Uint32ToFloat(x1));
    float u2 = Simt::Sqrt(RANDOM_THREAD_L * Simt::Log(u1));
    Simt::Sincos(v1, *f0, *f1);
    *f0 *= u2;
    *f1 *= u2;
}

__aicore__ inline void FilterSample(float* results, int& index, float f0)
{
    if (Simt::Abs(f0) < RANDOM_THREAD_R) {
        results[index] = f0;
        index = index + 1;
    }
}

__aicore__ inline void GenSamples(float* results, const uint32_t* key, const uint32_t* counter)
{
    int index = 0;
    uint32_t counterSkip[ALG_COUNTER_SIZE];
    CopyArray4(counterSkip, counter);
    while (index < GROUP_SIZE) {
        uint32_t counterRst[ALG_COUNTER_SIZE];
        CopyArray4(counterRst, counterSkip);
        PhiloxRandom(key, counterRst, counterSkip);
        // Repeatedly take samples from the normal distribution, until we have
        // the desired number of elements that fall within the pre-defined cutoff
        // threshold.
        float f[IDX_2];
        BoxMullerFloat(counterRst[0], counterRst[1], &f[0], &f[1]);

        FilterSample(results, index, f[0]);
        if (index >= GROUP_SIZE) {
            return;
        }

        FilterSample(results, index, f[1]);
        if (index >= GROUP_SIZE) {
            return;
        }

        BoxMullerFloat(counterRst[IDX_2], counterRst[IDX_3], &f[0], &f[1]);

        FilterSample(results, index, f[0]);
        if (index >= GROUP_SIZE) {
            return;
        }

        FilterSample(results, index, f[1]);
        if (index >= GROUP_SIZE) {
            return;
        }
    }
}

} // namespace TruncatedNormalV2

#endif // RANDOM_TRUNCATED_NORMAL_V2_BASE_H_
