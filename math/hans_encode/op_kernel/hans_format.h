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
 * \file hans_format.h
 * \brief Compatibility-critical constants shared by the Ascend 950 HANS kernels.
 *
 * This header is shared by device kernels and host tiling code, so it depends
 * only on <cstdint>. Do not include
 * device-only headers such as kernel_operator.h.
 */
#ifndef HANS_FORMAT_H
#define HANS_FORMAT_H

#include <cstdint>

namespace HansFormat {

constexpr int32_t MAGIC = 12138;
constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t STATE_COUNT = 4096;
constexpr int32_t GROUP_COUNT = STATE_COUNT / BLOCK_SIZE;
constexpr int32_t PDF_LENGTH = 256;

constexpr int32_t HEADER_INT32_COUNT = 128;
constexpr int32_t HEADER_BYTES = HEADER_INT32_COUNT * static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t DEVICE_START_IDX = 8;
constexpr int32_t HOST_START_IDX = 64;
constexpr int32_t MAX_CORE_COUNT = HOST_START_IDX - DEVICE_START_IDX;

constexpr int32_t HEADER_MAGIC_IDX = 0;
constexpr int32_t HEADER_CORE_COUNT_IDX = 1;
constexpr int32_t HEADER_TOTAL_LOOPS_IDX = 2;
constexpr int32_t HEADER_FIXED_LOOPS_IDX = 3;
constexpr int32_t HEADER_VAR_LOOPS_IDX = 4;

// Bit width of one overflow record, matching the uint16_t record element type.
// Encoding and decoding use this width for state >>= RECORD_BITS / state <<= RECORD_BITS,
// counter -= RECORD_BITS, and the overflow condition counter + bits > RECORD_BITS.
constexpr int32_t RECORD_BITS = 16;

constexpr int32_t STATE_TAIL_BYTES = STATE_COUNT * static_cast<int32_t>(sizeof(uint16_t));
constexpr int32_t COUNTER_TAIL_BYTES = GROUP_COUNT * static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t TAIL_BYTES = STATE_TAIL_BYTES + COUNTER_TAIL_BYTES;
constexpr int32_t GROUP_BITS_BYTES = GROUP_COUNT * static_cast<int32_t>(sizeof(uint16_t));
constexpr int32_t GROUP_OVERFLOW_BYTES = BLOCK_SIZE * RECORD_BITS / 8;

constexpr int32_t THREAD_COUNT = 1024;
constexpr uint32_t STATE_LOW_MASK = (1U << RECORD_BITS) - 1U;

// Ascend 950 reserves 32 KiB of UB for the SIMT DCache. Host tiling sets
// SetLocalMemorySize(platformUbSize - SIMT_DCACHE_BYTES), leaving
// 192 - 32 = 160 KiB of usable UB for the kernel, rather than 192 KiB.
constexpr int32_t TOTAL_UB_BYTES = 192 * 1024;
constexpr int32_t SIMT_DCACHE_BYTES = 32 * 1024;
constexpr int32_t USABLE_UB_BYTES = TOTAL_UB_BYTES - SIMT_DCACHE_BYTES;

// --- Format and layout invariants -----------------------------------------
static_assert(MAX_CORE_COUNT == 56, "The 512-byte HANS header can describe at most 56 cores.");
static_assert(TAIL_BYTES == 8448, "Changing the HANS per-core tail breaks bitstream compatibility.");
static_assert(HOST_START_IDX + MAX_CORE_COUNT <= HEADER_INT32_COUNT,
              "HOST region must not overflow the 512-byte header.");
static_assert(STATE_COUNT == GROUP_COUNT * BLOCK_SIZE, "STATE_COUNT must equal GROUP_COUNT * BLOCK_SIZE.");

// --- DMA alignment: payload offsets are sums of the following constants ----
static_assert(HEADER_BYTES % 32 == 0, "header DMA requires a 32-byte aligned length.");
static_assert(GROUP_BITS_BYTES % 32 == 0, "record group-bits block must be 32-byte aligned.");
static_assert(GROUP_OVERFLOW_BYTES % 32 == 0, "record overflow block must be 32-byte aligned.");
static_assert(STATE_TAIL_BYTES % 32 == 0, "state tail must be 32-byte aligned.");
static_assert(COUNTER_TAIL_BYTES % 32 == 0, "counter tail must be 32-byte aligned.");
static_assert(HEADER_BYTES <= 65535, "DataCopyPad blockLen is uint16_t.");

// --- Capacity requirements for the RegBase path ----------------------------
// Each dhistv2 repeat loads 256 B8 lanes; the caller zero-pads the tail to this boundary.
static_assert(STATE_COUNT % 256 == 0, "regbase histogram pads the tail to a 256-lane boundary.");
// Histogram accumulators are uint16; in the worst case, an entire tile falls into one bin.
static_assert(STATE_COUNT <= 65535, "uint16 dhistv2 accumulator must not overflow within one tile.");
// state = (state << bits) + rank, with bits <= 8 (rank < 256 when PDF_LENGTH is 256).
static_assert(RECORD_BITS + 8 <= 32, "state register must hold counter bits plus one group's bits.");

// --- SIMT thread partitioning ---------------------------------------------
static_assert(THREAD_COUNT % GROUP_COUNT == 0, "state-update threads must divide evenly across groups.");
static_assert(THREAD_COUNT >= PDF_LENGTH, "rank-LUT build requires one SIMT thread per PDF symbol.");

} // namespace HansFormat

#endif // HANS_FORMAT_H
