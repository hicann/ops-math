/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAGGED_BIN_COUNT_PRECISION_POLICY_H
#define RAGGED_BIN_COUNT_PRECISION_POLICY_H

#include <cstdint>

namespace NsRaggedBinCount {

// A VALUE-mapped weighted launch enters the deterministic partial-reduction
// path only for this high-contention subdomain.  Keep the policy shared by the
// host workspace calculation and the kernel dispatch so they cannot drift.
constexpr int64_t PRECISE_VALUE_MAX_BINS = 2;
constexpr int64_t PRECISE_VALUE_MAX_WORK = 262144;

// Each core emits this many fixed-order partials per output element.  A slot
// stores high, low, flags and padding (16 bytes); the first 32 user-workspace
// bytes remain reserved for the whole-input validation flag.
constexpr int64_t PRECISE_VALUE_PARTITIONS_PER_CORE = 16;
constexpr uint64_t PRECISE_VALUE_PARTIAL_BYTES = 16U;
constexpr uint64_t USER_WORKSPACE_HEADER_BYTES = 32U;

// Deterministic mode may reduce arbitrarily long rows. Bound its partials by
// output size, independently of the small-input precision optimization above.
constexpr uint64_t DETERMINISTIC_WORKSPACE_LIMIT = 16U * 1024U * 1024U;
constexpr uint64_t DETERMINISTIC_PARTIAL_BYTES = 64U;
constexpr uint64_t DETERMINISTIC_HISTOGRAM_LIMIT = 64U * 1024U * 1024U;
constexpr uint64_t DETERMINISTIC_HISTOGRAM_SLOT_BYTES = 128U;
constexpr uint32_t DETERMINISTIC_ZERO_BUFFER_BYTES = 32U * 1024U;
constexpr int64_t DETERMINISTIC_WEIGHT_VALUES_PER_PARTITION = 256;
constexpr int64_t DETERMINISTIC_COUNT_VALUES_PER_PARTITION = 1024;
constexpr int64_t DETERMINISTIC_SMALL_WEIGHTED_OUTPUTS = 256;
constexpr int64_t DETERMINISTIC_DIRECT_ROW_VALUES = 256;

#ifdef __CCE_AICORE__
__aicore__ inline int64_t DeterministicPartitionsPerCore(int64_t outputElements)
#else
constexpr int64_t DeterministicPartitionsPerCore(int64_t outputElements)
#endif
{
    // Fill a VF even for a single hot bin, without growing large-output partials.
    return outputElements > 0 && outputElements < 64 ? (1024 + outputElements - 1) / outputElements :
                                                       PRECISE_VALUE_PARTITIONS_PER_CORE;
}

#ifdef __CCE_AICORE__
__aicore__ inline int64_t DeterministicPartitionsPerCore(int64_t numValues, int64_t outputElements, uint32_t coreNum,
                                                         bool hasWeights)
#else
constexpr int64_t DeterministicPartitionsPerCore(int64_t numValues, int64_t outputElements, uint32_t coreNum,
                                                 bool hasWeights)
#endif
{
    if (coreNum == 0U || numValues <= 0 || outputElements <= 0) {
        return 1;
    }
    const int64_t perCore = numValues / coreNum + static_cast<int64_t>(numValues % coreNum != 0);
    const int64_t grain = hasWeights && outputElements <= DETERMINISTIC_SMALL_WEIGHTED_OUTPUTS ?
                              DETERMINISTIC_WEIGHT_VALUES_PER_PARTITION :
                              DETERMINISTIC_COUNT_VALUES_PER_PARTITION;
    const int64_t needed = perCore / grain + static_cast<int64_t>(perCore % grain != 0);
    const uint64_t capacity = DETERMINISTIC_WORKSPACE_LIMIT / DETERMINISTIC_PARTIAL_BYTES /
                              static_cast<uint64_t>(outputElements) / coreNum;
    const int64_t budget = capacity > 0U ? static_cast<int64_t>(capacity) : 1;
    const int64_t base = DeterministicPartitionsPerCore(outputElements);
    const int64_t maximum = base < budget ? base : budget;
    return needed < maximum ? needed : maximum;
}

#ifdef __CCE_AICORE__
__aicore__ inline bool UseDeterministicPartials(int64_t outputElements, uint32_t coreNum)
#else
constexpr bool UseDeterministicPartials(int64_t outputElements, uint32_t coreNum)
#endif
{
    return coreNum > 0U && outputElements > 0 &&
           static_cast<uint64_t>(outputElements) <=
               DETERMINISTIC_WORKSPACE_LIMIT / DETERMINISTIC_PARTIAL_BYTES / coreNum;
}

#ifdef __CCE_AICORE__
__aicore__ inline bool UseDeterministicHistogram(int64_t outputElements)
#else
constexpr bool UseDeterministicHistogram(int64_t outputElements)
#endif
{
    return outputElements > 0 &&
           static_cast<uint64_t>(outputElements) <= DETERMINISTIC_HISTOGRAM_LIMIT / DETERMINISTIC_HISTOGRAM_SLOT_BYTES;
}

#ifdef __CCE_AICORE__
__aicore__ inline bool UseDeterministicDirect(int64_t numValues, int64_t numRows)
#else
constexpr bool UseDeterministicDirect(int64_t numValues, int64_t numRows)
#endif
{
    return numRows > 0 &&
           numValues / numRows + static_cast<int64_t>(numValues % numRows != 0) <= DETERMINISTIC_DIRECT_ROW_VALUES;
}

#ifdef __CCE_AICORE__
__aicore__ inline bool UsePreciseValuePath(int64_t numValues, int64_t numBins)
#else
constexpr bool UsePreciseValuePath(int64_t numValues, int64_t numBins)
#endif
{
    return numValues > 0 && numBins > 0 && numBins <= PRECISE_VALUE_MAX_BINS &&
           numValues <= PRECISE_VALUE_MAX_WORK / numBins;
}

} // namespace NsRaggedBinCount

#endif // RAGGED_BIN_COUNT_PRECISION_POLICY_H
