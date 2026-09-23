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
 * \file top_k_small_bitonic_packed_simt.h
 * \brief SIMT (warp 级) 路线：packed 网络与多行调度，以及按类型选择后端的统一入口。
 */
#ifndef TOP_K_SMALL_BITONIC_PACKED_SIMT_H
#define TOP_K_SMALL_BITONIC_PACKED_SIMT_H

#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/math_functions.h"
#include "top_k_constant_var_simd.h"
#include "top_k_util_type_simd.h"
#include "top_k_small_bitonic_common.h"
#include "top_k_small_bitonic_reg_finalize.h"

namespace topkV2 {
using namespace AscendC;

/*!
 * \brief 判断 lhs 是否"优于" rhs (SIMT 标量路径)。
 *
 * 与 Reg 路径的 BitonicSmallRegValueBetter 语义一致，但用标量比较实现。
 * 浮点类型统一转 float 后比较，NaN 语义：isLargest 时 NaN 视为最大，否则最小。
 */
template <typename T>
__simt_callee__ inline bool BitonicSmallValueBetter(T lhs, T rhs, bool isLargest)
{
    if constexpr (IsBitonicFloatType<T>) {
        float lhsFloat = static_cast<float>(lhs);
        float rhsFloat = static_cast<float>(rhs);
        bool lhsNan = isnan(lhsFloat);
        bool rhsNan = isnan(rhsFloat);
        if (isLargest) {
            return (lhsNan && !rhsNan) || (lhsFloat > rhsFloat);
        }
        return (rhsNan && !lhsNan) || (lhsFloat < rhsFloat);
    }
    return isLargest ? (lhs > rhs) : (lhs < rhs);
}

/*!
 * \brief 判断 lhs 和 rhs 是否等价 (SIMT 标量路径)。
 */
template <typename T>
__simt_callee__ inline bool BitonicSmallValueEquivalent(T lhs, T rhs)
{
    if constexpr (IsBitonicFloatType<T>) {
        float lhsFloat = static_cast<float>(lhs);
        float rhsFloat = static_cast<float>(rhs);
        return (isnan(lhsFloat) && isnan(rhsFloat)) || (lhsFloat == rhsFloat);
    }
    return lhs == rhs;
}

/*!
 * \brief 返回无效索引标记值 (SIMT 标量路径)。
 *
 * 无效候选 lane 的索引设为 -1 (全 1 位模式) 哨兵，排序网络按索引是否等于
 * 该哨兵判定有效性，并将无效项排到末尾。
 */
template <typename IndexT>
__simt_callee__ inline IndexT BitonicSmallInvalidIndex()
{
    return static_cast<IndexT>(-1);
}

/*!
 * \brief 双调网络的单次 compare-swap 阶段 (SIMT warp 级路径)。
 *
 * 每个 lane 线程持有一个元素，通过 asc_shfl_xor 获取对端 lane 的数据。
 * CompareValue=true 时按值比较（含 NaN 语义），false 时按 index 比较（用于分组排序）。
 * 交换逻辑与 Reg 路径一致，但用标量条件判断 + 直接赋值实现。
 */
template <typename T, typename IndexT, bool CompareValue>
__simt_callee__ inline void BitonicSmallSwapStage(T& value, IndexT& index, uint32_t stride, uint32_t size,
                                                  bool isLargest)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    T peerValue = asc_shfl_xor(value, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
    IndexT peerIndex = asc_shfl_xor(index, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);

    bool isLow = (lane & stride) == 0U;
    T valueA = isLow ? value : peerValue;
    T valueB = isLow ? peerValue : value;
    IndexT indexA = isLow ? index : peerIndex;
    IndexT indexB = isLow ? peerIndex : index;
    IndexT invalid = BitonicSmallInvalidIndex<IndexT>();
    bool validA = indexA != invalid;
    bool validB = indexB != invalid;
    bool comp = CompareValue ? BitonicSmallValueBetter<T>(valueA, valueB, isLargest) : (indexA < indexB);
    bool swap = (comp && validA) || !validB;

    uint32_t lowLane = lane & ~stride;
    uint32_t comparatorLane = (lowLane / (stride * 2U)) * stride + (lowLane % stride);
    bool dir = size != BITONIC_SMALL_TOPK_SIZE && ((comparatorLane & (size / 2U)) != 0U);
    if (swap == dir) {
        value = peerValue;
        index = peerIndex;
    }
}

/*!
 * \brief 双调排序网络完整序列 (SIMT warp 级路径)。
 *
 * 用两层 for 循环展开 15 个 SwapStage，比 Reg 路径的模板展开更紧凑。
 * CompareValue=true 时按值排序，false 时按 index 排序（用于恢复原始顺序）。
 */
template <typename T, typename IndexT, bool CompareValue>
__simt_callee__ inline void BitonicSmallBitonicNetwork(T& value, IndexT& index, bool isLargest)
{
    for (uint32_t size = 2U; size < BITONIC_SMALL_TOPK_SIZE; size *= 2U) {
        for (uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            BitonicSmallSwapStage<T, IndexT, CompareValue>(value, index, stride, size, isLargest);
        }
    }
    for (uint32_t stride = BITONIC_SMALL_TOPK_SIZE / 2U; stride > 0U; stride /= 2U) {
        BitonicSmallSwapStage<T, IndexT, CompareValue>(value, index, stride, BITONIC_SMALL_TOPK_SIZE, isLargest);
    }
}

/*!
 * \brief 全等场景的单次 compare-swap (SIMT warp 级路径)。
 *
 * 当所有候选值相等时，无需值比较，仅按 index 排序。
 * 交换条件：对端 index 无效时交换，将无效项挤到末尾。
 */
template <typename IndexT>
__simt_callee__ inline void BitonicSmallAllEqualSwapStage(IndexT& index, uint32_t stride, uint32_t size)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    IndexT peerIndex = asc_shfl_xor(index, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
    bool isLow = (lane & stride) == 0U;
    IndexT indexB = isLow ? peerIndex : index;
    bool swap = indexB == BitonicSmallInvalidIndex<IndexT>(); // invalidIndex = -1

    uint32_t lowLane = lane & ~stride;
    uint32_t comparatorLane = (lowLane / (stride * 2U)) * stride + (lowLane % stride);
    bool dir = size != BITONIC_SMALL_TOPK_SIZE && ((comparatorLane & (size / 2U)) != 0U);
    if (swap == dir) {
        index = peerIndex;
    }
}

/*!
 * \brief 全等场景的双调排序网络 (SIMT warp 级路径)。
 *
 * 当所有候选值相等时，仅按 index 排序，是 BitonicSmallBitonicNetwork 的轻量特化版。
 */
template <typename IndexT>
__simt_callee__ inline void BitonicSmallAllEqualBitonicNetwork(IndexT& index)
{
    for (uint32_t size = 2U; size < BITONIC_SMALL_TOPK_SIZE; size *= 2U) {
        for (uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            BitonicSmallAllEqualSwapStage<IndexT>(index, stride, size);
        }
    }
    for (uint32_t stride = BITONIC_SMALL_TOPK_SIZE / 2U; stride > 0U; stride /= 2U) {
        BitonicSmallAllEqualSwapStage<IndexT>(index, stride, BITONIC_SMALL_TOPK_SIZE);
    }
}

/*!
 * \brief 在位掩码中找到第 rank 个置位的位置 (SIMT 标量路径)。
 *
 * 用二分法在 32 位掩码中查找：每次将掩码分为高低半部，统计低位置位数，
 * 若 rank 落在低位则保留低位掩码，否则减去低位计数并右移。
 * 用于 SIMT finalize 中重构 BITONIC 兼容的 gather 源 lane 顺序。
 */
__simt_callee__ inline uint32_t BitonicSmallNthSetBit(uint32_t mask, uint32_t rank)
{
    uint32_t base = 0U;
    for (uint32_t halfSize = 16U; halfSize > 0U; halfSize /= 2U) {
        uint32_t lowerMask = mask & ((1U << halfSize) - 1U);
        uint32_t lowerCount = static_cast<uint32_t>(__popc(lowerMask));
        if (rank < lowerCount) {
            mask = lowerMask;
        } else {
            rank -= lowerCount;
            mask >>= halfSize;
            base += halfSize;
        }
    }
    return base;
}

/*!
 * \brief 编译期 log2，仅用于 2 的幂。
 */
__aicore__ constexpr uint32_t BitonicSmallConstLog2(uint32_t value)
{
    uint32_t logValue = 0U;
    while (value > 1U) {
        value >>= 1U;
        ++logValue;
    }
    return logValue;
}

/*!
 * \brief 打包网络的单次 compare-swap 阶段 (小 k 分档打包，SIMT warp 级路径)。
 */
template <typename T, typename IndexT, bool CompareValue, uint32_t SEG_WIDTH>
__simt_callee__ inline void BitonicSmallPackedSwapStage(T& value, IndexT& index, uint32_t stride, uint32_t size,
                                                        bool isLargest)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t local = lane & (SEG_WIDTH - 1U);
    T peerValue = asc_shfl_xor(value, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
    IndexT peerIndex = asc_shfl_xor(index, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);

    bool isLow = (local & stride) == 0U;
    T valueA = isLow ? value : peerValue;
    T valueB = isLow ? peerValue : value;
    IndexT indexA = isLow ? index : peerIndex;
    IndexT indexB = isLow ? peerIndex : index;
    IndexT invalid = BitonicSmallInvalidIndex<IndexT>();
    bool validA = indexA != invalid;
    bool validB = indexB != invalid;
    bool comp = CompareValue ? BitonicSmallValueBetter<T>(valueA, valueB, isLargest) : (indexA < indexB);
    bool swap = (comp && validA) || !validB;

    uint32_t lowLocal = local & ~stride;
    uint32_t comparatorLocal = (lowLocal / (stride * 2U)) * stride + (lowLocal % stride);
    bool dir = size != BITONIC_SMALL_TOPK_SIZE && ((comparatorLocal & (size / 2U)) != 0U);
    if (swap == dir) {
        value = peerValue;
        index = peerIndex;
    }
}

/*!
 * \brief 打包网络完整序列 (SIMT warp 级路径)。
 */
template <typename T, typename IndexT, bool CompareValue, uint32_t SEG_WIDTH, uint32_t EXTRA_PASSES>
__simt_callee__ inline void BitonicSmallPackedBitonicNetwork(T& value, IndexT& index, bool isLargest)
{
    for (uint32_t size = 2U; size <= SEG_WIDTH; size *= 2U) {
        for (uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            BitonicSmallPackedSwapStage<T, IndexT, CompareValue, SEG_WIDTH>(value, index, stride, size, isLargest);
        }
    }
    for (uint32_t pass = 0U; pass < EXTRA_PASSES; ++pass) {
        for (uint32_t stride = SEG_WIDTH / 2U; stride > 0U; stride /= 2U) {
            BitonicSmallPackedSwapStage<T, IndexT, CompareValue, SEG_WIDTH>(value, index, stride,
                                                                            BITONIC_SMALL_TOPK_SIZE, isLargest);
        }
    }
}

/*!
 * \brief 打包收尾的 full 段重收集 (SIMT warp 级路径)。
 */
template <typename T, typename IndexT, bool IsLargest, uint32_t SEG_WIDTH, uint32_t EXTRA_PASSES>
__simt_callee__ inline void BitonicSmallPackedFullRegather(T& value, IndexT& index, bool valid, bool rowActive,
                                                           uint32_t k, uint32_t segBase, uint32_t local)
{
    constexpr uint32_t SEG_MASK_ALL = (1U << SEG_WIDTH) - 1U;
    T threshold = asc_shfl(value, static_cast<int32_t>(segBase + k - 1U), BITONIC_SMALL_TOPK_SIZE);
    BitonicSmallPackedBitonicNetwork<T, IndexT, false, SEG_WIDTH, 0U>(value, index, IsLargest);
    bool strict = valid && BitonicSmallValueBetter<T>(value, threshold, IsLargest);
    bool equal = valid && BitonicSmallValueEquivalent<T>(value, threshold);
    uint32_t strictBallot = asc_ballot(static_cast<int32_t>(strict));
    uint32_t equalBallot = asc_ballot(static_cast<int32_t>(equal));
    uint32_t segStrict = (strictBallot >> segBase) & SEG_MASK_ALL;
    uint32_t segEqual = (equalBallot >> segBase) & SEG_MASK_ALL;
    uint32_t strictCount = static_cast<uint32_t>(__popc(segStrict));
    uint32_t sourceLane = 0U;
    if (local < k) {
        uint32_t rank = local < strictCount ? local : local - strictCount;
        sourceLane = segBase + BitonicSmallNthSetBit(local < strictCount ? segStrict : segEqual, rank);
    }
    value = asc_shfl(value, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    index = asc_shfl(index, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    if (local >= k || !rowActive) {
        index = BitonicSmallInvalidIndex<IndexT>();
    }
    BitonicSmallPackedBitonicNetwork<T, IndexT, true, SEG_WIDTH, EXTRA_PASSES>(value, index, IsLargest);
}

/*!
 * \brief 打包收尾选择函数 (SIMT warp 级路径)：一个 warp 的 32 lane 按
 *        SEG_WIDTH 分段承载 32/SEG_WIDTH 行，一套指令流驱动全部行。
 */
template <typename T, typename IndexT, bool IsLargest, uint32_t SEG_WIDTH>
__simt_callee__ inline bool BitonicSmallPackedFinalizeSelection(T& value, IndexT& index, bool valid, bool rowActive,
                                                                uint32_t k)
{
    // 跨段合并到全网络宽度所需的额外遍数 = log2(网络宽度) - log2(段宽)
    static_assert(SEG_WIDTH >= 2U && SEG_WIDTH <= BITONIC_SMALL_TOPK_SIZE && (SEG_WIDTH & (SEG_WIDTH - 1U)) == 0U,
                  "SEG_WIDTH must be a power of two in [2, BITONIC_SMALL_TOPK_SIZE]");
    constexpr uint32_t EXTRA_PASSES = BitonicSmallConstLog2(BITONIC_SMALL_TOPK_SIZE) - BitonicSmallConstLog2(SEG_WIDTH);
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t seg = lane / SEG_WIDTH;
    uint32_t segBase = seg * SEG_WIDTH;
    uint32_t local = lane - segBase;
    constexpr uint32_t SEG_MASK_ALL = (1U << SEG_WIDTH) - 1U;
    uint32_t allEqualBits = (1U << k) - 2U;

    T previous = asc_shfl(value, static_cast<int32_t>(lane - (local == 0U ? 0U : 1U)), BITONIC_SMALL_TOPK_SIZE);
    bool duplicate = valid && local > 0U && BitonicSmallValueEquivalent<T>(value, previous);
    uint32_t duplicateBallot = asc_ballot(static_cast<int32_t>(duplicate));
    uint32_t segDup = (duplicateBallot >> segBase) & SEG_MASK_ALL;
    bool segClean = segDup == 0U;
    bool segAllEqual = (segDup & allEqualBits) == allEqualBits;
    bool segFull = rowActive && !segClean && !segAllEqual;
    bool segNeedAllEqual = rowActive && segAllEqual;
    uint32_t fullBallot = asc_ballot(static_cast<int32_t>(segFull));
    uint32_t allEqualBallot = asc_ballot(static_cast<int32_t>(segNeedAllEqual));
    if (fullBallot == 0U && allEqualBallot == 0U) {
        return false;
    }

    if (fullBallot != 0U) {
        BitonicSmallPackedFullRegather<T, IndexT, IsLargest, SEG_WIDTH, EXTRA_PASSES>(value, index, valid, rowActive, k,
                                                                                      segBase, local);
    } else {
        BitonicSmallPackedBitonicNetwork<T, IndexT, true, SEG_WIDTH, EXTRA_PASSES>(value, index, IsLargest);
    }
    return valid && !segClean;
}

/*!
 * \brief SIMT 收尾选择的全等快路径：k 个候选值全部等价时的终排与写回。
 *
 * 整数类型与值无关，仅按 index 排序 (AllEqual 网络)；浮点类型虽值全等仍走
 * 含 NaN 位序的完整值网络。仅前 k 个 lane 写回。
 */
template <typename T, typename RegT, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallFinalizeSelectionAllEqual(RegT& regValue, IndexT& regIndex, T& value,
                                                                  IndexT& index, uint32_t lane, uint32_t k)
{
    if constexpr (std::is_integral_v<T>) {
        BitonicSmallAllEqualBitonicNetwork<IndexT>(regIndex);
        if (lane < k) {
            index = regIndex;
        }
    } else {
        BitonicSmallBitonicNetwork<RegT, IndexT, true>(regValue, regIndex, IsLargest);
        if (lane < k) {
            value = static_cast<T>(regValue);
            index = regIndex;
        }
    }
}

/*!
 * \brief 收尾选择函数 (SIMT warp 级路径)，处理已按值排序的候选集。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallFinalizeSelection(T& value, IndexT& index, bool valid, uint32_t k)
{
    using RegT = BitonicSmallRegType<T>;
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    RegT regValue = valid ? static_cast<RegT>(value) : static_cast<RegT>(0);
    IndexT regIndex = valid ? index : BitonicSmallInvalidIndex<IndexT>();

    RegT previous = asc_shfl(regValue, static_cast<int32_t>(lane == 0U ? 0U : lane - 1U), BITONIC_SMALL_TOPK_SIZE);
    bool duplicate = valid && lane > 0U && BitonicSmallValueEquivalent<RegT>(regValue, previous);
    uint32_t duplicateMask = asc_ballot(static_cast<int32_t>(duplicate));
    if (duplicateMask == 0U) {
        return;
    }

    uint32_t allEqualBits = k >= BITONIC_SMALL_TOPK_SIZE ? 0xFFFFFFFEU : ((1U << k) - 2U);
    if ((duplicateMask & allEqualBits) == allEqualBits) {
        BitonicSmallFinalizeSelectionAllEqual<T, RegT, IndexT, IsLargest>(regValue, regIndex, value, index, lane, k);
        return;
    }

    // Selection candidates are value-sorted, so the last valid lane is the kth threshold.
    RegT threshold = asc_shfl(regValue, static_cast<int32_t>(k - 1U), BITONIC_SMALL_TOPK_SIZE);

    // Reconstruct BITONIC's strict-better/equal gather order inside the selected k candidates.
    BitonicSmallBitonicNetwork<RegT, IndexT, false>(regValue, regIndex, false);
    bool strict = valid && BitonicSmallValueBetter<RegT>(regValue, threshold, IsLargest);
    bool equal = valid && BitonicSmallValueEquivalent<RegT>(regValue, threshold);
    uint32_t strictMask = asc_ballot(static_cast<int32_t>(strict));
    uint32_t equalMask = asc_ballot(static_cast<int32_t>(equal));
    uint32_t strictCount = static_cast<uint32_t>(__popc(strictMask));
    uint32_t sourceLane = 0U;
    if (lane < k) {
        uint32_t rank = lane < strictCount ? lane : lane - strictCount;
        sourceLane = BitonicSmallNthSetBit(lane < strictCount ? strictMask : equalMask, rank);
    }
    regValue = asc_shfl(regValue, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    regIndex = asc_shfl(regIndex, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    if (lane >= k) {
        regIndex = BitonicSmallInvalidIndex<IndexT>();
    }

    BitonicSmallBitonicNetwork<RegT, IndexT, true>(regValue, regIndex, IsLargest);
    if (lane < k) {
        value = static_cast<T>(regValue);
        index = regIndex;
    }
}

/*!
 * \brief 收尾选择函数 (SIMT warp 级路径)，处理未排序的精确候选集。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallFinalizeExactSelection(T& value, IndexT& index, bool valid, uint32_t k)
{
    using RegT = BitonicSmallRegType<T>;
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    RegT regValue = valid ? static_cast<RegT>(value) : static_cast<RegT>(0);
    IndexT regIndex = valid ? index : BitonicSmallInvalidIndex<IndexT>();

    // The gathered candidates are exact but not value-sorted. Reduce their worst value to recover the kth threshold.
    RegT firstValue = asc_shfl(regValue, 0, BITONIC_SMALL_TOPK_SIZE);
    RegT threshold = valid ? regValue : firstValue;
    for (uint32_t stride = BITONIC_SMALL_TOPK_SIZE / 2U; stride > 0U; stride /= 2U) {
        RegT peer = asc_shfl_xor(threshold, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
        if (BitonicSmallValueBetter<RegT>(threshold, peer, IsLargest)) {
            threshold = peer;
        }
    }

    // Restore BITONIC's source-order strict/equal gather before applying SmallBitonicSort<32>.
    BitonicSmallBitonicNetwork<RegT, IndexT, false>(regValue, regIndex, false);
    bool strict = valid && BitonicSmallValueBetter<RegT>(regValue, threshold, IsLargest);
    bool equal = valid && BitonicSmallValueEquivalent<RegT>(regValue, threshold);
    uint32_t strictMask = asc_ballot(static_cast<int32_t>(strict));
    uint32_t equalMask = asc_ballot(static_cast<int32_t>(equal));
    uint32_t strictCount = static_cast<uint32_t>(__popc(strictMask));
    uint32_t sourceLane = 0U;
    if (lane < k) {
        uint32_t rank = lane < strictCount ? lane : lane - strictCount;
        sourceLane = BitonicSmallNthSetBit(lane < strictCount ? strictMask : equalMask, rank);
    }
    regValue = asc_shfl(regValue, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    regIndex = asc_shfl(regIndex, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    if (!valid) {
        regIndex = BitonicSmallInvalidIndex<IndexT>();
    }
    BitonicSmallBitonicNetwork<RegT, IndexT, true>(regValue, regIndex, IsLargest);
    if (valid) {
        value = static_cast<T>(regValue);
        index = regIndex;
    }
}

/*!
 * \brief 按阈值 key 从输入中 gather 候选元素 (SIMT Kernel)。
 *
 * 两趟扫描：
 *   pass 0：收集 key < threshold（严格优于阈值）的元素。
 *   pass 1：收集 key == threshold（等于阈值）的元素。
 * 用 asc_ballot + __popc 做 warp 内前缀和，确定每个选中元素的输出位置。
 * outputPos = written + rank，确保不超出 quota。
 */
template <typename T, typename KeyT, typename IndexT>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_SIZE) __aicore__
    void BitonicGatherThresholdTileKernel(uint32_t axisSize, uint32_t quota, uint64_t indexBase, KeyT threshold,
                                          __ubuf__ T* inputValues, __ubuf__ KeyT* keys, __ubuf__ T* outputValues,
                                          __ubuf__ IndexT* outputIndices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t written = 0U;
    for (uint32_t pass = 0U; pass < 2U && written < quota; ++pass) {
        for (uint32_t base = 0U; base < axisSize && written < quota; base += BITONIC_SMALL_TOPK_SIZE) {
            uint32_t col = base + lane;
            bool inRange = col < axisSize;
            KeyT key = inRange ? keys[col] : static_cast<KeyT>(0);
            bool take = inRange && (pass == 0U ? key < threshold : key == threshold);
            uint32_t takeMask = asc_ballot(static_cast<int32_t>(take));
            uint32_t lowerMask = lane == 0U ? 0U : ((1U << lane) - 1U);
            uint32_t rank = static_cast<uint32_t>(__popc(takeMask & lowerMask));
            uint32_t outputPos = written + rank;
            if (take && outputPos < quota) {
                outputValues[outputPos] = inputValues[col];
                outputIndices[outputPos] = static_cast<IndexT>(indexBase + col);
            }
            written += static_cast<uint32_t>(__popc(takeMask));
        }
    }
}

/*!
 * \brief BitonicGatherThresholdTileKernel 的 LocalTensor 封装。
 *
 * 将 LocalTensor 参数转换为 __ubuf__ 指针后调用 Kernel。
 */
template <typename T, typename KeyT, typename IndexT>
__aicore__ inline void RunBitonicGatherThresholdTile(LocalTensor<T> inputValues, LocalTensor<KeyT> keys,
                                                     LocalTensor<T> outputValues, LocalTensor<IndexT> outputIndices,
                                                     uint32_t axisSize, uint32_t quota, uint64_t indexBase,
                                                     KeyT threshold)
{
    asc_vf_call<BitonicGatherThresholdTileKernel<T, KeyT, IndexT>>(
        dim3(BITONIC_SMALL_TOPK_SIZE), axisSize, quota, indexBase, threshold, (__ubuf__ T*)inputValues.GetPhyAddr(),
        (__ubuf__ KeyT*)keys.GetPhyAddr(), (__ubuf__ T*)outputValues.GetPhyAddr(),
        (__ubuf__ IndexT*)outputIndices.GetPhyAddr());
}

/*!
 * \brief 精确选择收尾 Kernel (SIMT)，封装 BitonicSmallFinalizeExactSelection。
 *
 * 单行处理：从 UB 读取 k 个候选，调用 BitonicSmallFinalizeExactSelection 排序后写回。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_SIZE) __aicore__
    void BitonicFinalizeExactSelectionKernel(uint32_t k, __ubuf__ T* values, __ubuf__ IndexT* indices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    bool valid = lane < k;
    T value = valid ? values[lane] : static_cast<T>(0);
    IndexT index = valid ? indices[lane] : BitonicSmallInvalidIndex<IndexT>();
    BitonicSmallFinalizeExactSelection<T, IndexT, IsLargest>(value, index, valid, k);
    if (valid) {
        values[lane] = value;
        indices[lane] = index;
    }
}

/*!
 * \brief BitonicFinalizeExactSelectionKernel 的 LocalTensor 封装。
 */
template <typename T, typename IndexT, bool IsLargest>
__aicore__ inline void RunBitonicFinalizeExactSelection(LocalTensor<T> values, LocalTensor<IndexT> indices, uint32_t k)
{
    asc_vf_call<BitonicFinalizeExactSelectionKernel<T, IndexT, IsLargest>>(
        dim3(BITONIC_SMALL_TOPK_SIZE), k, (__ubuf__ T*)values.GetPhyAddr(), (__ubuf__ IndexT*)indices.GetPhyAddr());
}
/*!
 * \brief 小源行重复候选重收集与终排 (SIMT warp 级路径)。
 *
 * 候选行存在重复值时，从原始输入 (axisLen <= 32) 两趟收集严格优于/等于阈值的
 * 元素，恢复 BITONIC 兼容的源序后按值网络终排。warp 全体 lane 均匀进入
 * (调用点条件 duplicateMask 为 ballot 结果，warp 一致)。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallSourceRowRegatherDuplicate(__ubuf__ T* inputValues, uint32_t row,
                                                                   uint32_t inputCount, uint32_t inputStride,
                                                                   uint32_t lane, uint32_t k, T threshold, T& value,
                                                                   IndexT& index)
{
    T gatheredValue = static_cast<T>(0);
    IndexT gatheredIndex = BitonicSmallInvalidIndex<IndexT>();
    uint32_t gatheredCount = 0U;
    uint32_t inputIndex = lane;
    bool inputValid = inputIndex < inputCount;
    T inputValue = inputValid ? inputValues[row * inputStride + inputIndex] : static_cast<T>(0);
    for (uint32_t pass = 0U; pass < 2U && gatheredCount < k; ++pass) {
        bool selected = inputValid && (pass == 0U ? BitonicSmallValueBetter<T>(inputValue, threshold, IsLargest) :
                                                    BitonicSmallValueEquivalent<T>(inputValue, threshold));
        uint32_t selectedMask = asc_ballot(static_cast<int32_t>(selected));
        uint32_t selectedCount = static_cast<uint32_t>(__popc(selectedMask));
        if (lane >= gatheredCount && lane < gatheredCount + selectedCount && lane < k) {
            uint32_t sourceLane = BitonicSmallNthSetBit(selectedMask, lane - gatheredCount);
            gatheredValue = asc_shfl(inputValue, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
            gatheredIndex = static_cast<IndexT>(sourceLane);
        }
        gatheredCount += selectedCount;
    }
    value = gatheredValue;
    index = gatheredIndex;
    BitonicSmallBitonicNetwork<T, IndexT, true>(value, index, IsLargest);
}

/*!
 * \brief 小源行收尾 Kernel (SIMT)，处理 axisLen <= 32 的特殊场景。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_THREADS) __aicore__
    void BitonicFinalizeSmallSourceRowsKernel(uint32_t rowCount, uint32_t inputCount, uint32_t k, uint32_t inputStride,
                                              uint32_t valueStride, uint32_t indexStride, __ubuf__ T* inputValues,
                                              __ubuf__ T* outputValues, __ubuf__ IndexT* outputIndices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t rowStep = static_cast<uint32_t>(blockDim.y);
    for (uint32_t row = static_cast<uint32_t>(threadIdx.y); row < rowCount; row += rowStep) {
        uint32_t valueOffset = row * valueStride + lane;
        uint32_t indexOffset = row * indexStride + lane;
        bool valid = lane < k;
        T value = valid ? outputValues[valueOffset] : static_cast<T>(0);
        IndexT index = valid ? outputIndices[indexOffset] : BitonicSmallInvalidIndex<IndexT>();
        T threshold = asc_shfl(value, static_cast<int32_t>(k - 1U), BITONIC_SMALL_TOPK_SIZE);

        T firstValue = asc_shfl(value, 0, BITONIC_SMALL_TOPK_SIZE);
        if (BitonicSmallValueEquivalent<T>(firstValue, threshold)) {
            BitonicSmallAllEqualBitonicNetwork<IndexT>(index);
        } else {
            T previous = asc_shfl(value, static_cast<int32_t>(lane == 0U ? 0U : lane - 1U), BITONIC_SMALL_TOPK_SIZE);
            bool duplicate = valid && lane > 0U && BitonicSmallValueEquivalent<T>(value, previous);
            uint32_t duplicateMask = asc_ballot(static_cast<int32_t>(duplicate));
            if (duplicateMask != 0U) {
                BitonicSmallSourceRowRegatherDuplicate<T, IndexT, IsLargest>(inputValues, row, inputCount, inputStride,
                                                                             lane, k, threshold, value, index);
            }
        }

        if (valid) {
            outputValues[valueOffset] = value;
            outputIndices[indexOffset] = index;
        }
    }
}

/*!
 * \brief BitonicFinalizeSmallSourceRowsKernel 的多行批量调度封装。
 */
template <typename T, typename IndexT, bool IsLargest>
__aicore__ inline void RunBitonicFinalizeSmallSourceRows(LocalTensor<T> inputValues, LocalTensor<T> outputValues,
                                                         LocalTensor<IndexT> outputIndices, uint32_t inputCount,
                                                         uint32_t k, uint32_t rowCount, uint32_t inputStride,
                                                         uint32_t valueStride, uint32_t indexStride)
{
    for (uint32_t rowStart = 0U; rowStart < rowCount; rowStart += BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH) {
        uint32_t rows = rowCount - rowStart;
        rows = rows > BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH ? BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH : rows;
        uint32_t warps = rows > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : rows;
        asc_vf_call<BitonicFinalizeSmallSourceRowsKernel<T, IndexT, IsLargest>>(
            dim3(BITONIC_SMALL_TOPK_SIZE, warps), rows, inputCount, k, inputStride, valueStride, indexStride,
            (__ubuf__ T*)inputValues[rowStart * inputStride].GetPhyAddr(),
            (__ubuf__ T*)outputValues[rowStart * valueStride].GetPhyAddr(),
            (__ubuf__ IndexT*)outputIndices[rowStart * indexStride].GetPhyAddr());
    }
}

/*!
 * \brief 打包路径的 grid-stride 行循环 (SEG_WIDTH 模板实例，SIMT warp 级)。
 */
template <typename T, typename IndexT, bool IsLargest, uint32_t SEG_WIDTH>
__simt_callee__ inline void BitonicFinalizePackedRowsLoop(__ubuf__ T* values, __ubuf__ IndexT* indices,
                                                          uint32_t rowCount, uint32_t k, uint32_t rowStep,
                                                          uint32_t valueStride, uint32_t indexStride)
{
    constexpr uint32_t SEGS_PER_WARP = BITONIC_SMALL_TOPK_SIZE / SEG_WIDTH;
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t seg = lane / SEG_WIDTH;
    uint32_t local = lane - seg * SEG_WIDTH;
    uint32_t rowGroupStep = rowStep * SEGS_PER_WARP;
    for (uint32_t rowBase = static_cast<uint32_t>(threadIdx.y) * SEGS_PER_WARP; rowBase < rowCount;
         rowBase += rowGroupStep) {
        uint32_t row = rowBase + seg;
        bool rowActive = row < rowCount;
        bool valid = rowActive && local < k;
        uint32_t valueOffset = row * valueStride + local;
        uint32_t indexOffset = row * indexStride + local;
        T value = valid ? values[valueOffset] : static_cast<T>(0);
        IndexT index = valid ? indices[indexOffset] : BitonicSmallInvalidIndex<IndexT>();
        bool write = BitonicSmallPackedFinalizeSelection<T, IndexT, IsLargest, SEG_WIDTH>(value, index, valid,
                                                                                          rowActive, k);
        if (write) {
            values[valueOffset] = value;
            indices[indexOffset] = index;
        }
    }
}

/*!
 * \brief 逐行路径的 grid-stride 行循环 (SIMT warp 级，每行一个 warp)。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallPerRowRowsLoop(__ubuf__ T* values, __ubuf__ IndexT* indices, uint32_t rowCount,
                                                       uint32_t k, uint32_t rowStep, uint32_t valueStride,
                                                       uint32_t indexStride)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t row = static_cast<uint32_t>(threadIdx.y); row < rowCount; row += rowStep) {
        uint32_t valueOffset = row * valueStride + lane;
        uint32_t indexOffset = row * indexStride + lane;
        bool valid = lane < k;
        T value = valid ? values[valueOffset] : static_cast<T>(0);
        IndexT index = valid ? indices[indexOffset] : BitonicSmallInvalidIndex<IndexT>();
        BitonicSmallFinalizeSelection<T, IndexT, IsLargest>(value, index, valid, k);
        if (valid) {
            values[valueOffset] = value;
            indices[indexOffset] = index;
        }
    }
}

/*!
 * \brief 行批量收尾 Kernel (SIMT)，封装 BitonicSmallFinalizeSelection。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_THREADS) __aicore__
    void BitonicFinalizeSelectionRowsKernel(uint32_t rowCount, uint32_t k, uint32_t valueStride, uint32_t indexStride,
                                            __ubuf__ T* values, __ubuf__ IndexT* indices)
{
    uint32_t rowStep = static_cast<uint32_t>(blockDim.y);
    if constexpr (std::is_same_v<T, int8_t>) {
        BitonicSmallPerRowRowsLoop<T, IndexT, IsLargest>(values, indices, rowCount, k, rowStep, valueStride,
                                                         indexStride);
        return;
    }
    if (k <= BITONIC_SMALL_TOPK_PACKED_MAX_K) {
        if (k <= 2U) {
            BitonicFinalizePackedRowsLoop<T, IndexT, IsLargest, 2U>(values, indices, rowCount, k, rowStep, valueStride,
                                                                    indexStride);
        } else if (k <= 4U) {
            BitonicFinalizePackedRowsLoop<T, IndexT, IsLargest, 4U>(values, indices, rowCount, k, rowStep, valueStride,
                                                                    indexStride);
        } else if (k <= 8U) {
            BitonicFinalizePackedRowsLoop<T, IndexT, IsLargest, 8U>(values, indices, rowCount, k, rowStep, valueStride,
                                                                    indexStride);
        } else if constexpr (sizeof(T) != 1U) {
            BitonicFinalizePackedRowsLoop<T, IndexT, IsLargest, 16U>(values, indices, rowCount, k, rowStep, valueStride,
                                                                     indexStride);
        } else {
            BitonicSmallPerRowRowsLoop<T, IndexT, IsLargest>(values, indices, rowCount, k, rowStep, valueStride,
                                                             indexStride);
        }
        return;
    }
    BitonicSmallPerRowRowsLoop<T, IndexT, IsLargest>(values, indices, rowCount, k, rowStep, valueStride, indexStride);
}

/*!
 * \brief BitonicFinalizeSelectionRowsKernel 的多行批量调度封装。
 */
template <typename T, typename IndexT, bool IsLargest>
__aicore__ inline void RunBitonicFinalizeSelectionRows(LocalTensor<T> values, LocalTensor<IndexT> indices, uint32_t k,
                                                       uint32_t rowCount, uint32_t valueStride, uint32_t indexStride)
{
    for (uint32_t rowStart = 0U; rowStart < rowCount; rowStart += BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH) {
        uint32_t rows = rowCount - rowStart;
        rows = rows > BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH ? BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH : rows;
        // rowsPerWarp 镜像 kernel 内 dispatch：int8/k>16/uint8 的 k∈(8,16] 走逐行路径为 1；
        // packed 路径每 warp 承载 32/SEG_WIDTH 行 (k<=2/4/8/16 -> SEG_WIDTH 2/4/8/16)。
        uint32_t rowsPerWarp = 1U;
        if constexpr (!std::is_same_v<T, int8_t>) {
            if (k <= BITONIC_SMALL_TOPK_PACKED_MAX_K) {
                if (k <= 2U) {
                    rowsPerWarp = BITONIC_SMALL_TOPK_SIZE / 2U;
                } else if (k <= 4U) {
                    rowsPerWarp = BITONIC_SMALL_TOPK_SIZE / 4U;
                } else if (k <= 8U) {
                    rowsPerWarp = BITONIC_SMALL_TOPK_SIZE / 8U;
                } else if constexpr (sizeof(T) != 1U) {
                    rowsPerWarp = BITONIC_SMALL_TOPK_SIZE / 16U;
                }
            }
        }
        uint32_t warpsNeeded = (rows + rowsPerWarp - 1U) / rowsPerWarp;
        uint32_t warps = warpsNeeded > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : warpsNeeded;
        asc_vf_call<BitonicFinalizeSelectionRowsKernel<T, IndexT, IsLargest>>(
            dim3(BITONIC_SMALL_TOPK_SIZE, warps), rows, k, valueStride, indexStride,
            (__ubuf__ T*)values[rowStart * valueStride].GetPhyAddr(),
            (__ubuf__ IndexT*)indices[rowStart * indexStride].GetPhyAddr());
    }
}

/*!
 * \brief 最终收尾主入口：按类型分发到 Reg 或 SIMT 路径。
 */
template <typename T, typename T_INDEX_TO, bool IS_LARGEST>
__aicore__ inline void RunBitonicSmallTopKFinalize(LocalTensor<T> values, LocalTensor<T_INDEX_TO> indices, uint32_t k,
                                                   uint32_t batchNum, uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (sizeof(T) != 1U && sizeof(T_INDEX_TO) == sizeof(uint32_t)) {
        __ubuf__ T* valueBase = (__ubuf__ T*)values.GetPhyAddr();
        __ubuf__ uint32_t* indexBase = (__ubuf__ uint32_t*)indices.GetPhyAddr();
        for (uint32_t rowStart = 0U; rowStart < batchNum; rowStart += BITONIC_SMALL_TOPK_MAX_ROWS) {
            uint32_t rows = batchNum - rowStart;
            rows = rows > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : rows;
            BitonicSmallRegFinalizeSelectionBatch<T, IS_LARGEST>(valueBase + rowStart * valueStride,
                                                                 indexBase + rowStart * indexStride, k, rows,
                                                                 valueStride, indexStride);
        }
    } else {
        RunBitonicFinalizeSelectionRows<T, T_INDEX_TO, IS_LARGEST>(values, indices, k, batchNum, valueStride,
                                                                   indexStride);
    }
}
/*!
 * \brief 最终收尾主入口 (独立 flag 数组版)：检测 flag 落盘调用方提供的独立
 * flagArr，删除 flag 借写/恢复/修复机制，每批 V_S 事件同步降为 1 组。
 *
 * \param[in] flagBase    调用方提供的 flag 数组 (至少 32 个 uint32 / 128B，32B 对齐；每批仅前
 *                        rowCount 项有效，批间复用)
 */
template <typename T, typename T_INDEX_TO, bool IS_LARGEST>
__aicore__ inline void RunBitonicSmallTopKFinalizeWithFlags(LocalTensor<T> values, LocalTensor<T_INDEX_TO> indices,
                                                            __ubuf__ uint32_t* flagBase, uint32_t k, uint32_t batchNum,
                                                            uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (sizeof(T) != 1U && sizeof(T_INDEX_TO) == sizeof(uint32_t)) {
        __ubuf__ T* valueBase = (__ubuf__ T*)values.GetPhyAddr();
        __ubuf__ uint32_t* indexBase = (__ubuf__ uint32_t*)indices.GetPhyAddr();
        for (uint32_t rowStart = 0U; rowStart < batchNum; rowStart += BITONIC_SMALL_TOPK_MAX_ROWS) {
            uint32_t rows = batchNum - rowStart;
            rows = rows > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : rows;
            BitonicSmallRegFinalizeSelectionFlagBatch<T, IS_LARGEST>(valueBase + rowStart * valueStride,
                                                                     indexBase + rowStart * indexStride, flagBase, k,
                                                                     rows, valueStride, indexStride);
        }
    } else {
        RunBitonicFinalizeSelectionRows<T, T_INDEX_TO, IS_LARGEST>(values, indices, k, batchNum, valueStride,
                                                                   indexStride);
    }
}
/*!
 * \brief 非尾轴场景调用AscendC::Topk高阶api的双调排序入口。
 */
template <typename SortT, bool IsLargest, bool IsBitonicSort>
__aicore__ inline void RunBitonicSmallTopKFinalizeNonLast(LocalTensor<SortT> values, LocalTensor<uint32_t> indices,
                                                          LocalTensor<SortT> sortInput, uint32_t axisLen, uint32_t k,
                                                          uint32_t batchNum, uint32_t axisRowElems,
                                                          uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (IsBitonicSort) {
        if constexpr (sizeof(SortT) == 1U) {
            if (axisLen <= BITONIC_SMALL_TOPK_SIZE) {
                RunBitonicFinalizeSmallSourceRows<SortT, uint32_t, IsLargest>(
                    sortInput, values, indices, axisLen, k, batchNum, axisRowElems, valueStride, indexStride);
                return;
            }
        }
        if constexpr (sizeof(SortT) != 1U) {
            __ubuf__ SortT* valueBase = (__ubuf__ SortT*)values.GetPhyAddr();
            __ubuf__ uint32_t* indexBase = (__ubuf__ uint32_t*)indices.GetPhyAddr();
            for (uint32_t rowStart = 0U; rowStart < batchNum; rowStart += BITONIC_SMALL_TOPK_MAX_ROWS) {
                uint32_t rows = batchNum - rowStart;
                rows = rows > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : rows;
                BitonicSmallRegFinalizeSelectionBatch<SortT, IsLargest>(valueBase + rowStart * valueStride,
                                                                        indexBase + rowStart * indexStride, k, rows,
                                                                        valueStride, indexStride);
            }
        } else {
            RunBitonicFinalizeSelectionRows<SortT, uint32_t, IsLargest>(values, indices, k, batchNum, valueStride,
                                                                        indexStride);
        }
    }
}

/*!
 * \brief 非尾轴场景调用Ascend::Sort高阶api的进行双调排序入口。
 */
template <typename SortT, bool IsLargest, bool IsBitonicSort>
__aicore__ inline void RunBitonicSmallMergeSortFinalizeNonLast(LocalTensor<SortT> values, LocalTensor<uint32_t> indices,
                                                               uint32_t k, uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (IsBitonicSort) {
        if constexpr (sizeof(SortT) > sizeof(uint32_t) || sizeof(SortT) == 1U) {
            RunBitonicFinalizeSelectionRows<SortT, uint32_t, IsLargest>(values, indices, k, 1U, valueStride,
                                                                        indexStride);
        } else {
            __ubuf__ SortT* valueAddr = (__ubuf__ SortT*)values.GetPhyAddr();
            __ubuf__ uint32_t* indexAddr = (__ubuf__ uint32_t*)indices.GetPhyAddr();
            BitonicSmallRegFinalizeSelectionBatch<SortT, IsLargest>(valueAddr, indexAddr, k, 1U, valueStride,
                                                                    indexStride);
        }
    }
}

} // namespace topkV2

#endif // TOP_K_SMALL_BITONIC_PACKED_SIMT_H
