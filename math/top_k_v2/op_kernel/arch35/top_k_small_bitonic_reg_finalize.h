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
 * \file top_k_small_bitonic_reg_finalize.h
 * \brief Reg/SIMD (向量寄存器) 路线的重复值检测与收尾终排。
 */
#ifndef TOP_K_SMALL_BITONIC_REG_FINALIZE_H
#define TOP_K_SMALL_BITONIC_REG_FINALIZE_H

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

namespace topkV2 {
using namespace AscendC;

/*!
 * \brief 32 位网络的分组比较掩码 (Reg/SIMD 路径，group+index 双级序)。
 *
 * 组号小者优先；组号相等时原始 index 小者优先，保证重复值稳定序。
 */
__simd_callee__ inline void BitonicSmallReg32GroupCompareMask(Reg::MaskReg& compareMask,
                                                              Reg::RegTensor<uint32_t>& lowIndex,
                                                              Reg::RegTensor<uint32_t>& highIndex,
                                                              Reg::RegTensor<uint32_t>& group,
                                                              Reg::RegTensor<uint32_t>& peerGroup,
                                                              Reg::MaskReg& lowLaneMask, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> lowGroup;
    Reg::RegTensor<uint32_t> highGroup;
    Reg::MaskReg groupEqualMask;
    Reg::MaskReg indexLessMask;
    Reg::Select<uint32_t>(lowGroup, group, peerGroup, lowLaneMask);
    Reg::Select<uint32_t>(highGroup, peerGroup, group, lowLaneMask);
    Reg::Compare<uint32_t, CMPMODE::LT>(compareMask, lowGroup, highGroup, activeMask);
    Reg::Compare<uint32_t, CMPMODE::EQ>(groupEqualMask, lowGroup, highGroup, activeMask);
    Reg::Compare<uint32_t, CMPMODE::LT>(indexLessMask, lowIndex, highIndex, activeMask);
    Reg::And(groupEqualMask, groupEqualMask, indexLessMask, activeMask);
    Reg::Or(compareMask, compareMask, groupEqualMask, activeMask);
}

/*!
 * \brief 32 位网络的交换决策与对端数据回写 (Reg/SIMD 路径)。
 *
 * 由 compareMask 推导 swapMask（低半有效才换 / 高半无效必换），
 * 按双调网络相位推导 directionMask，二者合成 takePeerMask 后完成
 * valueBits/key/index/group 的条件交换。
 */
template <bool CompareGroup, uint32_t Size>
__simd_callee__ inline void BitonicSmallReg32ApplySwapByMask(
    Reg::RegTensor<uint32_t>& valueBits, Reg::RegTensor<uint32_t>& key, Reg::RegTensor<uint32_t>& index,
    Reg::RegTensor<uint32_t>& group, Reg::RegTensor<uint32_t>& peerValueBits, Reg::RegTensor<uint32_t>& peerKey,
    Reg::RegTensor<uint32_t>& peerIndex, Reg::RegTensor<uint32_t>& peerGroup, Reg::MaskReg& compareMask,
    Reg::RegTensor<uint32_t>& lowIndex, Reg::RegTensor<uint32_t>& highIndex, Reg::RegTensor<uint32_t>& lane,
    Reg::MaskReg& activeMask)
{
    Reg::MaskReg lowValidMask;
    Reg::MaskReg highValidMask;
    Reg::MaskReg highInvalidMask;
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::NE>(lowValidMask, lowIndex, UINT32_MAX, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(highValidMask, highIndex, UINT32_MAX, activeMask);
    Reg::And(swapMask, compareMask, lowValidMask, activeMask);
    Reg::Not(highInvalidMask, highValidMask, activeMask);
    Reg::Or(swapMask, swapMask, highInvalidMask, activeMask);
    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::RegTensor<uint32_t> strideReg;
        Reg::RegTensor<uint32_t> peerLane;
        Reg::Duplicate(strideReg, Size);
        Reg::And(peerLane, lane, strideReg, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, peerLane, 0U, activeMask);
    }
    Reg::MaskReg takePeerMask;
    Reg::Xor(takePeerMask, swapMask, directionMask, activeMask);
    Reg::Not(takePeerMask, takePeerMask, activeMask);
    Reg::Select<uint32_t>(valueBits, peerValueBits, valueBits, takePeerMask);
    Reg::Select<uint32_t>(key, peerKey, key, takePeerMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
    if constexpr (CompareGroup) {
        Reg::Select<uint32_t>(group, peerGroup, group, takePeerMask);
    }
}

/*!
 * \brief 32 位双调网络的单次 compare-swap 阶段 (Reg/SIMD 路径)。
 *
 * 对 32 通道的向量寄存器，按给定 Stride 计算 peer lane (lane XOR stride)，
 * 通过 Reg::Gather 获取对端数据，将本地和对端分为 low/high 两部分，
 * 再根据比较结果和排序方向决定是否交换（比较掩码与交换回写分别由
 * GroupCompareMask / ApplySwapByMask 完成）。
 */
template <bool CompareGroup, uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallReg32SwapStage(Reg::RegTensor<uint32_t>& valueBits,
                                                       Reg::RegTensor<uint32_t>& key, Reg::RegTensor<uint32_t>& index,
                                                       Reg::RegTensor<uint32_t>& group, Reg::RegTensor<uint32_t>& lane,
                                                       Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> strideReg;
    Reg::RegTensor<uint32_t> peerLane;
    Reg::Duplicate(strideReg, Stride);
    Reg::Xor(peerLane, lane, strideReg, activeMask);
    Reg::RegTensor<uint32_t> peerValueBits;
    Reg::RegTensor<uint32_t> peerKey;
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::RegTensor<uint32_t> peerGroup;
    Reg::Gather(peerValueBits, valueBits, peerLane);
    Reg::Gather(peerKey, key, peerLane);
    Reg::Gather(peerIndex, index, peerLane);
    if constexpr (CompareGroup) {
        Reg::Gather(peerGroup, group, peerLane);
    }
    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, peerLane, activeMask);
    Reg::RegTensor<uint32_t> lowKey;
    Reg::RegTensor<uint32_t> highKey;
    Reg::RegTensor<uint32_t> lowIndex;
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<uint32_t>(lowKey, key, peerKey, lowLaneMask);
    Reg::Select<uint32_t>(highKey, peerKey, key, lowLaneMask);
    Reg::Select<uint32_t>(lowIndex, index, peerIndex, lowLaneMask);
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);
    Reg::MaskReg compareMask;
    if constexpr (CompareGroup) {
        BitonicSmallReg32GroupCompareMask(compareMask, lowIndex, highIndex, group, peerGroup, lowLaneMask, activeMask);
    } else {
        Reg::Compare<uint32_t, CMPMODE::GT>(compareMask, lowKey, highKey, activeMask);
    }
    BitonicSmallReg32ApplySwapByMask<CompareGroup, Size>(valueBits, key, index, group, peerValueBits, peerKey,
                                                         peerIndex, peerGroup, compareMask, lowIndex, highIndex, lane,
                                                         activeMask);
}

/*!
 * \brief 32 位双调排序网络完整序列 (Reg/SIMD 路径)。
 *
 * 标准双调排序网络对 32 个元素展开 15 个 SwapStage，按 (Stride, Size) 序列：
 * 先构建双调序列（size 递增），再做双调合并（stride 递减），最终全排序。
 */
template <bool CompareGroup>
__simd_callee__ inline void BitonicSmallReg32BitonicNetwork(Reg::RegTensor<uint32_t>& valueBits,
                                                            Reg::RegTensor<uint32_t>& key,
                                                            Reg::RegTensor<uint32_t>& index,
                                                            Reg::RegTensor<uint32_t>& group,
                                                            Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 2U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 4U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 4U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 4U, 8U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 8U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 8U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 8U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 4U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 16U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 8U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 4U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 32U>(valueBits, key, index, group, lane, activeMask);
}

/*!
 * \brief 双行打包 (64-lane) 公共 lane/mask 初始化 (Reg/SIMD 路径)。
 */
__simd_callee__ inline void BitonicSmallRegPairLaneMasks(uint32_t k, Reg::MaskReg& activeMask,
                                                         Reg::RegTensor<uint32_t>& lane,
                                                         Reg::RegTensor<uint32_t>& laneMod, Reg::MaskReg& validMask,
                                                         Reg::MaskReg& highHalfMask)
{
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::RegTensor<uint32_t> laneMaskReg;
    Reg::Duplicate(laneMaskReg, BITONIC_SMALL_TOPK_SIZE - 1U);
    Reg::And(laneMod, lane, laneMaskReg, activeMask);
    Reg::RegTensor<uint32_t> kReg;
    Reg::Duplicate(kReg, k);
    Reg::Compare<uint32_t, CMPMODE::LT>(validMask, laneMod, kReg, activeMask);
    Reg::Compares<uint32_t, CMPMODE::GE>(highHalfMask, lane, BITONIC_SMALL_TOPK_SIZE, activeMask);
}

/*!
 * \brief 双行打包行偏移 gather 索引计算 (Reg/SIMD 路径)。
 *
 * 行 B (高半) 元素位于基址 + rowOffset，行 A (低半) 取本行元素；无效 lane 的
 * 索引折叠到 0，保证 UB 访问安全。
 */
__simd_callee__ inline void BitonicSmallRegPairGatherIndex(uint32_t rowOffset, Reg::RegTensor<uint32_t>& laneMod,
                                                           Reg::RegTensor<uint32_t>& zeroIndex,
                                                           Reg::MaskReg& activeMask, Reg::MaskReg& validMask,
                                                           Reg::MaskReg& highHalfMask, Reg::RegTensor<uint32_t>& rawIdx,
                                                           Reg::RegTensor<uint32_t>& safeIdx)
{
    Reg::RegTensor<uint32_t> rowOffsetReg;
    Reg::Duplicate(rowOffsetReg, rowOffset);
    Reg::RegTensor<uint32_t> offsetIdx;
    Reg::Select<uint32_t>(offsetIdx, rowOffsetReg, zeroIndex, highHalfMask);
    Reg::Add(rawIdx, laneMod, offsetIdx, activeMask);
    Reg::Select<uint32_t>(safeIdx, rawIdx, zeroIndex, validMask);
}

/*!
 * \brief 双行打包 index 加载与借写修复 (Reg/SIMD 路径)。
 *
 * 按 safeIdx gather 行 A/B 的 index，无效 lane 填 UINT32_MAX；借写版
 * (RepairIndex0=true) 用编排层暂存值恢复行 A/B 的 index[0]。
 */
template <bool RepairIndex0>
__simd_callee__ inline void BitonicSmallRegLoadPairIndex(__ubuf__ uint32_t* indexAddrA,
                                                         Reg::RegTensor<uint32_t>& indexSafeIdx,
                                                         Reg::RegTensor<uint32_t>& index, uint32_t fixedIndex0A,
                                                         uint32_t fixedIndex0B, Reg::RegTensor<uint32_t>& lane,
                                                         Reg::MaskReg& activeMask, Reg::MaskReg& validMask)
{
    Reg::Gather(index, indexAddrA, indexSafeIdx, activeMask);
    Reg::RegTensor<uint32_t> invalidIndex;
    Reg::Duplicate(invalidIndex, UINT32_MAX);
    Reg::Select<uint32_t>(index, index, invalidIndex, validMask);
    if constexpr (RepairIndex0) {
        Reg::RegTensor<uint32_t> fixedIndexRegA;
        Reg::Duplicate(fixedIndexRegA, fixedIndex0A);
        Reg::MaskReg laneZeroMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
        Reg::Select<uint32_t>(index, fixedIndexRegA, index, laneZeroMask);
        Reg::RegTensor<uint32_t> fixedIndexRegB;
        Reg::Duplicate(fixedIndexRegB, fixedIndex0B);
        Reg::MaskReg lane32Mask;
        Reg::Compares<uint32_t, CMPMODE::EQ>(lane32Mask, lane, BITONIC_SMALL_TOPK_SIZE, activeMask);
        Reg::Select<uint32_t>(index, fixedIndexRegB, index, lane32Mask);
    }
}

/*!
 * \brief 双行打包阈值 lane 索引计算 (Reg/SIMD 路径)。
 *
 * 每行阈值位于该行第 k-1 个候选：lane 高半折叠 (halfBase) + k-1。
 */
__simd_callee__ inline void BitonicSmallRegPairThresholdIndex(Reg::RegTensor<uint32_t>& lane, uint32_t k,
                                                              Reg::MaskReg& activeMask,
                                                              Reg::RegTensor<uint32_t>& thresholdIndex)
{
    Reg::RegTensor<uint32_t> halfMaskReg;
    Reg::Duplicate(halfMaskReg, BITONIC_SMALL_TOPK_SIZE);
    Reg::RegTensor<uint32_t> halfBase;
    Reg::And(halfBase, lane, halfMaskReg, activeMask);
    Reg::RegTensor<uint32_t> kMinus1Reg;
    Reg::Duplicate(kMinus1Reg, k - 1U);
    Reg::Add(thresholdIndex, halfBase, kMinus1Reg, activeMask);
}

/*!
 * \brief strict/equal/invalid 三分类 group 标记 (Reg/SIMD 路径)。
 *
 * 0=严格优于阈值 / 1=等于阈值 / 2=无效 (超过 k)，供网络首趟按组排序。
 */
__simd_callee__ inline void BitonicSmallRegClassifyStrictGroup(Reg::MaskReg& strictMask, Reg::MaskReg& validMask,
                                                               Reg::MaskReg& activeMask,
                                                               Reg::RegTensor<uint32_t>& group)
{
    Reg::Duplicate(group, 1U);
    Reg::RegTensor<uint32_t> strictGroup;
    Reg::Duplicate(strictGroup, 0U);
    Reg::RegTensor<uint32_t> invalidGroup;
    Reg::Duplicate(invalidGroup, 2U);
    Reg::Select<uint32_t>(group, strictGroup, group, strictMask);
    Reg::Select<uint32_t>(group, group, invalidGroup, validMask);
}

/*!
 * \brief 16 位类型双行候选值加载与位模式展开 (Reg/SIMD 路径，双行打包)。
 *
 * 两行各加载 k 个 16 位候选展开为 uint32 位模式，按 highHalfMask 选择所属行，
 * 无效候选 lane 的寄存器值填 0。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegLoadPairValueB16(__ubuf__ T* valueAddrA, uint32_t rowOffsetValue, uint32_t k,
                                                            Reg::RegTensor<uint32_t>& laneMod, Reg::MaskReg& validMask,
                                                            Reg::MaskReg& highHalfMask,
                                                            Reg::RegTensor<uint32_t>& valueBits)
{
    Reg::RegTensor<uint32_t> valueBitsA;
    Reg::RegTensor<uint32_t> valueBitsB;
    BitonicSmallRegLoadB16Bits<T>(valueBitsA, valueAddrA, k);
    BitonicSmallRegLoadB16Bits<T>(valueBitsB, valueAddrA + rowOffsetValue, k);
    Reg::RegTensor<uint32_t> gatherA;
    Reg::Gather(gatherA, valueBitsA, laneMod);
    Reg::RegTensor<uint32_t> gatherB;
    Reg::Gather(gatherB, valueBitsB, laneMod);
    Reg::Select<uint32_t>(valueBits, gatherB, gatherA, highHalfMask);
    Reg::RegTensor<uint32_t> zeroValueBits;
    Reg::Duplicate(zeroValueBits, 0U);
    Reg::Select<uint32_t>(valueBits, valueBits, zeroValueBits, validMask);
}

/*!
 * \brief 16 位类型双行排序结果写回 (Reg/SIMD 路径，双行打包)。
 *
 * 行 A 取低半前 k、行 B 取高半前 k，收缩为 16 位存回各自 UB 行。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegStorePairValueB16(__ubuf__ T* valueAddrA, uint32_t rowOffsetValue,
                                                             uint32_t k, Reg::RegTensor<uint32_t>& valueBits,
                                                             Reg::RegTensor<uint32_t>& laneMod,
                                                             Reg::MaskReg& activeMask)
{
    Reg::MaskReg validMaskLow = Reg::UpdateMask<uint32_t>(k);
    Reg::RegTensor<uint32_t> outBitsA;
    Reg::Gather(outBitsA, valueBits, laneMod);
    BitonicSmallRegStoreB16Bits<T>(valueAddrA, outBitsA, k, validMaskLow);
    Reg::RegTensor<uint32_t> outOffsetB;
    Reg::Duplicate(outOffsetB, BITONIC_SMALL_TOPK_SIZE);
    Reg::RegTensor<uint32_t> outIndexB;
    Reg::Add(outIndexB, outOffsetB, laneMod, activeMask);
    Reg::RegTensor<uint32_t> outBitsB;
    Reg::Gather(outBitsB, valueBits, outIndexB);
    BitonicSmallRegStoreB16Bits<T>(valueAddrA + rowOffsetValue, outBitsB, k, validMaskLow);
}

/*!
 * \brief 16 位类型的全量收尾选择函数 (Reg/SIMD 路径，双行打包 64-lane)。
 */
template <typename T, bool IsLargest, bool RepairIndex0 = true>
__simd_callee__ inline void BitonicSmallRegFinalizeSelectionB16Pair(__ubuf__ T* valueAddrA,
                                                                    __ubuf__ uint32_t* indexAddrA, uint32_t k,
                                                                    uint32_t fixedIndex0A, uint32_t fixedIndex0B,
                                                                    uint32_t rowOffsetValue, uint32_t rowOffsetIndex)
{
    static_assert(sizeof(T) == 2U);
    uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE * 2U;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<uint32_t> laneMod;
    Reg::MaskReg validMask;
    Reg::MaskReg highHalfMask;
    BitonicSmallRegPairLaneMasks(k, activeMask, lane, laneMod, validMask, highHalfMask);
    Reg::RegTensor<uint32_t> zeroIndex;
    Reg::Duplicate(zeroIndex, 0U);
    Reg::RegTensor<uint32_t> indexRawIdx;
    Reg::RegTensor<uint32_t> indexSafeIdx;
    BitonicSmallRegPairGatherIndex(rowOffsetIndex, laneMod, zeroIndex, activeMask, validMask, highHalfMask, indexRawIdx,
                                   indexSafeIdx);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::RegTensor<uint32_t> valueBits;
    BitonicSmallRegLoadPairValueB16<T>(valueAddrA, rowOffsetValue, k, laneMod, validMask, highHalfMask, valueBits);
    Reg::RegTensor<uint32_t> index;
    BitonicSmallRegLoadPairIndex<RepairIndex0>(indexAddrA, indexSafeIdx, index, fixedIndex0A, fixedIndex0B, lane,
                                               activeMask, validMask);
    Reg::RegTensor<uint32_t> key;
    BitonicSmallRegBuildKey32<T, IsLargest>(key, valueBits, activeMask);
    Reg::RegTensor<uint32_t> thresholdIndex;
    BitonicSmallRegPairThresholdIndex(lane, k, activeMask, thresholdIndex);
    Reg::RegTensor<uint32_t> thresholdKey;
    Reg::Gather(thresholdKey, key, thresholdIndex);
    Reg::MaskReg strictMask;
    Reg::Compare<uint32_t, CMPMODE::GT>(strictMask, key, thresholdKey, activeMask);
    Reg::And(strictMask, strictMask, validMask, activeMask);
    Reg::RegTensor<uint32_t> group;
    BitonicSmallRegClassifyStrictGroup(strictMask, validMask, activeMask, group);
    BitonicSmallReg32BitonicNetwork<true>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32BitonicNetwork<false>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallRegStorePairValueB16<T>(valueAddrA, rowOffsetValue, k, valueBits, laneMod, activeMask);
    Reg::Scatter(indexAddrA, index, indexRawIdx, validMask);
}

/*!
 * \brief 64 位双调网络的单次 compare-swap 阶段 (Reg/SIMD 路径)。
 *
 * 64 位类型无法放入单个 32 位寄存器，需拆成 valueLow/valueHigh 两半处理。
 * 每次 Gather/Select 都要处理 low/high 两个寄存器。
 * 比较逻辑：先比 keyHigh（高位），相等再比 keyLow（低位），即 64 位比较的拆分实现。
 * 其余 swap 逻辑与 32 位版本 (BitonicSmallReg32SwapStage) 一致。
 */
template <bool CompareGroup, uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallReg64SwapStage(Reg::RegTensor<uint32_t>& valueLow,
                                                       Reg::RegTensor<uint32_t>& valueHigh,
                                                       Reg::RegTensor<uint32_t>& keyLow,
                                                       Reg::RegTensor<uint32_t>& keyHigh,
                                                       Reg::RegTensor<uint32_t>& index, Reg::RegTensor<uint32_t>& group,
                                                       Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> strideReg;
    Reg::RegTensor<uint32_t> peerLane;
    Reg::Duplicate(strideReg, Stride);
    Reg::Xor(peerLane, lane, strideReg, activeMask);
    Reg::RegTensor<uint32_t> peerValueLow;
    Reg::RegTensor<uint32_t> peerValueHigh;
    Reg::RegTensor<uint32_t> peerKeyLow;
    Reg::RegTensor<uint32_t> peerKeyHigh;
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::RegTensor<uint32_t> peerGroup;
    Reg::Gather(peerValueLow, valueLow, peerLane);
    Reg::Gather(peerValueHigh, valueHigh, peerLane);
    Reg::Gather(peerKeyLow, keyLow, peerLane);
    Reg::Gather(peerKeyHigh, keyHigh, peerLane);
    Reg::Gather(peerIndex, index, peerLane);
    if constexpr (CompareGroup) {
        Reg::Gather(peerGroup, group, peerLane);
    }
    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, peerLane, activeMask);
    Reg::RegTensor<uint32_t> lowKeyLow;
    Reg::RegTensor<uint32_t> lowKeyHigh;
    Reg::RegTensor<uint32_t> highKeyLow;
    Reg::RegTensor<uint32_t> highKeyHigh;
    Reg::RegTensor<uint32_t> lowIndex;
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<uint32_t>(lowKeyLow, keyLow, peerKeyLow, lowLaneMask);
    Reg::Select<uint32_t>(lowKeyHigh, keyHigh, peerKeyHigh, lowLaneMask);
    Reg::Select<uint32_t>(highKeyLow, peerKeyLow, keyLow, lowLaneMask);
    Reg::Select<uint32_t>(highKeyHigh, peerKeyHigh, keyHigh, lowLaneMask);
    Reg::Select<uint32_t>(lowIndex, index, peerIndex, lowLaneMask);
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);
    Reg::MaskReg compareMask;
    if constexpr (CompareGroup) {
        Reg::RegTensor<uint32_t> lowGroup;
        Reg::RegTensor<uint32_t> highGroup;
        Reg::MaskReg groupEqualMask;
        Reg::MaskReg indexLessMask;
        Reg::Select<uint32_t>(lowGroup, group, peerGroup, lowLaneMask);
        Reg::Select<uint32_t>(highGroup, peerGroup, group, lowLaneMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(compareMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(groupEqualMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(indexLessMask, lowIndex, highIndex, activeMask);
        Reg::And(groupEqualMask, groupEqualMask, indexLessMask, activeMask);
        Reg::Or(compareMask, compareMask, groupEqualMask, activeMask);
    } else {
        Reg::MaskReg highEqualMask;
        Reg::MaskReg lowGreaterMask;
        Reg::Compare<uint32_t, CMPMODE::GT>(compareMask, lowKeyHigh, highKeyHigh, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(highEqualMask, lowKeyHigh, highKeyHigh, activeMask);
        Reg::Compare<uint32_t, CMPMODE::GT>(lowGreaterMask, lowKeyLow, highKeyLow, activeMask);
        Reg::And(highEqualMask, highEqualMask, lowGreaterMask, activeMask);
        Reg::Or(compareMask, compareMask, highEqualMask, activeMask);
    }
    Reg::MaskReg lowValidMask;
    Reg::MaskReg highValidMask;
    Reg::MaskReg highInvalidMask;
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::NE>(lowValidMask, lowIndex, UINT32_MAX, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(highValidMask, highIndex, UINT32_MAX, activeMask);
    Reg::And(swapMask, compareMask, lowValidMask, activeMask);
    Reg::Not(highInvalidMask, highValidMask, activeMask);
    Reg::Or(swapMask, swapMask, highInvalidMask, activeMask);
    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::Duplicate(strideReg, Size);
        Reg::And(peerLane, lane, strideReg, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, peerLane, 0U, activeMask);
    }
    Reg::MaskReg takePeerMask;
    Reg::Xor(takePeerMask, swapMask, directionMask, activeMask);
    Reg::Not(takePeerMask, takePeerMask, activeMask);
    Reg::Select<uint32_t>(valueLow, peerValueLow, valueLow, takePeerMask);
    Reg::Select<uint32_t>(valueHigh, peerValueHigh, valueHigh, takePeerMask);
    Reg::Select<uint32_t>(keyLow, peerKeyLow, keyLow, takePeerMask);
    Reg::Select<uint32_t>(keyHigh, peerKeyHigh, keyHigh, takePeerMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
    if constexpr (CompareGroup) {
        Reg::Select<uint32_t>(group, peerGroup, group, takePeerMask);
    }
}

/*!
 * \brief 64 位双调排序网络完整序列 (Reg/SIMD 路径)。
 *
 * 与 BitonicSmallReg32BitonicNetwork 结构相同，展开 15 个 SwapStage。
 * 区别在于每个 stage 处理 valueLow/valueHigh 和 keyLow/keyHigh 双寄存器。
 */
template <bool CompareGroup>
__simd_callee__ inline void BitonicSmallReg64BitonicNetwork(
    Reg::RegTensor<uint32_t>& valueLow, Reg::RegTensor<uint32_t>& valueHigh, Reg::RegTensor<uint32_t>& keyLow,
    Reg::RegTensor<uint32_t>& keyHigh, Reg::RegTensor<uint32_t>& index, Reg::RegTensor<uint32_t>& group,
    Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 2U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 4U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 4U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 4U, 8U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 8U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 8U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 8U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 4U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 16U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                       activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 8U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 4U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
}

/*!
 * \brief 64 位候选值加载与 low/high 拆分 (Reg/SIMD 路径)。
 *
 * 交错存储的 64 位候选按 rawMask 加载为 2k 个 uint32，DeInterleave
 * 拆为 low/high 两寄存器，无效候选 lane 的寄存器值填 0。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegLoadB64Value(__ubuf__ T* valueAddr, Reg::RegTensor<uint32_t>& rawValue,
                                                        Reg::RegTensor<uint32_t>& valueLow,
                                                        Reg::RegTensor<uint32_t>& valueHigh, Reg::MaskReg& rawMask,
                                                        Reg::MaskReg& validMask)
{
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(rawValue, (__ubuf__ uint32_t*)valueAddr, 1U, rawMask);
    Reg::DeInterleave(valueLow, valueHigh, rawValue, rawValue);
    Reg::RegTensor<uint32_t> zeroValue;
    Reg::Duplicate(zeroValue, 0U);
    Reg::Select<uint32_t>(valueLow, valueLow, zeroValue, validMask);
    Reg::Select<uint32_t>(valueHigh, valueHigh, zeroValue, validMask);
}

/*!
 * \brief 64 位候选 index 加载、无效填充与借写修复 (Reg/SIMD 路径)。
 *
 * 按 validMask 加载候选 index，无效 lane 填 UINT32_MAX；借写版
 * (RepairIndex0=true) 用编排层暂存值恢复 index[0]。
 */
template <bool RepairIndex0>
__simd_callee__ inline void BitonicSmallRegLoadB64Index(__ubuf__ uint32_t* indexAddr, Reg::RegTensor<uint32_t>& index,
                                                        uint32_t fixedIndex0, Reg::MaskReg& validMask)
{
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(index, indexAddr, 1U, validMask);
    Reg::RegTensor<uint32_t> invalidIndex;
    Reg::Duplicate(invalidIndex, UINT32_MAX);
    Reg::Select<uint32_t>(index, index, invalidIndex, validMask);
    if constexpr (RepairIndex0) {
        Reg::RegTensor<uint32_t> fixedIndexReg;
        Reg::MaskReg laneZeroMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
        Reg::Duplicate(fixedIndexReg, fixedIndex0);
        Reg::Select<uint32_t>(index, fixedIndexReg, index, laneZeroMask);
    }
}

/*!
 * \brief 64 位整数 key 构建 (Reg/SIMD 路径)。
 *
 * 有符号类型高位异或 0x80000000，low 直接复制；求最小值时两半整体
 * 取反，统一为最大值语义的降序 key。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegBuildB64Key(Reg::RegTensor<uint32_t>& valueLow,
                                                       Reg::RegTensor<uint32_t>& valueHigh,
                                                       Reg::RegTensor<uint32_t>& keyLow,
                                                       Reg::RegTensor<uint32_t>& keyHigh,
                                                       Reg::RegTensor<uint32_t>& rawValue, Reg::MaskReg& activeMask)
{
    Reg::Duplicate(keyHigh, std::is_signed_v<T> ? 0x80000000U : 0U);
    Reg::Xor(keyHigh, valueHigh, keyHigh, activeMask);
    Reg::Or(keyLow, valueLow, valueLow, activeMask);
    if constexpr (!IsLargest) {
        Reg::Duplicate(rawValue, UINT32_MAX);
        Reg::Xor(keyHigh, keyHigh, rawValue, activeMask);
        Reg::Xor(keyLow, keyLow, rawValue, activeMask);
    }
}

/*!
 * \brief 64 位阈值严格优于掩码 (Reg/SIMD 路径)。
 *
 * 阈值取第 k-1 个候选 key（双寄存器）：keyHigh 大于阈值高位、或高位
 * 相等且 keyLow 大于阈值低位，即严格优于阈值；再与有效掩码相与。
 */
__simd_callee__ inline void BitonicSmallRegB64StrictThresholdMask(Reg::RegTensor<uint32_t>& keyLow,
                                                                  Reg::RegTensor<uint32_t>& keyHigh, uint32_t k,
                                                                  Reg::MaskReg& activeMask, Reg::MaskReg& validMask,
                                                                  Reg::MaskReg& strictMask)
{
    Reg::RegTensor<uint32_t> thresholdIndex;
    Reg::RegTensor<uint32_t> thresholdLow;
    Reg::RegTensor<uint32_t> thresholdHigh;
    Reg::Duplicate(thresholdIndex, k - 1U);
    Reg::Gather(thresholdLow, keyLow, thresholdIndex);
    Reg::Gather(thresholdHigh, keyHigh, thresholdIndex);
    Reg::MaskReg highGreaterMask;
    Reg::MaskReg highEqualMask;
    Reg::MaskReg lowGreaterMask;
    Reg::Compare<uint32_t, CMPMODE::GT>(highGreaterMask, keyHigh, thresholdHigh, activeMask);
    Reg::Compare<uint32_t, CMPMODE::EQ>(highEqualMask, keyHigh, thresholdHigh, activeMask);
    Reg::Compare<uint32_t, CMPMODE::GT>(lowGreaterMask, keyLow, thresholdLow, activeMask);
    Reg::And(strictMask, highEqualMask, lowGreaterMask, activeMask);
    Reg::Or(strictMask, strictMask, highGreaterMask, activeMask);
    Reg::And(strictMask, strictMask, validMask, activeMask);
}

/*!
 * \brief 64 位排序结果交错打包写回 (Reg/SIMD 路径)。
 */
__simd_callee__ inline void BitonicSmallRegStoreB64Value(__ubuf__ uint32_t* valueAddr,
                                                         Reg::RegTensor<uint32_t>& rawValue,
                                                         Reg::RegTensor<uint32_t>& valueLow,
                                                         Reg::RegTensor<uint32_t>& valueHigh, Reg::MaskReg& rawMask)
{
    Reg::RegTensor<uint32_t> unusedValue;
    Reg::Interleave<uint32_t>(rawValue, unusedValue, valueLow, valueHigh);
    Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(valueAddr, rawValue, 1U, rawMask);
}

/*!
 * \brief 64 位类型的全量收尾选择函数 (Reg/SIMD 路径)。
 */
template <typename T, bool IsLargest, bool RepairIndex0 = true>
__simd_callee__ inline void BitonicSmallRegFinalizeSelectionB64(__ubuf__ T* valueAddr, __ubuf__ uint32_t* indexAddr,
                                                                uint32_t k, uint32_t fixedIndex0)
{
    static_assert(std::is_integral_v<T> && sizeof(T) == sizeof(uint64_t));
    uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
    uint32_t validCount = k;
    uint32_t rawCount = k * 2U;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
    Reg::MaskReg rawMask = Reg::UpdateMask<uint32_t>(rawCount);
    Reg::RegTensor<uint32_t> rawValue;
    Reg::RegTensor<uint32_t> valueLow;
    Reg::RegTensor<uint32_t> valueHigh;
    Reg::RegTensor<uint32_t> keyLow;
    Reg::RegTensor<uint32_t> keyHigh;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<uint32_t> lane;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    BitonicSmallRegLoadB64Value<T>(valueAddr, rawValue, valueLow, valueHigh, rawMask, validMask);
    BitonicSmallRegLoadB64Index<RepairIndex0>(indexAddr, index, fixedIndex0, validMask);
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    BitonicSmallRegBuildB64Key<T, IsLargest>(valueLow, valueHigh, keyLow, keyHigh, rawValue, activeMask);
    Reg::MaskReg strictMask;
    BitonicSmallRegB64StrictThresholdMask(keyLow, keyHigh, k, activeMask, validMask, strictMask);
    Reg::RegTensor<uint32_t> group;
    BitonicSmallRegClassifyStrictGroup(strictMask, validMask, activeMask, group);
    BitonicSmallReg64BitonicNetwork<true>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane, activeMask);
    BitonicSmallReg64BitonicNetwork<false>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane, activeMask);
    BitonicSmallRegStoreB64Value((__ubuf__ uint32_t*)valueAddr, rawValue, valueLow, valueHigh, rawMask);
    Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(indexAddr, index, 1U, validMask);
}

/*!
 * \brief 判断 lhs 是否"优于" rhs (Reg/SIMD 路径)。
 *
 * IsLargest=true 时求最大值，lhs > rhs 则 lhs 优于 rhs；
 * IsLargest=false 时求最小值，lhs < rhs 则 lhs 优于 rhs。
 * 浮点类型有 NaN 特殊语义：IsLargest 时 NaN 视为最大，否则最小。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegValueBetter(Reg::MaskReg& betterMask, Reg::RegTensor<T>& lhs,
                                                       Reg::RegTensor<T>& rhs, Reg::MaskReg& activeMask)
{
    if constexpr (IsBitonicFloatType<T>) {
        Reg::MaskReg lhsNanMask;
        Reg::MaskReg rhsNanMask;
        Reg::MaskReg notNanMask;
        Reg::MaskReg strictMask;
        Reg::Compare<T, CMPMODE::NE>(lhsNanMask, lhs, lhs, activeMask);
        Reg::Compare<T, CMPMODE::NE>(rhsNanMask, rhs, rhs, activeMask);
        if constexpr (IsLargest) {
            Reg::Compare<T, CMPMODE::GT>(strictMask, lhs, rhs, activeMask);
            Reg::Not(notNanMask, rhsNanMask, activeMask);
            Reg::And(lhsNanMask, lhsNanMask, notNanMask, activeMask);
        } else {
            Reg::Compare<T, CMPMODE::LT>(strictMask, lhs, rhs, activeMask);
            Reg::Not(notNanMask, lhsNanMask, activeMask);
            Reg::And(lhsNanMask, rhsNanMask, notNanMask, activeMask);
        }
        Reg::Or(betterMask, strictMask, lhsNanMask, activeMask);
    } else if constexpr (IsLargest) {
        Reg::Compare<T, CMPMODE::GT>(betterMask, lhs, rhs, activeMask);
    } else {
        Reg::Compare<T, CMPMODE::LT>(betterMask, lhs, rhs, activeMask);
    }
}

/*!
 * \brief 判断 lhs 和 rhs 是否等价 (Reg/SIMD 路径)。
 *
 * 整数类型直接比较相等；浮点类型额外将 NaN==NaN 视为等价（通过
 * "两者都为 NaN" 的掩码与相等掩码取或实现）。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegValueEquivalent(Reg::MaskReg& equivalentMask, Reg::RegTensor<T>& lhs,
                                                           Reg::RegTensor<T>& rhs, Reg::MaskReg& activeMask)
{
    Reg::Compare<T, CMPMODE::EQ>(equivalentMask, lhs, rhs, activeMask);
    if constexpr (IsBitonicFloatType<T>) {
        Reg::MaskReg lhsNanMask;
        Reg::MaskReg rhsNanMask;
        Reg::Compare<T, CMPMODE::NE>(lhsNanMask, lhs, lhs, activeMask);
        Reg::Compare<T, CMPMODE::NE>(rhsNanMask, rhs, rhs, activeMask);
        Reg::And(lhsNanMask, lhsNanMask, rhsNanMask, activeMask);
        Reg::Or(equivalentMask, equivalentMask, lhsNanMask, activeMask);
    }
}
/*!
 * \brief 计算前驱 lane 序号 (Reg/SIMD 路径，无重复值早退检测辅助)。
 *
 * prevLane = (lane - 1) & 31：lane >= 1 时为真实前驱；lane0 环绕到 31，
 * 保证 Gather 索引始终合法，lane0 的重复判定由后续 lane > 0 掩码屏蔽。
 */
__simd_callee__ inline void BitonicSmallRegPrevLane(Reg::RegTensor<uint32_t>& prevLane, Reg::RegTensor<uint32_t>& lane,
                                                    Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> oneReg;
    Reg::RegTensor<uint32_t> laneMaskReg;
    Reg::Duplicate(oneReg, 1U);
    Reg::Duplicate(laneMaskReg, BITONIC_SMALL_TOPK_SIZE - 1U);
    Reg::Sub(prevLane, lane, oneReg, activeMask);
    Reg::And(prevLane, prevLane, laneMaskReg, activeMask);
}

/*!
 * \brief 计算前驱 lane 序号 (Reg/SIMD 路径，双行打包 64-lane 半区局部化)。
 */
__simd_callee__ inline void BitonicSmallRegPrevLanePair(Reg::RegTensor<uint32_t>& prevLane,
                                                        Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> oneReg;
    Reg::RegTensor<uint32_t> laneMaskReg;
    Reg::RegTensor<uint32_t> halfMaskReg;
    Reg::RegTensor<uint32_t> prevMod;
    Reg::RegTensor<uint32_t> halfBase;
    Reg::Duplicate(oneReg, 1U);
    Reg::Duplicate(laneMaskReg, BITONIC_SMALL_TOPK_SIZE - 1U);
    Reg::Duplicate(halfMaskReg, BITONIC_SMALL_TOPK_SIZE);
    Reg::Sub(prevMod, lane, oneReg, activeMask);
    Reg::And(prevMod, prevMod, laneMaskReg, activeMask);
    Reg::And(halfBase, lane, halfMaskReg, activeMask);
    Reg::Add(prevLane, prevMod, halfBase, activeMask);
}

/*!
 * \brief 将寄存器 lane0 的值单元素写入 indexAddr[0] (Reg/SIMD 路径)。
 */
__simd_callee__ inline void BitonicSmallRegStoreIndex0(__ubuf__ uint32_t* indexAddr, Reg::RegTensor<uint32_t>& valueReg)
{
    Reg::MaskReg laneZeroMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
    Reg::StoreAlign<uint32_t>(indexAddr, valueReg, laneZeroMask);
}

/*!
 * \brief 无重复值检测公共收尾 (Reg/SIMD 路径，向量检测块行内核尾部)。
 */
__simd_callee__ inline void BitonicSmallRegDetectDuplicateTail(Reg::RegTensor<uint32_t>& duplicateAny,
                                                               Reg::MaskReg& equivalentMask, Reg::MaskReg& activeMask,
                                                               Reg::MaskReg& validMask, Reg::MaskReg& laneGtZeroMask,
                                                               Reg::RegTensor<uint32_t>& oneValue)
{
    Reg::MaskReg duplicateMask;
    Reg::And(duplicateMask, equivalentMask, laneGtZeroMask, activeMask);
    Reg::And(duplicateMask, duplicateMask, validMask, activeMask);
    Reg::Reduce<Reg::ReduceType::MAX>(duplicateAny, oneValue, duplicateMask);
}

/*!
 * \brief 逐位全等 break 检测公共收尾 (Reg/SIMD 路径，全等快路径)。
 */
__simd_callee__ inline void BitonicSmallRegDetectAllEqBreakTail(Reg::RegTensor<uint32_t>& breakAny,
                                                                Reg::MaskReg& bitwiseNeqMask, Reg::MaskReg& activeMask,
                                                                Reg::MaskReg& validMask, Reg::MaskReg& laneGtZeroMask,
                                                                Reg::RegTensor<uint32_t>& twoValue)
{
    Reg::MaskReg breakMask;
    Reg::And(breakMask, bitwiseNeqMask, laneGtZeroMask, activeMask);
    Reg::And(breakMask, breakMask, validMask, activeMask);
    Reg::Reduce<Reg::ReduceType::MAX>(breakAny, twoValue, breakMask);
}

/*!
 * \brief 行 flag 2 bit 合成 (Reg/SIMD 路径, 全等快路径)。
 *
 * 状态位定义见 BITONIC_SMALL_TOPK_ROW_FLAG_* 常量。
 */
__simd_callee__ inline void BitonicSmallRegCombineRowFlag(Reg::RegTensor<uint32_t>& rowFlag,
                                                          Reg::RegTensor<uint32_t>& duplicateAny,
                                                          Reg::RegTensor<uint32_t>& breakAny,
                                                          Reg::RegTensor<uint32_t>& zeroIndex, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> duplicateBcast;
    Reg::RegTensor<uint32_t> breakBcast;
    Reg::Gather(duplicateBcast, duplicateAny, zeroIndex);
    Reg::Gather(breakBcast, breakAny, zeroIndex);
    Reg::Or(rowFlag, duplicateBcast, breakBcast, activeMask);
}
/*!
 * \brief 无重复值检测行内核 (Reg/SIMD 路径，16 位类型)。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB16Row(
    __ubuf__ T* valueAddr, Reg::RegTensor<uint32_t>& duplicateAny, Reg::RegTensor<uint32_t>& breakAny, uint32_t k,
    Reg::MaskReg& activeMask, Reg::MaskReg& validMask, Reg::RegTensor<uint32_t>& lane,
    Reg::RegTensor<uint32_t>& prevLane, Reg::MaskReg& laneGtZeroMask, Reg::RegTensor<uint32_t>& oneValue,
    Reg::RegTensor<uint32_t>& twoValue, Reg::RegTensor<uint32_t>& zeroValueBits)
{
    Reg::RegTensor<uint32_t> valueBits;
    Reg::RegTensor<uint32_t> key;
    Reg::RegTensor<uint32_t> prevKey;
    Reg::MaskReg equivalentMask;
    BitonicSmallRegLoadB16Bits<T>(valueBits, valueAddr, k);
    Reg::Select<uint32_t>(valueBits, valueBits, zeroValueBits, validMask);
    BitonicSmallRegBuildKey32<T, IsLargest>(key, valueBits, activeMask);
    Reg::Gather(prevKey, key, prevLane);
    Reg::Compare<uint32_t, CMPMODE::EQ>(equivalentMask, key, prevKey, activeMask);
    BitonicSmallRegDetectDuplicateTail(duplicateAny, equivalentMask, activeMask, validMask, laneGtZeroMask, oneValue);
    Reg::RegTensor<uint32_t> prevBits;
    Reg::RegTensor<uint32_t> bitsXor;
    Reg::MaskReg bitsNeqMask;
    Reg::Gather(prevBits, valueBits, prevLane);
    Reg::Xor(bitsXor, valueBits, prevBits, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(bitsNeqMask, bitsXor, 0U, activeMask);
    BitonicSmallRegDetectAllEqBreakTail(breakAny, bitsNeqMask, activeMask, validMask, laneGtZeroMask, twoValue);
}

/*!
 * \brief 无重复值检测行内核 (Reg/SIMD 路径，64 位类型)。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB64Row(
    __ubuf__ T* valueAddr, Reg::RegTensor<uint32_t>& duplicateAny, Reg::RegTensor<uint32_t>& breakAny,
    Reg::MaskReg& activeMask, Reg::MaskReg& validMask, Reg::MaskReg& rawMask, Reg::RegTensor<uint32_t>& lane,
    Reg::RegTensor<uint32_t>& prevLane, Reg::MaskReg& laneGtZeroMask, Reg::RegTensor<uint32_t>& oneValue,
    Reg::RegTensor<uint32_t>& twoValue, Reg::RegTensor<uint32_t>& zeroValue, Reg::RegTensor<uint32_t>& keyHighXor,
    Reg::RegTensor<uint32_t>& allBitsValue)
{
    Reg::RegTensor<uint32_t> rawValue;
    Reg::RegTensor<uint32_t> valueLow;
    Reg::RegTensor<uint32_t> valueHigh;
    Reg::RegTensor<uint32_t> keyLow;
    Reg::RegTensor<uint32_t> keyHigh;
    Reg::RegTensor<uint32_t> prevKeyLow;
    Reg::RegTensor<uint32_t> prevKeyHigh;
    Reg::MaskReg adjacentLowEqualMask;
    Reg::MaskReg adjacentHighEqualMask;
    Reg::MaskReg equivalentMask;
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(rawValue, (__ubuf__ uint32_t*)valueAddr, 1U, rawMask);
    Reg::DeInterleave(valueLow, valueHigh, rawValue, rawValue);
    Reg::Select<uint32_t>(valueLow, valueLow, zeroValue, validMask);
    Reg::Select<uint32_t>(valueHigh, valueHigh, zeroValue, validMask);
    Reg::Xor(keyHigh, valueHigh, keyHighXor, activeMask);
    Reg::Or(keyLow, valueLow, valueLow, activeMask);
    if constexpr (!IsLargest) {
        Reg::Xor(keyHigh, keyHigh, allBitsValue, activeMask);
        Reg::Xor(keyLow, keyLow, allBitsValue, activeMask);
    }
    Reg::Gather(prevKeyLow, keyLow, prevLane);
    Reg::Gather(prevKeyHigh, keyHigh, prevLane);
    Reg::Compare<uint32_t, CMPMODE::EQ>(adjacentLowEqualMask, keyLow, prevKeyLow, activeMask);
    Reg::Compare<uint32_t, CMPMODE::EQ>(adjacentHighEqualMask, keyHigh, prevKeyHigh, activeMask);
    Reg::And(equivalentMask, adjacentLowEqualMask, adjacentHighEqualMask, activeMask);
    BitonicSmallRegDetectDuplicateTail(duplicateAny, equivalentMask, activeMask, validMask, laneGtZeroMask, oneValue);
    Reg::MaskReg bitsNeqMask;
    Reg::Not(bitsNeqMask, equivalentMask, activeMask);
    BitonicSmallRegDetectAllEqBreakTail(breakAny, bitsNeqMask, activeMask, validMask, laneGtZeroMask, twoValue);
}

/*!
 * \brief 无重复值检测行对内核 (Reg/SIMD 路径，双行打包 64-lane，通用 4 字节)。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateCommonPairRow(
    __ubuf__ T* valueAddrA, Reg::RegTensor<uint32_t>& duplicateAnyA, Reg::RegTensor<uint32_t>& duplicateAnyB,
    Reg::RegTensor<uint32_t>& breakAnyA, Reg::RegTensor<uint32_t>& breakAnyB, Reg::MaskReg& activeMask,
    Reg::MaskReg& validMask, Reg::MaskReg& lowHalfMask, Reg::MaskReg& highHalfMask, Reg::RegTensor<uint32_t>& prevLane,
    Reg::MaskReg& laneGtZeroMask, Reg::RegTensor<uint32_t>& oneValue, Reg::RegTensor<uint32_t>& twoValue,
    Reg::RegTensor<T>& zeroValue, Reg::RegTensor<uint32_t>& valueSafeIdx)
{
    Reg::RegTensor<T> value;
    Reg::RegTensor<T> prevValue;
    Reg::MaskReg equivalentMask;
    Reg::Gather(value, valueAddrA, valueSafeIdx, activeMask);
    Reg::Select<T>(value, value, zeroValue, validMask);
    Reg::Gather(prevValue, value, prevLane);
    BitonicSmallRegValueEquivalent<T>(equivalentMask, value, prevValue, activeMask);

    Reg::MaskReg duplicateMaskA;
    Reg::And(duplicateMaskA, equivalentMask, laneGtZeroMask, activeMask);
    Reg::And(duplicateMaskA, duplicateMaskA, validMask, activeMask);
    Reg::And(duplicateMaskA, duplicateMaskA, lowHalfMask, activeMask);
    Reg::Reduce<Reg::ReduceType::MAX>(duplicateAnyA, oneValue, duplicateMaskA);

    Reg::MaskReg duplicateMaskB;
    Reg::And(duplicateMaskB, equivalentMask, laneGtZeroMask, activeMask);
    Reg::And(duplicateMaskB, duplicateMaskB, validMask, activeMask);
    Reg::And(duplicateMaskB, duplicateMaskB, highHalfMask, activeMask);
    Reg::Reduce<Reg::ReduceType::MAX>(duplicateAnyB, oneValue, duplicateMaskB);

    Reg::MaskReg bitsNeqMask;
    if constexpr (IsBitonicFloatType<T>) {
        Reg::RegTensor<uint32_t> prevRawBits;
        Reg::RegTensor<uint32_t> bitsXor;
        Reg::Gather(prevRawBits, (Reg::RegTensor<uint32_t>&)value, prevLane);
        Reg::Xor(bitsXor, (Reg::RegTensor<uint32_t>&)value, prevRawBits, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(bitsNeqMask, bitsXor, 0U, activeMask);
    } else {
        Reg::Not(bitsNeqMask, equivalentMask, activeMask);
    }
    Reg::MaskReg breakMaskA;
    Reg::And(breakMaskA, bitsNeqMask, laneGtZeroMask, activeMask);
    Reg::And(breakMaskA, breakMaskA, validMask, activeMask);
    Reg::And(breakMaskA, breakMaskA, lowHalfMask, activeMask);
    Reg::Reduce<Reg::ReduceType::MAX>(breakAnyA, twoValue, breakMaskA);
    Reg::MaskReg breakMaskB;
    Reg::And(breakMaskB, bitsNeqMask, laneGtZeroMask, activeMask);
    Reg::And(breakMaskB, breakMaskB, validMask, activeMask);
    Reg::And(breakMaskB, breakMaskB, highHalfMask, activeMask);
    Reg::Reduce<Reg::ReduceType::MAX>(breakAnyB, twoValue, breakMaskB);
}

/*!
 * \brief 检测 flag 的逐行落盘/合并 (Reg/SIMD 路径，批检测行尾分发)。
 */
template <bool ToFlagArray>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateEmit(
    Reg::RegTensor<uint32_t>& flagVec, Reg::RegTensor<uint32_t>& rowFlag, Reg::RegTensor<uint32_t>& lane,
    uint16_t rowIndex, Reg::MaskReg& activeMask, __ubuf__ uint32_t* flagStoreBase, uint32_t flagStoreStride)
{
    if constexpr (ToFlagArray) {
        Reg::RegTensor<uint32_t> rowIndexReg;
        Reg::Duplicate(rowIndexReg, static_cast<uint32_t>(rowIndex));
        Reg::MaskReg laneEqRowMask;
        Reg::Compare<uint32_t, CMPMODE::EQ>(laneEqRowMask, lane, rowIndexReg, activeMask);
        Reg::Select<uint32_t>(flagVec, rowFlag, flagVec, laneEqRowMask);
    } else {
        BitonicSmallRegStoreIndex0(flagStoreBase + rowIndex * flagStoreStride, rowFlag);
    }
}

/*!
 * \brief 无重复值检测双行打包路径的行对发射 (Reg/SIMD，B32 pair 循环迭代体)。
 */
template <typename T, bool ToFlagArray>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB32PairEmit(
    Reg::RegTensor<uint32_t>& flagVec, Reg::RegTensor<uint32_t>& lane, __ubuf__ T* valueAddrPair, uint16_t rowIndexA,
    uint16_t rowIndexB, Reg::MaskReg& activeMask, Reg::MaskReg& validMask, Reg::MaskReg& lowHalfMask,
    Reg::MaskReg& highHalfMask, Reg::RegTensor<uint32_t>& prevLane, Reg::MaskReg& laneGtZeroMask,
    Reg::RegTensor<uint32_t>& oneValue, Reg::RegTensor<uint32_t>& twoValue, Reg::RegTensor<T>& zeroValue,
    Reg::RegTensor<uint32_t>& valueSafeIdx, Reg::RegTensor<uint32_t>& zeroIndex, __ubuf__ uint32_t* flagStoreBase,
    uint32_t flagStoreStride)
{
    Reg::RegTensor<uint32_t> duplicateAnyA;
    Reg::RegTensor<uint32_t> duplicateAnyB;
    Reg::RegTensor<uint32_t> breakAnyA;
    Reg::RegTensor<uint32_t> breakAnyB;
    Reg::RegTensor<uint32_t> rowFlagA;
    Reg::RegTensor<uint32_t> rowFlagB;
    BitonicSmallRegDetectDuplicateCommonPairRow<T>(valueAddrPair, duplicateAnyA, duplicateAnyB, breakAnyA, breakAnyB,
                                                   activeMask, validMask, lowHalfMask, highHalfMask, prevLane,
                                                   laneGtZeroMask, oneValue, twoValue, zeroValue, valueSafeIdx);
    BitonicSmallRegCombineRowFlag(rowFlagA, duplicateAnyA, breakAnyA, zeroIndex, activeMask);
    BitonicSmallRegCombineRowFlag(rowFlagB, duplicateAnyB, breakAnyB, zeroIndex, activeMask);
    BitonicSmallRegDetectDuplicateEmit<ToFlagArray>(flagVec, rowFlagA, lane, rowIndexA, activeMask, flagStoreBase,
                                                    flagStoreStride);
    BitonicSmallRegDetectDuplicateEmit<ToFlagArray>(flagVec, rowFlagB, lane, rowIndexB, activeMask, flagStoreBase,
                                                    flagStoreStride);
}

/*!
 * \brief 无重复值检测双行打包路径的尾行发射 (Reg/SIMD，B32 尾行体)。
 *
 * 尾行安全索引 tailSafeIdx 依赖 laneMod，在本体内由 Select 构造；
 * 行地址与行号由批入口传入。
 */
template <typename T, bool ToFlagArray>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB32TailEmit(
    Reg::RegTensor<uint32_t>& flagVec, Reg::RegTensor<uint32_t>& lane, Reg::RegTensor<uint32_t>& laneMod,
    __ubuf__ T* valueAddrTail, uint16_t rowIndex, Reg::MaskReg& activeMask, Reg::MaskReg& validMask,
    Reg::MaskReg& lowHalfMask, Reg::MaskReg& highHalfMask, Reg::RegTensor<uint32_t>& prevLane,
    Reg::MaskReg& laneGtZeroMask, Reg::RegTensor<uint32_t>& oneValue, Reg::RegTensor<uint32_t>& twoValue,
    Reg::RegTensor<T>& zeroValue, Reg::RegTensor<uint32_t>& zeroIndex, __ubuf__ uint32_t* flagStoreBase,
    uint32_t flagStoreStride)
{
    Reg::RegTensor<uint32_t> tailSafeIdx;
    Reg::Select<uint32_t>(tailSafeIdx, laneMod, zeroIndex, validMask);
    Reg::RegTensor<uint32_t> tailDuplicateAnyA;
    Reg::RegTensor<uint32_t> tailDuplicateAnyB;
    Reg::RegTensor<uint32_t> tailBreakAnyA;
    Reg::RegTensor<uint32_t> tailBreakAnyB;
    Reg::RegTensor<uint32_t> tailRowFlag;
    BitonicSmallRegDetectDuplicateCommonPairRow<T>(
        valueAddrTail, tailDuplicateAnyA, tailDuplicateAnyB, tailBreakAnyA, tailBreakAnyB, activeMask, validMask,
        lowHalfMask, highHalfMask, prevLane, laneGtZeroMask, oneValue, twoValue, zeroValue, tailSafeIdx);
    BitonicSmallRegCombineRowFlag(tailRowFlag, tailDuplicateAnyA, tailBreakAnyA, zeroIndex, activeMask);
    BitonicSmallRegDetectDuplicateEmit<ToFlagArray>(flagVec, tailRowFlag, lane, rowIndex, activeMask, flagStoreBase,
                                                    flagStoreStride);
}

/*!
 * \brief 无重复值检测批的 16 位类型路径 (Reg/SIMD，sizeof(T) == 2 分支)。
 */
template <typename T, bool IsLargest, bool ToFlagArray>
__aicore__ inline void BitonicSmallRegDetectDuplicateBatchB16(__ubuf__ T* valueBase, __ubuf__ uint32_t* flagStoreBase,
                                                              uint32_t k, uint32_t rowCount, uint32_t valueStride,
                                                              uint32_t flagStoreStride)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
        uint32_t validCount = k;
        Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
        Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
        Reg::RegTensor<uint32_t> lane;
        Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
        Reg::RegTensor<uint32_t> prevLane;
        BitonicSmallRegPrevLane(prevLane, lane, activeMask);
        Reg::MaskReg laneGtZeroMask;
        Reg::Compares<uint32_t, CMPMODE::GT>(laneGtZeroMask, lane, 0U, activeMask);
        Reg::RegTensor<uint32_t> oneValue;
        Reg::Duplicate(oneValue, BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE);
        Reg::RegTensor<uint32_t> twoValue;
        Reg::Duplicate(twoValue, BITONIC_SMALL_TOPK_ROW_FLAG_HAS_VALUE_BREAK);
        Reg::RegTensor<uint32_t> zeroValueBits;
        Reg::Duplicate(zeroValueBits, 0U);
        Reg::RegTensor<uint32_t> zeroIndex;
        Reg::Duplicate(zeroIndex, 0U);
        Reg::RegTensor<uint32_t> flagVec;
        if constexpr (ToFlagArray) {
            Reg::Duplicate(flagVec, 0U);
        }
        uint16_t rowCountU16 = static_cast<uint16_t>(rowCount);
        for (uint16_t i = 0U; i < rowCountU16; ++i) {
            Reg::RegTensor<uint32_t> duplicateAny;
            Reg::RegTensor<uint32_t> breakAny;
            Reg::RegTensor<uint32_t> rowFlag;
            BitonicSmallRegDetectDuplicateB16Row<T, IsLargest>(valueBase + i * valueStride, duplicateAny, breakAny, k,
                                                               activeMask, validMask, lane, prevLane, laneGtZeroMask,
                                                               oneValue, twoValue, zeroValueBits);
            BitonicSmallRegCombineRowFlag(rowFlag, duplicateAny, breakAny, zeroIndex, activeMask);
            BitonicSmallRegDetectDuplicateEmit<ToFlagArray>(flagVec, rowFlag, lane, i, activeMask, flagStoreBase,
                                                            flagStoreStride);
        }
        if constexpr (ToFlagArray) {
            Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(flagStoreBase, flagVec, 1U, activeMask);
        }
    }
}

/*!
 * \brief B64 重复检测的行级比较常量 (Reg/SIMD 路径)。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB64RowConstants(
    Reg::RegTensor<uint32_t>& oneValue, Reg::RegTensor<uint32_t>& twoValue, Reg::RegTensor<uint32_t>& zeroValue,
    Reg::RegTensor<uint32_t>& keyHighXor, Reg::RegTensor<uint32_t>& allBitsValue, Reg::RegTensor<uint32_t>& zeroIndex)
{
    Reg::Duplicate(oneValue, BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE);
    Reg::Duplicate(twoValue, BITONIC_SMALL_TOPK_ROW_FLAG_HAS_VALUE_BREAK);
    Reg::Duplicate(zeroValue, 0U);
    Reg::Duplicate(keyHighXor, std::is_signed_v<T> ? 0x80000000U : 0U);
    Reg::Duplicate(allBitsValue, UINT32_MAX);
    Reg::Duplicate(zeroIndex, 0U);
}

/*!
 * \brief 无重复值检测批的 64 位类型路径 (Reg/SIMD，sizeof(T) == 8 分支)。
 */
template <typename T, bool IsLargest, bool ToFlagArray>
__aicore__ inline void BitonicSmallRegDetectDuplicateBatchB64(__ubuf__ T* valueBase, __ubuf__ uint32_t* flagStoreBase,
                                                              uint32_t k, uint32_t rowCount, uint32_t valueStride,
                                                              uint32_t flagStoreStride)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
        uint32_t validCount = k;
        uint32_t rawCount = k * 2U;
        Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
        Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
        Reg::MaskReg rawMask = Reg::UpdateMask<uint32_t>(rawCount);
        Reg::RegTensor<uint32_t> lane;
        Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
        Reg::RegTensor<uint32_t> prevLane;
        BitonicSmallRegPrevLane(prevLane, lane, activeMask);
        Reg::MaskReg laneGtZeroMask;
        Reg::Compares<uint32_t, CMPMODE::GT>(laneGtZeroMask, lane, 0U, activeMask);
        Reg::RegTensor<uint32_t> oneValue;
        Reg::RegTensor<uint32_t> twoValue;
        Reg::RegTensor<uint32_t> zeroValue;
        Reg::RegTensor<uint32_t> keyHighXor;
        Reg::RegTensor<uint32_t> allBitsValue;
        Reg::RegTensor<uint32_t> zeroIndex;
        BitonicSmallRegDetectDuplicateB64RowConstants<T>(oneValue, twoValue, zeroValue, keyHighXor, allBitsValue,
                                                         zeroIndex);
        Reg::RegTensor<uint32_t> flagVec;
        if constexpr (ToFlagArray) {
            Reg::Duplicate(flagVec, 0U);
        }
        uint16_t rowCountU16 = static_cast<uint16_t>(rowCount);
        for (uint16_t i = 0U; i < rowCountU16; ++i) {
            Reg::RegTensor<uint32_t> duplicateAny;
            Reg::RegTensor<uint32_t> breakAny;
            Reg::RegTensor<uint32_t> rowFlag;
            BitonicSmallRegDetectDuplicateB64Row<T, IsLargest>(
                valueBase + i * valueStride, duplicateAny, breakAny, activeMask, validMask, rawMask, lane, prevLane,
                laneGtZeroMask, oneValue, twoValue, zeroValue, keyHighXor, allBitsValue);
            BitonicSmallRegCombineRowFlag(rowFlag, duplicateAny, breakAny, zeroIndex, activeMask);
            BitonicSmallRegDetectDuplicateEmit<ToFlagArray>(flagVec, rowFlag, lane, i, activeMask, flagStoreBase,
                                                            flagStoreStride);
        }
        if constexpr (ToFlagArray) {
            Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(flagStoreBase, flagVec, 1U, activeMask);
        }
    }
}

/*!
 * \brief B32 双行打包重复检测的 lane/mask 初始化 (Reg/SIMD 路径)。
 */
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB32Masks(uint32_t k, Reg::MaskReg& activeMask,
                                                                   Reg::RegTensor<uint32_t>& lane,
                                                                   Reg::RegTensor<uint32_t>& laneMod,
                                                                   Reg::MaskReg& validMask, Reg::MaskReg& highHalfMask,
                                                                   Reg::RegTensor<uint32_t>& prevLane,
                                                                   Reg::MaskReg& laneGtZeroMask)
{
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0); // 0..63 全宽
    Reg::RegTensor<uint32_t> laneMaskReg;
    Reg::Duplicate(laneMaskReg, BITONIC_SMALL_TOPK_SIZE - 1U);
    Reg::And(laneMod, lane, laneMaskReg, activeMask);
    Reg::RegTensor<uint32_t> kReg;
    Reg::Duplicate(kReg, k);
    Reg::Compare<uint32_t, CMPMODE::LT>(validMask, laneMod, kReg, activeMask);
    Reg::Compares<uint32_t, CMPMODE::GE>(highHalfMask, lane, BITONIC_SMALL_TOPK_SIZE, activeMask);
    BitonicSmallRegPrevLanePair(prevLane, lane, activeMask);
    Reg::Compares<uint32_t, CMPMODE::GT>(laneGtZeroMask, laneMod, 0U, activeMask);
}

/*!
 * \brief B32 双行打包重复检测的行级常量 (Reg/SIMD 路径)。
 *
 * oneValue/twoValue 为行 flag 状态位，zeroValue/zeroIndex 为 T 类型值与索引的防御 0。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegDetectDuplicateB32RowConstants(Reg::RegTensor<uint32_t>& oneValue,
                                                                          Reg::RegTensor<uint32_t>& twoValue,
                                                                          Reg::RegTensor<T>& zeroValue,
                                                                          Reg::RegTensor<uint32_t>& zeroIndex)
{
    Reg::Duplicate(oneValue, BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE);
    Reg::Duplicate(twoValue, BITONIC_SMALL_TOPK_ROW_FLAG_HAS_VALUE_BREAK);
    Reg::Duplicate(zeroValue, static_cast<T>(0));
    Reg::Duplicate(zeroIndex, 0U);
}

/*!
 * \brief 无重复值检测批的 32 位类型路径 (Reg/SIMD，sizeof(T) == 4 分支，双行打包)。
 */
template <typename T, bool IsLargest, bool ToFlagArray>
__aicore__ inline void BitonicSmallRegDetectDuplicateBatchB32(__ubuf__ T* valueBase, __ubuf__ uint32_t* flagStoreBase,
                                                              uint32_t k, uint32_t rowCount, uint32_t valueStride,
                                                              uint32_t flagStoreStride)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE * 2U;
        Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
        Reg::RegTensor<uint32_t> lane;
        Reg::RegTensor<uint32_t> laneMod;
        Reg::MaskReg validMask;
        Reg::MaskReg lowHalfMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL32>();
        Reg::MaskReg highHalfMask;
        Reg::RegTensor<uint32_t> prevLane;
        Reg::MaskReg laneGtZeroMask;
        BitonicSmallRegDetectDuplicateB32Masks(k, activeMask, lane, laneMod, validMask, highHalfMask, prevLane,
                                               laneGtZeroMask);
        Reg::RegTensor<uint32_t> oneValue;
        Reg::RegTensor<uint32_t> twoValue;
        Reg::RegTensor<T> zeroValue;
        Reg::RegTensor<uint32_t> zeroIndex;
        BitonicSmallRegDetectDuplicateB32RowConstants<T>(oneValue, twoValue, zeroValue, zeroIndex);
        Reg::RegTensor<uint32_t> valueRawIdx;
        Reg::RegTensor<uint32_t> valueSafeIdx;
        BitonicSmallRegPairGatherIndex(valueStride, laneMod, zeroIndex, activeMask, validMask, highHalfMask,
                                       valueRawIdx, valueSafeIdx);
        Reg::RegTensor<uint32_t> flagVec;
        if constexpr (ToFlagArray) {
            Reg::Duplicate(flagVec, 0U);
        }
        uint16_t pairCountU16 = static_cast<uint16_t>(rowCount / 2U);
        for (uint16_t p = 0U; p < pairCountU16; ++p) {
            BitonicSmallRegDetectDuplicateB32PairEmit<T, ToFlagArray>(
                flagVec, lane, valueBase + 2U * p * valueStride, p * 2U, p * 2U + 1U, activeMask, validMask,
                lowHalfMask, highHalfMask, prevLane, laneGtZeroMask, oneValue, twoValue, zeroValue, valueSafeIdx,
                zeroIndex, flagStoreBase, flagStoreStride);
        }
        BitonicSmallRegDetectDuplicateB32TailEmit<T, ToFlagArray>(
            flagVec, lane, laneMod, valueBase + (rowCount - 1U) * valueStride, static_cast<uint16_t>(rowCount - 1U),
            activeMask, validMask, lowHalfMask, highHalfMask, prevLane, laneGtZeroMask, oneValue, twoValue, zeroValue,
            zeroIndex, flagStoreBase, flagStoreStride);
        if constexpr (ToFlagArray) {
            Reg::MaskReg laneLowMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL32>();
            Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(flagStoreBase, flagVec, 1U, laneLowMask);
        }
    }
}

/*!
 * \brief 无重复值检测批
 */
template <typename T, bool IsLargest, bool ToFlagArray>
__aicore__ inline void BitonicSmallRegDetectDuplicateBatch(__ubuf__ T* valueBase, __ubuf__ uint32_t* flagStoreBase,
                                                           uint32_t k, uint32_t rowCount, uint32_t valueStride,
                                                           uint32_t flagStoreStride)
{
    if constexpr (sizeof(T) == 2U) {
        BitonicSmallRegDetectDuplicateBatchB16<T, IsLargest, ToFlagArray>(valueBase, flagStoreBase, k, rowCount,
                                                                          valueStride, flagStoreStride);
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        BitonicSmallRegDetectDuplicateBatchB64<T, IsLargest, ToFlagArray>(valueBase, flagStoreBase, k, rowCount,
                                                                          valueStride, flagStoreStride);
    } else {
        BitonicSmallRegDetectDuplicateBatchB32<T, IsLargest, ToFlagArray>(valueBase, flagStoreBase, k, rowCount,
                                                                          valueStride, flagStoreStride);
    }
}

/*!
 * \brief 恢复被 flag 借写的 index[0] (Reg/SIMD 路径，早退出块)。
 */
__simd_callee__ inline void BitonicSmallRegRestoreIndex0(__ubuf__ uint32_t* indexAddr, uint32_t savedIndex0)
{
    Reg::RegTensor<uint32_t> savedIndexReg;
    Reg::Duplicate(savedIndexReg, savedIndex0);
    BitonicSmallRegStoreIndex0(indexAddr, savedIndexReg);
}

/*!
 * \brief 通用类型的双调网络单次 compare-swap 阶段 (Reg/SIMD 路径)。
 */
template <typename T, bool IsLargest, bool CompareGroup, uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallRegSwapStage(Reg::RegTensor<T>& value, Reg::RegTensor<uint32_t>& index,
                                                     Reg::RegTensor<uint32_t>& group, Reg::RegTensor<uint32_t>& lane,
                                                     Reg::MaskReg& activeMask)
{
    using GatherIndexT = BitonicSmallGatherIndexType<T>;
    using GatherSignedIndexT = BitonicSmallGatherSignedIndexType<T>;
    Reg::RegTensor<GatherSignedIndexT> valueLane;
    Reg::RegTensor<GatherSignedIndexT> valueStride;
    Reg::RegTensor<GatherSignedIndexT> valuePeerIndex;
    Reg::RegTensor<uint32_t> indexStride;
    Reg::RegTensor<uint32_t> indexPeerIndex;
    Reg::RegTensor<uint32_t> directionBit;
    Reg::Arange(valueLane, static_cast<GatherSignedIndexT>(0));
    Reg::Duplicate(valueStride, static_cast<GatherSignedIndexT>(Stride));
    Reg::Xor(valuePeerIndex, valueLane, valueStride, activeMask);
    Reg::Duplicate(indexStride, Stride);
    Reg::Xor(indexPeerIndex, lane, indexStride, activeMask);

    Reg::RegTensor<T> peerValue;
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::RegTensor<uint32_t> peerGroup;
    Reg::Gather(peerValue, value, (Reg::RegTensor<GatherIndexT>&)valuePeerIndex);
    Reg::Gather(peerIndex, index, indexPeerIndex);
    if constexpr (CompareGroup) {
        Reg::Gather(peerGroup, group, indexPeerIndex);
    }

    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, indexPeerIndex, activeMask);
    Reg::RegTensor<T> lowValue;
    Reg::RegTensor<T> highValue;
    Reg::RegTensor<uint32_t> lowIndex;
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<T>(lowValue, value, peerValue, lowLaneMask);
    Reg::Select<T>(highValue, peerValue, value, lowLaneMask);
    Reg::Select<uint32_t>(lowIndex, index, peerIndex, lowLaneMask);
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);

    Reg::MaskReg compareMask;
    if constexpr (CompareGroup) {
        Reg::RegTensor<uint32_t> lowGroup;
        Reg::RegTensor<uint32_t> highGroup;
        Reg::MaskReg groupEqualMask;
        Reg::MaskReg indexLessMask;
        Reg::Select<uint32_t>(lowGroup, group, peerGroup, lowLaneMask);
        Reg::Select<uint32_t>(highGroup, peerGroup, group, lowLaneMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(compareMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(groupEqualMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(indexLessMask, lowIndex, highIndex, activeMask);
        Reg::And(groupEqualMask, groupEqualMask, indexLessMask, activeMask);
        Reg::Or(compareMask, compareMask, groupEqualMask, activeMask);
    } else {
        BitonicSmallRegValueBetter<T, IsLargest>(compareMask, lowValue, highValue, activeMask);
    }

    Reg::MaskReg lowValidMask;
    Reg::MaskReg highValidMask;
    Reg::MaskReg highInvalidMask;
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::NE>(lowValidMask, lowIndex, UINT32_MAX, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(highValidMask, highIndex, UINT32_MAX, activeMask);
    Reg::And(swapMask, compareMask, lowValidMask, activeMask);
    Reg::Not(highInvalidMask, highValidMask, activeMask);
    Reg::Or(swapMask, swapMask, highInvalidMask, activeMask);

    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::Duplicate(indexStride, Size);
        Reg::And(directionBit, lane, indexStride, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, directionBit, 0U, activeMask);
    }

    Reg::MaskReg notSwapMask;
    Reg::MaskReg notDirectionMask;
    Reg::MaskReg takePeerMask;
    Reg::MaskReg keepDirectionMask;
    Reg::Not(notSwapMask, swapMask, activeMask);
    Reg::Not(notDirectionMask, directionMask, activeMask);
    Reg::And(takePeerMask, swapMask, directionMask, activeMask);
    Reg::And(keepDirectionMask, notSwapMask, notDirectionMask, activeMask);
    Reg::Or(takePeerMask, takePeerMask, keepDirectionMask, activeMask);
    Reg::Select<T>(value, peerValue, value, takePeerMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
    if constexpr (CompareGroup) {
        Reg::Select<uint32_t>(group, peerGroup, group, takePeerMask);
    }
}

/*!
 * \brief 通用类型的双调排序网络完整序列 (Reg/SIMD 路径)。
 */
template <typename T, bool IsLargest, bool CompareGroup>
__simd_callee__ inline void BitonicSmallRegBitonicNetwork(Reg::RegTensor<T>& value, Reg::RegTensor<uint32_t>& index,
                                                          Reg::RegTensor<uint32_t>& group,
                                                          Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 2U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 4U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 4U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 4U, 8U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 8U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 8U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 8U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 4U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 16U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 8U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 4U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 32U>(value, index, group, lane, activeMask);
}

/*!
 * \brief 通用类型 (4 字节) 的全量收尾选择函数 (Reg/SIMD 路径，双行打包 64-lane)。
 */
template <typename T, bool IsLargest, bool RepairIndex0 = true>
__simd_callee__ inline void BitonicSmallRegFinalizeSelectionCommonPair(__ubuf__ T* valueAddrA,
                                                                       __ubuf__ uint32_t* indexAddrA, uint32_t k,
                                                                       uint32_t fixedIndex0A, uint32_t fixedIndex0B,
                                                                       uint32_t rowOffsetValue, uint32_t rowOffsetIndex)
{
    static_assert(sizeof(T) == sizeof(uint32_t));
    uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE * 2U;
    Reg::MaskReg activeMask = Reg::UpdateMask<T>(activeCount);
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<uint32_t> laneMod;
    Reg::MaskReg validMask;
    Reg::MaskReg highHalfMask;
    BitonicSmallRegPairLaneMasks(k, activeMask, lane, laneMod, validMask, highHalfMask);
    Reg::RegTensor<uint32_t> zeroIndex;
    Reg::Duplicate(zeroIndex, 0U);
    Reg::RegTensor<uint32_t> valueRawIdx;
    Reg::RegTensor<uint32_t> valueSafeIdx;
    BitonicSmallRegPairGatherIndex(rowOffsetValue, laneMod, zeroIndex, activeMask, validMask, highHalfMask, valueRawIdx,
                                   valueSafeIdx);
    Reg::RegTensor<uint32_t> indexRawIdx;
    Reg::RegTensor<uint32_t> indexSafeIdx;
    BitonicSmallRegPairGatherIndex(rowOffsetIndex, laneMod, zeroIndex, activeMask, validMask, highHalfMask, indexRawIdx,
                                   indexSafeIdx);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::RegTensor<T> value;
    Reg::Gather(value, valueAddrA, valueSafeIdx, activeMask);
    Reg::RegTensor<T> zeroValue;
    Reg::Duplicate(zeroValue, static_cast<T>(0));
    Reg::Select<T>(value, value, zeroValue, validMask);
    Reg::RegTensor<uint32_t> index;
    BitonicSmallRegLoadPairIndex<RepairIndex0>(indexAddrA, indexSafeIdx, index, fixedIndex0A, fixedIndex0B, lane,
                                               activeMask, validMask);
    Reg::RegTensor<uint32_t> thresholdIndex;
    BitonicSmallRegPairThresholdIndex(lane, k, activeMask, thresholdIndex);
    Reg::RegTensor<T> threshold;
    Reg::Gather(threshold, value, thresholdIndex);
    Reg::MaskReg strictMask;
    BitonicSmallRegValueBetter<T, IsLargest>(strictMask, value, threshold, activeMask);
    Reg::And(strictMask, strictMask, validMask, activeMask);

    // Reorder the selected candidates by (strict/equal group, original index) to match BITONIC gather.
    Reg::RegTensor<uint32_t> group;
    BitonicSmallRegClassifyStrictGroup(strictMask, validMask, activeMask, group);
    BitonicSmallRegBitonicNetwork<T, IsLargest, true>(value, index, group, lane, activeMask);
    BitonicSmallRegBitonicNetwork<T, IsLargest, false>(value, index, group, lane, activeMask);

    Reg::Scatter(valueAddrA, value, valueRawIdx, validMask);
    Reg::Scatter(indexAddrA, index, indexRawIdx, validMask);
}

/*!
 * \brief 全等场景的单次 compare-swap 阶段 (Reg/SIMD 路径, 全等快路径)。
 */
template <uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallRegAllEqualSwapStage(Reg::RegTensor<uint32_t>& index,
                                                             Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> strideReg;
    Reg::RegTensor<uint32_t> peerLane;
    Reg::Duplicate(strideReg, Stride);
    Reg::Xor(peerLane, lane, strideReg, activeMask);
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::Gather(peerIndex, index, peerLane);
    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, peerLane, activeMask);
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::EQ>(swapMask, highIndex, UINT32_MAX, activeMask);
    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::RegTensor<uint32_t> sizeReg;
        Reg::RegTensor<uint32_t> directionBit;
        Reg::Duplicate(sizeReg, Size);
        Reg::And(directionBit, lane, sizeReg, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, directionBit, 0U, activeMask);
    }
    Reg::MaskReg takePeerMask;
    Reg::Xor(takePeerMask, swapMask, directionMask, activeMask);
    Reg::Not(takePeerMask, takePeerMask, activeMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
}

/*!
 * \brief 全等场景的双调排序网络完整序列 (Reg/SIMD 路径，全等快路径)。
 */
__simd_callee__ inline void BitonicSmallRegAllEqualBitonicNetwork(Reg::RegTensor<uint32_t>& index,
                                                                  Reg::RegTensor<uint32_t>& lane,
                                                                  Reg::MaskReg& activeMask)
{
    BitonicSmallRegAllEqualSwapStage<1U, 2U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<2U, 4U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<1U, 4U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<4U, 8U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<2U, 8U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<1U, 8U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<8U, 16U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<4U, 16U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<2U, 16U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<1U, 16U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<16U, 32U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<8U, 32U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<4U, 32U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<2U, 32U>(index, lane, activeMask);
    BitonicSmallRegAllEqualSwapStage<1U, 32U>(index, lane, activeMask);
}

/*!
 * \brief 全等收尾选择函数 (Reg/SIMD 路径，全等快路径，dtype 无关)。
 */
template <bool RepairIndex0 = true>
__simd_callee__ inline void BitonicSmallRegFinalizeSelectionAllEqual(__ubuf__ uint32_t* indexAddr, uint32_t k,
                                                                     uint32_t fixedIndex0)
{
    uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
    uint32_t validCount = k;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<uint32_t> invalidIndex;
    Reg::RegTensor<uint32_t> lane;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(index, indexAddr, 1U, validMask);
    Reg::Duplicate(invalidIndex, UINT32_MAX);
    Reg::Select<uint32_t>(index, index, invalidIndex, validMask);
    if constexpr (RepairIndex0) {
        Reg::RegTensor<uint32_t> fixedIndexReg;
        Reg::MaskReg laneZeroMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
        Reg::Duplicate(fixedIndexReg, fixedIndex0);
        Reg::Select<uint32_t>(index, fixedIndexReg, index, laneZeroMask);
    }
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    BitonicSmallRegAllEqualBitonicNetwork(index, lane, activeMask);
    Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(indexAddr, index, 1U, validMask);
}

/*!
 * \brief 全等行批置换内核 (Reg/SIMD 路径，常数置换快路径，dtype 无关)。
 */
template <bool FromFlagArray>
__aicore__ inline void BitonicSmallRegAllEqualBatchPermute(__ubuf__ uint32_t* indexBase, __ubuf__ uint32_t* flagBase,
                                                           uint32_t k, uint32_t rowCount, uint32_t indexStride)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
        uint32_t validCount = k;
        Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
        Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
        Reg::RegTensor<uint32_t> invalidIndex;
        Reg::Duplicate(invalidIndex, UINT32_MAX);
        Reg::RegTensor<uint32_t> zeroIndex;
        Reg::Duplicate(zeroIndex, 0U);
        Reg::RegTensor<uint32_t> sigma;
        Reg::RegTensor<uint32_t> lane;
        Reg::Arange((Reg::RegTensor<int32_t>&)sigma, 0);
        Reg::Select<uint32_t>(sigma, sigma, invalidIndex, validMask);
        Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
        BitonicSmallRegAllEqualBitonicNetwork(sigma, lane, activeMask);
        Reg::RegTensor<uint32_t> sigmaSafe;
        Reg::Select<uint32_t>(sigmaSafe, sigma, zeroIndex, validMask);
        Reg::RegTensor<uint32_t> flagVec;
        if constexpr (FromFlagArray) {
            Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(flagVec, flagBase, 1U, activeMask);
        }
        uint16_t rowCountU16 = static_cast<uint16_t>(rowCount);
        for (uint16_t i = 0U; i < rowCountU16; ++i) {
            __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
            Reg::RegTensor<uint32_t> index;
            Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(index, indexAddr, 1U, validMask);
            Reg::RegTensor<uint32_t> rowFlag;
            Reg::MaskReg allEqMask;
            if constexpr (FromFlagArray) {
                Reg::RegTensor<uint32_t> rowIndexReg;
                Reg::Duplicate(rowIndexReg, static_cast<uint32_t>(i));
                Reg::Gather(rowFlag, flagVec, rowIndexReg);
            } else {
                Reg::Gather(rowFlag, index, zeroIndex);
            }
            Reg::Compares<uint32_t, CMPMODE::EQ>(allEqMask, rowFlag, BITONIC_SMALL_TOPK_ROW_FLAG_ALL_EQ, activeMask);
            Reg::RegTensor<uint32_t> permuted;
            Reg::Gather(permuted, index, sigmaSafe);
            Reg::RegTensor<uint32_t> out;
            Reg::Select<uint32_t>(out, permuted, index, allEqMask);
            Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(indexAddr, out, 1U, validMask);
        }
    }
}

/*!
 * \brief 恢复全部行的借写 index[0] (Reg/SIMD 路径，无重复早退块)。
 */
__aicore__ inline void BitonicSmallRegRestoreAllIndex0(__ubuf__ uint32_t* indexBase, const uint32_t* savedIndex0,
                                                       uint32_t rowCount, uint32_t indexStride)
{
    for (uint32_t i = 0U; i < rowCount; ++i) {
        uint32_t savedIndex0Row = savedIndex0[i];
        __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
        __VEC_SCOPE__ { BitonicSmallRegRestoreIndex0(indexAddr, savedIndex0Row); }
    }
}

/*!
 * \brief 重复行双行编排收敛 helper (Reg/SIMD 路径)：按 duplicateList 两行成对终排，
 *        奇数尾行自配对；B16/B32 经 sizeof(T) 编译期分发底层 Pair 函数。
 */
template <typename T, bool IsLargest, bool RepairIndex0>
__aicore__ inline void BitonicSmallRegFinalizeDuplicateRowPairs(__ubuf__ T* valueBase, __ubuf__ uint32_t* indexBase,
                                                                const uint16_t* duplicateList, uint16_t duplicateCount,
                                                                uint32_t k, uint32_t valueStride, uint32_t indexStride,
                                                                const uint32_t* savedIndex0)
{
    for (uint16_t q = 0U; q + 1U < duplicateCount; q += 2U) {
        uint32_t rowA = duplicateList[q];
        uint32_t rowB = duplicateList[q + 1U];
        __ubuf__ T* valueAddrA = valueBase + rowA * valueStride;
        __ubuf__ uint32_t* indexAddrA = indexBase + rowA * indexStride;
        uint32_t restoreA = 0U;
        uint32_t restoreB = 0U;
        if constexpr (RepairIndex0) {
            restoreA = savedIndex0[rowA];
            restoreB = savedIndex0[rowB];
        }
        __VEC_SCOPE__
        {
            if constexpr (sizeof(T) == 2U) {
                BitonicSmallRegFinalizeSelectionB16Pair<T, IsLargest, RepairIndex0>(
                    valueAddrA, indexAddrA, k, restoreA, restoreB, (rowB - rowA) * valueStride,
                    (rowB - rowA) * indexStride);
            } else {
                BitonicSmallRegFinalizeSelectionCommonPair<T, IsLargest, RepairIndex0>(
                    valueAddrA, indexAddrA, k, restoreA, restoreB, (rowB - rowA) * valueStride,
                    (rowB - rowA) * indexStride);
            }
        }
    }
    if ((duplicateCount & 1U) != 0U) {
        uint32_t rowT = duplicateList[duplicateCount - 1U];
        uint32_t restoreT = 0U;
        if constexpr (RepairIndex0) {
            restoreT = savedIndex0[rowT];
        }
        __VEC_SCOPE__
        {
            if constexpr (sizeof(T) == 2U) {
                BitonicSmallRegFinalizeSelectionB16Pair<T, IsLargest, RepairIndex0>(
                    valueBase + rowT * valueStride, indexBase + rowT * indexStride, k, restoreT, restoreT, 0U, 0U);
            } else {
                BitonicSmallRegFinalizeSelectionCommonPair<T, IsLargest, RepairIndex0>(
                    valueBase + rowT * valueStride, indexBase + rowT * indexStride, k, restoreT, restoreT, 0U, 0U);
            }
        }
    }
}

/*!
 * \brief 借写版重复行终排分发 (Reg/SIMD 路径)：B32/B16 走共享双行编排 + clean 行
 *        index[0] 恢复，B64 逐行终排（恢复内联）。
 */
template <typename T, bool IsLargest>
__aicore__ inline void BitonicSmallRegFinalizeBorrowedDuplicateRows(
    __ubuf__ T* valueBase, __ubuf__ uint32_t* indexBase, const uint16_t* duplicateList, uint16_t duplicateCount,
    const uint8_t* duplicateRows, const uint8_t* allEqRows, const uint32_t* savedIndex0, uint32_t k, uint32_t rowCount,
    uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        BitonicSmallRegFinalizeDuplicateRowPairs<T, IsLargest, true>(
            valueBase, indexBase, duplicateList, duplicateCount, k, valueStride, indexStride, savedIndex0);
        for (uint32_t i = 0U; i < rowCount; ++i) {
            if (duplicateRows[i] == 0U) {
                uint32_t savedIndex0Row = savedIndex0[i];
                __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
                __VEC_SCOPE__ { BitonicSmallRegRestoreIndex0(indexAddr, savedIndex0Row); }
            }
        }
    } else if constexpr (sizeof(T) == 2U) {
        BitonicSmallRegFinalizeDuplicateRowPairs<T, IsLargest, true>(
            valueBase, indexBase, duplicateList, duplicateCount, k, valueStride, indexStride, savedIndex0);
        for (uint32_t i = 0U; i < rowCount; ++i) {
            if (duplicateRows[i] == 0U) {
                uint32_t savedIndex0Row = savedIndex0[i];
                __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
                __VEC_SCOPE__ { BitonicSmallRegRestoreIndex0(indexAddr, savedIndex0Row); }
            }
        }
    } else {
        for (uint32_t i = 0U; i < rowCount; ++i) {
            __ubuf__ T* valueAddr = valueBase + i * valueStride;
            __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
            uint32_t savedIndex0Row = savedIndex0[i];
            if (duplicateRows[i] != 0U && allEqRows[i] == 0U) {
                __VEC_SCOPE__
                {
                    BitonicSmallRegFinalizeSelectionB64<T, IsLargest>(valueAddr, indexAddr, k, savedIndex0Row);
                }
            } else if (duplicateRows[i] == 0U) {
                __VEC_SCOPE__ { BitonicSmallRegRestoreIndex0(indexAddr, savedIndex0Row); }
            }
        }
    }
}

/*!
 * \brief 借写版全等行处理 (Reg/SIMD 路径)：allEqCount>=2 批置换+src0Pos 修补+S_V 同步，
 *        ==1 单行全等收尾。
 */
template <typename T>
__aicore__ inline void BitonicSmallRegFinalizeAllEqualBorrowed(__ubuf__ uint32_t* indexBase,
                                                               __local_mem__ uint32_t* flagBase,
                                                               const uint8_t* allEqRows, const uint32_t* savedIndex0,
                                                               uint32_t k, uint32_t rowCount, uint32_t indexStride,
                                                               uint32_t allEqCount, event_t syncEvent)
{
    if (allEqCount >= 2U) {
        BitonicSmallRegAllEqualBatchPermute<false>(indexBase, indexBase, k, rowCount, indexStride);
        SetFlag<HardEvent::V_S>(syncEvent);
        WaitFlag<HardEvent::V_S>(syncEvent);
        uint32_t src0Pos = BITONIC_SMALL_TOPK_ALLEQ_PERM_SRC0[k - 2U];
        for (uint32_t i = 0U; i < rowCount; ++i) {
            if (allEqRows[i] != 0U) {
                flagBase[i * indexStride + src0Pos] = savedIndex0[i];
            }
        }
        event_t svEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(svEvent);
        WaitFlag<HardEvent::S_V>(svEvent);
    } else if (allEqCount == 1U) {
        for (uint32_t i = 0U; i < rowCount; ++i) {
            if (allEqRows[i] != 0U) {
                __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
                uint32_t savedIndex0Row = savedIndex0[i];
                __VEC_SCOPE__ { BitonicSmallRegFinalizeSelectionAllEqual<true>(indexAddr, k, savedIndex0Row); }
            }
        }
    }
}

/*!
 * \brief Reg/SIMD 路径收尾选择编排入口 (检测批量化)：无重复值早退 +
 * 按位宽分发全量路径，按批编排。
 */
template <typename T, bool IsLargest>
__aicore__ inline void BitonicSmallRegFinalizeSelectionBatch(__ubuf__ T* valueBase, __ubuf__ uint32_t* indexBase,
                                                             uint32_t k, uint32_t rowCount, uint32_t valueStride,
                                                             uint32_t indexStride)
{
    if (rowCount == 0U) {
        return;
    }
    __local_mem__ uint32_t* flagBase = (__local_mem__ uint32_t*)indexBase;
    event_t syncEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(syncEvent);
    WaitFlag<HardEvent::V_S>(syncEvent);
    uint32_t savedIndex0[BITONIC_SMALL_TOPK_MAX_ROWS];
    for (uint32_t i = 0U; i < rowCount; ++i) {
        savedIndex0[i] = flagBase[i * indexStride];
    }
    BitonicSmallRegDetectDuplicateBatch<T, IsLargest, false>(valueBase, indexBase, k, rowCount, valueStride,
                                                             indexStride);
    SetFlag<HardEvent::V_S>(syncEvent);
    WaitFlag<HardEvent::V_S>(syncEvent);
    uint8_t duplicateRows[BITONIC_SMALL_TOPK_MAX_ROWS];
    uint8_t allEqRows[BITONIC_SMALL_TOPK_MAX_ROWS];
    uint16_t duplicateList[BITONIC_SMALL_TOPK_MAX_ROWS];
    uint16_t duplicateCount = 0U;
    uint32_t anyDuplicate = 0U;
    uint32_t allEqCount = 0U;
    for (uint32_t i = 0U; i < rowCount; ++i) {
        uint32_t rowFlag = flagBase[i * indexStride];
        duplicateRows[i] = (rowFlag & BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE) != 0U ? 1U : 0U;
        allEqRows[i] = (rowFlag == BITONIC_SMALL_TOPK_ROW_FLAG_ALL_EQ) ? 1U : 0U;
        anyDuplicate |= duplicateRows[i];
        allEqCount += allEqRows[i];
        if (duplicateRows[i] != 0U && allEqRows[i] == 0U) {
            duplicateList[duplicateCount] = static_cast<uint16_t>(i);
            ++duplicateCount;
        }
    }
    if (anyDuplicate == 0U) {
        BitonicSmallRegRestoreAllIndex0(indexBase, savedIndex0, rowCount, indexStride);
        return;
    }
    BitonicSmallRegFinalizeAllEqualBorrowed<T>(indexBase, flagBase, allEqRows, savedIndex0, k, rowCount, indexStride,
                                               allEqCount, syncEvent);
    BitonicSmallRegFinalizeBorrowedDuplicateRows<T, IsLargest>(valueBase, indexBase, duplicateList, duplicateCount,
                                                               duplicateRows, allEqRows, savedIndex0, k, rowCount,
                                                               valueStride, indexStride);
}

/*!
 * \brief Reg/SIMD 路径收尾选择编排入口 (独立 flag 数组版)：无重复值早退 +
 * 按位宽分发全量路径，检测 flag 落盘独立数组。
 */
template <typename T, bool IsLargest>
__aicore__ inline void BitonicSmallRegFinalizeSelectionFlagBatch(__ubuf__ T* valueBase, __ubuf__ uint32_t* indexBase,
                                                                 __ubuf__ uint32_t* flagBase, uint32_t k,
                                                                 uint32_t rowCount, uint32_t valueStride,
                                                                 uint32_t indexStride)
{
    if (rowCount == 0U) {
        return;
    }
    BitonicSmallRegDetectDuplicateBatch<T, IsLargest, true>(valueBase, flagBase, k, rowCount, valueStride, 1U);
    event_t syncEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(syncEvent);
    WaitFlag<HardEvent::V_S>(syncEvent);
    uint8_t duplicateRows[BITONIC_SMALL_TOPK_MAX_ROWS];
    uint8_t allEqRows[BITONIC_SMALL_TOPK_MAX_ROWS];
    uint16_t duplicateList[BITONIC_SMALL_TOPK_MAX_ROWS];
    uint16_t duplicateCount = 0U;
    uint32_t anyAllEq = 0U;
    for (uint32_t i = 0U; i < rowCount; ++i) {
        uint32_t rowFlag = flagBase[i];
        duplicateRows[i] = (rowFlag & BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE) != 0U ? 1U : 0U;
        allEqRows[i] = (rowFlag == BITONIC_SMALL_TOPK_ROW_FLAG_ALL_EQ) ? 1U : 0U;
        anyAllEq |= allEqRows[i];
        if (duplicateRows[i] != 0U && allEqRows[i] == 0U) {
            duplicateList[duplicateCount] = static_cast<uint16_t>(i);
            ++duplicateCount;
        }
    }
    if (anyAllEq != 0U) {
        BitonicSmallRegAllEqualBatchPermute<true>(indexBase, flagBase, k, rowCount, indexStride);
    }
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        BitonicSmallRegFinalizeDuplicateRowPairs<T, IsLargest, false>(
            valueBase, indexBase, duplicateList, duplicateCount, k, valueStride, indexStride, nullptr);
    } else if constexpr (sizeof(T) == 2U) {
        BitonicSmallRegFinalizeDuplicateRowPairs<T, IsLargest, false>(
            valueBase, indexBase, duplicateList, duplicateCount, k, valueStride, indexStride, nullptr);
    } else {
        for (uint32_t i = 0U; i < rowCount; ++i) {
            if (duplicateRows[i] != 0U && allEqRows[i] == 0U) {
                __ubuf__ T* valueAddr = valueBase + i * valueStride;
                __ubuf__ uint32_t* indexAddr = indexBase + i * indexStride;
                __VEC_SCOPE__ { BitonicSmallRegFinalizeSelectionB64<T, IsLargest, false>(valueAddr, indexAddr, k, 0U); }
            }
        }
    }
}

} // namespace topkV2

#endif // TOP_K_SMALL_BITONIC_REG_FINALIZE_H
