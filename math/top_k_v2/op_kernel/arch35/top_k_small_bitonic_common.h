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
 * \file top_k_small_bitonic_common.h
 * \brief 小规模 (2 <= k <= 32) 双调收尾排序的公共常量、类型别名与比较 key 语义。
 */
#ifndef TOP_K_SMALL_BITONIC_COMMON_H
#define TOP_K_SMALL_BITONIC_COMMON_H

#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/math_functions.h"
#include "top_k_constant_var_simd.h"
#include "top_k_util_type_simd.h"

namespace topkV2 {
using namespace AscendC;

constexpr uint32_t BITONIC_SMALL_TOPK_MAX_ROWS = 32U;
constexpr uint32_t BITONIC_SMALL_TOPK_THREADS = BITONIC_SMALL_TOPK_SIZE * BITONIC_SMALL_TOPK_MAX_ROWS;
constexpr uint32_t BITONIC_SMALL_TOPK_ROWS_PER_LAUNCH = 1024U;
constexpr uint32_t BITONIC_SMALL_TOPK_PACKED_MAX_K = 16U;
constexpr uint32_t BITONIC_SMALL_TOPK_ALLEQ_PERM_SRC0[BITONIC_SMALL_TOPK_SIZE - 1U] = {
    1U,  2U,  1U,  1U,  1U,                                               // k = 2..6
    5U,  5U,  5U,  5U,  5U,  5U,  5U,  5U,  5U,  5U,  5U, 5U, 5U, 5U, 5U, // k = 7..21
    21U, 21U, 21U, 21U, 21U, 21U, 21U, 21U, 21U, 21U, 21U};               // k = 22..32

// 重复检测行状态位 (BitonicSmallRegCombineRowFlag 合成，2 bit 协议)：
// bit0 = 存在相邻值等价对 (NaN 感知归一)；bit1 = 存在相邻逐位不等对。
// 组合语义：0=clean(防御值) / 1=allEq(行内逐位全等，走全等快路径) /
//           2=clean(严格序，早退) / 3=full(有重复且位模式不齐，走全量 finalize)。
constexpr uint32_t BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE = 1U;
constexpr uint32_t BITONIC_SMALL_TOPK_ROW_FLAG_HAS_VALUE_BREAK = 2U;
// allEq = 有值等价对且无逐位不等对
constexpr uint32_t BITONIC_SMALL_TOPK_ROW_FLAG_ALL_EQ = BITONIC_SMALL_TOPK_ROW_FLAG_HAS_DUPLICATE;

template <typename T>
using BitonicSmallGatherIndexType = std::conditional_t<
    sizeof(T) == sizeof(uint8_t), uint8_t, std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;

template <typename T>
using BitonicSmallGatherSignedIndexType = std::conditional_t<
    sizeof(T) == sizeof(uint8_t), int8_t, std::conditional_t<sizeof(T) == sizeof(uint16_t), int16_t, int32_t>>;

template <typename T>
using BitonicSmallRegType = std::conditional_t<
    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>, int32_t,
    std::conditional_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>,
                       uint32_t, T>>;

template <typename T>
constexpr bool IsBitonicFloatType = std::is_same_v<T, float> || std::is_same_v<T, half> ||
                                    std::is_same_v<T, bfloat16_t>;

constexpr Reg::CastTrait BITONIC_SMALL_CAST_U16_TO_U32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait BITONIC_SMALL_CAST_U16_TO_U32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait BITONIC_SMALL_CAST_U32_TO_U16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

/*!
 * \brief 将 16 位数据从 UB 加载到 32 位向量寄存器，并把位模式展开为 uint32。
 *
 * \param[out] valueBits  输出的 32 位位模式寄存器，每通道一个 16 位元素的 bits
 * \param[in]  valueAddr  UB 中 16 位数据的起始地址
 * \param[in]  validCount 有效元素个数 (<= 32)
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegLoadB16Bits(Reg::RegTensor<uint32_t>& valueBits, __ubuf__ T* valueAddr,
                                                       uint32_t validCount)
{
    uint32_t valueCount = validCount;
    Reg::MaskReg valueMask = Reg::UpdateMask<T>(valueCount);
    Reg::RegTensor<uint16_t> rawValue;
    Reg::RegTensor<uint32_t> evenValue;
    Reg::RegTensor<uint32_t> oddValue;
    Reg::RegTensor<uint32_t> unusedValue;
    Reg::LoadAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY>((Reg::RegTensor<T>&)rawValue, valueAddr, 1U, valueMask);
    Reg::Cast<uint32_t, uint16_t, BITONIC_SMALL_CAST_U16_TO_U32_EVEN>(evenValue, rawValue, valueMask);
    Reg::Cast<uint32_t, uint16_t, BITONIC_SMALL_CAST_U16_TO_U32_ODD>(oddValue, rawValue, valueMask);
    Reg::Interleave<uint32_t>(valueBits, unusedValue, evenValue, oddValue);
}

/*!
 * \brief 将 32 位位模式寄存器还原为 16 位数据并存回 UB。
 *
 * 这是 BitonicSmallRegLoadB16Bits 的逆操作：把每通道一个 16 位元素位模式的
 * uint32 寄存器，重新打包为 16 位连续存储后写回 UB。
 *
 *
 * \param[out] valueAddr  UB 目标地址
 * \param[in]  valueBits  排序后的 32 位位模式寄存器
 * \param[in]  validCount 有效元素个数
 * \param[in]  validMask  有效通道掩码
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegStoreB16Bits(__ubuf__ T* valueAddr, Reg::RegTensor<uint32_t>& valueBits,
                                                        uint32_t validCount, Reg::MaskReg& validMask)
{
    uint32_t valueCount = validCount;
    Reg::MaskReg valueMask = Reg::UpdateMask<T>(valueCount);
    Reg::RegTensor<uint16_t> rawValue;
    Reg::Cast<uint16_t, uint32_t, BITONIC_SMALL_CAST_U32_TO_U16>(rawValue, valueBits, validMask);
    Reg::Pack(rawValue, (Reg::RegTensor<uint32_t>&)rawValue);
    Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY>(valueAddr, (Reg::RegTensor<T>&)rawValue, 1U, valueMask);
}

/*!
 * \brief 浮点类型构建单调有序 key (±0 归一后，NaN 覆盖前) (Reg/SIMD 路径)。
 *
 * ±0 位模式先归一为 +0，避免 -0 翻转后越过普通负数；负数全位翻转、
 * 正数仅翻符号位，使 uint32 无符号比较与浮点序一致。
 * absoluteBits 输出原始绝对值位模式，供 NaN 哨兵覆盖段复用。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegBuildFloatOrderedKey32(Reg::RegTensor<uint32_t>& key,
                                                                  Reg::RegTensor<uint32_t>& rawBits,
                                                                  Reg::RegTensor<uint32_t>& absoluteBits,
                                                                  Reg::RegTensor<uint32_t>& allBitsReg,
                                                                  uint32_t signBit, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> exponentBits;
    Reg::RegTensor<uint32_t> positiveXor;
    Reg::RegTensor<uint32_t> xorMask;
    Reg::RegTensor<uint32_t> signMask;
    Reg::RegTensor<uint32_t> zeroReg;
    Reg::MaskReg zeroMask;
    Reg::MaskReg negativeMask;
    Reg::Duplicate(zeroReg, 0U);
    Reg::Duplicate(signMask, signBit);
    Reg::And(absoluteBits, rawBits, allBitsReg, activeMask);
    Reg::And(exponentBits, rawBits, signMask, activeMask);
    Reg::Xor(absoluteBits, rawBits, exponentBits, activeMask);
    Reg::Compares<uint32_t, CMPMODE::EQ>(zeroMask, absoluteBits, 0U, activeMask);
    Reg::Select<uint32_t>(rawBits, zeroReg, rawBits, zeroMask);
    Reg::And(exponentBits, rawBits, signMask, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(negativeMask, exponentBits, 0U, activeMask);
    Reg::Duplicate(positiveXor, signBit);
    Reg::Select<uint32_t>(xorMask, allBitsReg, positiveXor, negativeMask);
    Reg::Xor(key, rawBits, xorMask, activeMask);
    if constexpr (!IsLargest) {
        Reg::Xor(key, key, allBitsReg, activeMask);
    }
}

/*!
 * \brief 浮点类型 NaN 哨兵覆盖 (Reg/SIMD 路径)。
 *
 * 指数位全 1 且尾数非 0 判定为 NaN，key 覆盖为序边界哨兵
 * (IsLargest 取 allBits，否则取 1)，保证 NaN 与任意数值可比。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegOverwriteNanKey32(Reg::RegTensor<uint32_t>& key,
                                                             Reg::RegTensor<uint32_t>& absoluteBits, uint32_t allBits,
                                                             Reg::MaskReg& activeMask)
{
    constexpr uint32_t exponentMask = std::is_same_v<T, half> ? 0x7c00U :
                                                                (std::is_same_v<T, bfloat16_t> ? 0x7f80U : 0x7f800000U);
    constexpr uint32_t fractionMask = std::is_same_v<T, half> ? 0x03ffU :
                                                                (std::is_same_v<T, bfloat16_t> ? 0x007fU : 0x007fffffU);
    Reg::RegTensor<uint32_t> exponentBits;
    Reg::RegTensor<uint32_t> fractionBits;
    Reg::RegTensor<uint32_t> nanKey;
    Reg::RegTensor<uint32_t> signMask;
    Reg::MaskReg exponentMaskReg;
    Reg::MaskReg fractionMaskReg;
    Reg::MaskReg nanMask;
    Reg::Duplicate(signMask, exponentMask);
    Reg::And(exponentBits, absoluteBits, signMask, activeMask);
    Reg::Duplicate(signMask, fractionMask);
    Reg::And(fractionBits, absoluteBits, signMask, activeMask);
    Reg::Compares<uint32_t, CMPMODE::EQ>(exponentMaskReg, exponentBits, exponentMask, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(fractionMaskReg, fractionBits, 0U, activeMask);
    Reg::And(nanMask, exponentMaskReg, fractionMaskReg, activeMask);
    Reg::Duplicate(nanKey, IsLargest ? allBits : 1U);
    Reg::Select<uint32_t>(key, nanKey, key, nanMask);
}

/*!
 * \brief 整数类型构建单调有序 key (Reg/SIMD 路径)。
 *
 * 有符号类型翻转符号位，使 uint32 无符号比较与有符号序一致；
 * 求最小值时再整体取反，统一为最大值语义的降序 key。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegBuildIntegerKey32(Reg::RegTensor<uint32_t>& key,
                                                             Reg::RegTensor<uint32_t>& rawBits,
                                                             Reg::RegTensor<uint32_t>& allBitsReg, uint32_t signBit,
                                                             Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> xorMask;
    Reg::Duplicate(xorMask, std::is_signed_v<T> ? signBit : 0U);
    Reg::Xor(key, rawBits, xorMask, activeMask);
    if constexpr (!IsLargest) {
        Reg::Xor(key, key, allBitsReg, activeMask);
    }
}

/*!
 * \brief 将任意类型数据的位模式转换为可单调比较的 32 位无符号整数 key。
 *
 * 按 dtype 位模式分支：浮点类型走有序 key 构建 + NaN 哨兵覆盖，
 * 整数类型走符号位翻转，分支语义与拆分前完全一致。
 *
 * \param[out] key        生成的 32 位排序 key
 * \param[in]  valueBits  原始数据的 32 位位模式
 * \param[in]  activeMask 活跃通道掩码
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegBuildKey32(Reg::RegTensor<uint32_t>& key,
                                                      Reg::RegTensor<uint32_t>& valueBits, Reg::MaskReg& activeMask)
{
    constexpr uint32_t bitWidth = sizeof(T) * 8U;
    constexpr uint32_t signBit = 1U << (bitWidth - 1U);
    constexpr uint32_t allBits = sizeof(T) == sizeof(uint32_t) ? UINT32_MAX : ((1U << bitWidth) - 1U);
    Reg::RegTensor<uint32_t> rawBits;
    Reg::RegTensor<uint32_t> allBitsReg;
    Reg::Duplicate(allBitsReg, allBits);
    Reg::And(rawBits, valueBits, allBitsReg, activeMask);
    if constexpr (IsBitonicFloatType<T>) {
        Reg::RegTensor<uint32_t> absoluteBits;
        BitonicSmallRegBuildFloatOrderedKey32<T, IsLargest>(key, rawBits, absoluteBits, allBitsReg, signBit,
                                                            activeMask);
        BitonicSmallRegOverwriteNanKey32<T, IsLargest>(key, absoluteBits, allBits, activeMask);
    } else {
        BitonicSmallRegBuildIntegerKey32<T, IsLargest>(key, rawBits, allBitsReg, signBit, activeMask);
    }
}

} // namespace topkV2

#endif // TOP_K_SMALL_BITONIC_COMMON_H
