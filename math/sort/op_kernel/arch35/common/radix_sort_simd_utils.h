/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_SIMD_UTILS_H
#define RADIX_SORT_SIMD_UTILS_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"
#include "radix_sort_constants.h"

namespace RadixSortCommon {

using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

// Radix preprocessing maps signed/floating values to unsigned sortable keys.
// Floating-point -0.0 and +0.0 must be normalized to the same key. This path currently twiddles input on every radix
// round, so the normalization lives in TwiddleInFp16/TwiddleInFp32.

template <typename UT, uint64_t isDescend>
__aicore__ inline void Twiddle(uint16_t repeatTime, uint32_t vfLen, uint32_t inputNum, RegTensor<UT>& xorReg,
                               __local_mem__ UT* xValuePtr, __local_mem__ UT* uXValuePtr)
{
    MicroAPI::MaskReg xorMask;
    MicroAPI::RegTensor<UT> inputReg;
    MicroAPI::RegTensor<UT> vnotReg;
    MicroAPI::RegTensor<UT> xorResult;
    for (uint16_t i = 0; i < repeatTime; i++) {
        xorMask = MicroAPI::UpdateMask<UT>(inputNum);
        MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, vfLen);
        MicroAPI::Xor(xorResult, inputReg, xorReg, xorMask);
        if constexpr (isDescend == 0) {
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, xorResult, vfLen, xorMask);
        } else {
            MicroAPI::Not(vnotReg, xorResult, xorMask);
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, vfLen, xorMask);
        }
    }
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB32(LocalTensor<T1> inputX, LocalTensor<UT> uInputX, uint32_t numTileData)
{
    __local_mem__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __local_mem__ UT* uXValuePtr = (__ubuf__ UT*)uInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B32);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorValue;
        MicroAPI::MaskReg xorMask;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorValue, XOR_OP_VALUE, maskB32);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B32, inputNum, xorValue, xValuePtr, uXValuePtr);
    }
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB16(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __local_mem__ UT* uXValuePtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B16);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorReg;
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorReg, XOR_OP_VALUE_B16, maskB16);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B16, inputNum, xorReg, xValuePtr, uXValuePtr);
    }
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB8(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __local_mem__ UT* uXValuePtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B8);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorReg;
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorReg, XOR_OP_VALUE_B8, maskB8);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B8, inputNum, xorReg, xValuePtr, uXValuePtr);
    }
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB64(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __local_mem__ UT* uXValuePtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B64);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorReg;
        MicroAPI::MaskReg predicateDefaultB64 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorReg, XOR_OP_VALUE_B64, predicateDefaultB64);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B64, inputNum, xorReg, xValuePtr, uXValuePtr);
    }
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInFp16(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ uint16_t* xValuePtr = (__ubuf__ uint16_t*)inputX.GetPhyAddr();
    __local_mem__ uint16_t* uXValuePtr = (__ubuf__ uint16_t*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B16);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> inputReg, vnotReg;
        MicroAPI::RegTensor<uint16_t> xorMaskReg, vandMask;
        MicroAPI::RegTensor<uint16_t> twiddledZeroReg;
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg xorMask;
        MicroAPI::Duplicate(xorMaskReg, LOWEST_KEY_VALUE_B16, maskB16);
        MicroAPI::Duplicate(vandMask, XOR_OP_VALUE_B16, maskB16);
        MicroAPI::Duplicate(twiddledZeroReg, TWIDDLED_ZERO_BITS_FP16, maskB16);

        for (uint16_t i = 0; i < repeatTime; i++) {
            xorMask = MicroAPI::UpdateMask<uint16_t>(inputNum);
            // load input
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, VF_LEN_B16);
            // vand
            MicroAPI::RegTensor<uint16_t> andValueOne;
            MicroAPI::And(andValueOne, inputReg, vandMask, maskB16);
            // not equal
            MicroAPI::MaskReg cmpValueOne;
            MicroAPI::CompareScalar<uint16_t, CMPMODE::NE>(cmpValueOne, andValueOne, ZERO_VALUE_FLAG_B16, maskB16);
            // vsel
            MicroAPI::RegTensor<uint16_t> finalMaskOne;
            MicroAPI::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
            // vxor
            MicroAPI::RegTensor<uint16_t> xorVectorOne;
            MicroAPI::Xor(xorVectorOne, inputReg, finalMaskOne, maskB16);

            // 目前每个round都会做twiddleIn，因此直接在twiddleIn方法将负0转换为正0
            // 如果后期需要改为单次twiddleIn，则需要在提取位数时将负0转换为正0
            // get -0.0 mask
            MicroAPI::MaskReg minusZeroMask;
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(minusZeroMask, xorVectorOne, TWIDDLED_MINUS_ZERO_BITS_FP16,
                                                           maskB16);
            // change -0.0 to +0.0
            MicroAPI::RegTensor<uint16_t> resultReg;
            MicroAPI::Select(resultReg, twiddledZeroReg, xorVectorOne, minusZeroMask);

            if constexpr (isDescend == 0) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, resultReg, VF_LEN_B16,
                                                                                      xorMask);
            } else {
                MicroAPI::Not(vnotReg, resultReg, xorMask);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, VF_LEN_B16,
                                                                                      xorMask);
            }
        }
    }
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInFp32(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ uint32_t* xValuePtr = (__ubuf__ uint32_t*)inputX.GetPhyAddr();
    __local_mem__ uint32_t* uXValuePtr = (__ubuf__ uint32_t*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B32);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> inputReg, vnotReg;
        MicroAPI::RegTensor<uint32_t> xorMaskReg, vandMask;
        MicroAPI::RegTensor<uint32_t> twiddledZeroReg;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg xorMask;
        MicroAPI::Duplicate(xorMaskReg, LOWEST_KEY_VALUE_B32, maskB32);
        MicroAPI::Duplicate(vandMask, XOR_OP_VALUE, maskB32);
        MicroAPI::Duplicate(twiddledZeroReg, TWIDDLED_ZERO_BITS_FP32, maskB32);
        for (uint16_t i = 0; i < repeatTime; i++) {
            xorMask = MicroAPI::UpdateMask<uint32_t>(inputNum);
            // load input
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, VF_LEN_B32);
            // vand
            MicroAPI::RegTensor<uint32_t> andValueOne;
            MicroAPI::And(andValueOne, inputReg, vandMask, maskB32);
            // not equal
            MicroAPI::MaskReg cmpValueOne;
            MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(cmpValueOne, andValueOne, ZERO_VALUE_FLAG_B32, xorMask);
            // vsel
            MicroAPI::RegTensor<uint32_t> finalMaskOne;
            MicroAPI::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
            // vxor
            MicroAPI::RegTensor<uint32_t> xorVectorZero;
            MicroAPI::Xor(xorVectorZero, inputReg, finalMaskOne, maskB32);

            // 目前每个round都会做twiddleIn，因此直接在twiddleIn方法将负0转换为正0
            // 如果后期需要改为单次twiddleIn，则需要在提取位数时将负0转换为正0
            // get -0.0 mask
            MicroAPI::MaskReg minusZeroMask;
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(minusZeroMask, xorVectorZero, TWIDDLED_MINUS_ZERO_BITS_FP32,
                                                           maskB32);
            // change -0.0 to +0.0
            MicroAPI::RegTensor<uint32_t> resultReg;
            MicroAPI::Select(resultReg, twiddledZeroReg, xorVectorZero, minusZeroMask);

            if constexpr (isDescend == 0) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, resultReg, VF_LEN_B32,
                                                                                      xorMask);
            } else {
                MicroAPI::Not(vnotReg, resultReg, maskB32);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, VF_LEN_B32,
                                                                                      xorMask);
            }
        }
    }
}

template <typename T1, typename UT>
__aicore__ inline void ReverseInputData(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT* inputXValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __local_mem__ UT* reverseInputXPtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(UT);
    uint16_t repeatTime = CeilDivision(numTileData, vfLen);
    uint32_t inputElementNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> inputVectorOne;
        MicroAPI::RegTensor<UT> vnotVectorZero;
        MicroAPI::MaskReg predicateDefaultB8 = MicroAPI::CreateMask<UT>();
        for (uint16_t i = 0; i < repeatTime; i++) {
            MicroAPI::MaskReg vnotMask = MicroAPI::UpdateMask<UT>(inputElementNum);
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr, vfLen);
            MicroAPI::Not(vnotVectorZero, inputVectorOne, predicateDefaultB8);
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reverseInputXPtr, vnotVectorZero, vfLen,
                                                                            vnotMask);
        }
    }
}

} // namespace RadixSortCommon

#endif
