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
 * \file radix_more_core_base.h
 * \brief Radix sort multi-core infrastructure: free functions (histogram, scatter, copy)
 *        and CRTP base class for sort and kth_value kernels.
 *        Twiddle preprocessing is in radix_sort_simd_utils.h; constants in radix_sort_constants.h.
 */

#ifndef RADIX_MORE_CORE_BASE_H
#define RADIX_MORE_CORE_BASE_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"
#include "radix_sort_constants.h"
#include "radix_sort_simd_utils.h"

namespace RadixSortCommon {

using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T1>
__aicore__ inline void CopyDataIn(GlobalTensor<T1> inputX, LocalTensor<T1> xLocal, uint64_t tileOffset,
                                  uint32_t currTileSize)
{
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T1)) / sizeof(T1);
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T1);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
}

template <typename T3>
__aicore__ inline void ComputeSumChist(RegTensor<uint16_t>& chist0, RegTensor<uint16_t>& chist1,
                                       RegTensor<uint16_t>& hist0, RegTensor<uint16_t>& hist1, MaskReg& maskB16,
                                       MaskReg& maskB32, __local_mem__ T3* blockExcusiveUbRPtr,
                                       __local_mem__ T3* blockExcusiveUbWPtr, __local_mem__ uint16_t* histUbPtr,
                                       __local_mem__ uint16_t* histCumsumUbPtr)
{
    // chist is inclusive per-tile cumulative histogram. Subtract the current bin count to get the per-bin exclusive
    // offset used by the later scatter phase.
    MicroAPI::RegTensor<uint16_t> excusiveSumZero, excusiveSumOne, zeroReg;
    MicroAPI::Sub(excusiveSumZero, chist0, hist0, maskB16);
    MicroAPI::Sub(excusiveSumOne, chist1, hist1, maskB16);

    // Persist each tile's histogram and exclusive cumsum. ComputeOnePass reloads these after global bin offsets have
    // been accumulated across tiles/cores.
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist0, VF_LEN_B16, maskB16);
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist1, VF_LEN_B16, maskB16);
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumUbPtr, excusiveSumZero, VF_LEN_B16,
                                                                          maskB16);
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumUbPtr, excusiveSumOne, VF_LEN_B16,
                                                                          maskB16);

    MicroAPI::Duplicate(zeroReg, 0, maskB16);
    MicroAPI::RegTensor<uint32_t> sum0, sum1, sum2, sum3;
    MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)sum0, (MicroAPI::RegTensor<uint16_t>&)sum1, excusiveSumZero,
                         zeroReg);
    MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)sum2, (MicroAPI::RegTensor<uint16_t>&)sum3, excusiveSumOne,
                         zeroReg);
    if constexpr (sizeof(T3) == sizeof(uint32_t)) {
        MicroAPI::RegTensor<uint32_t> sumIn0, sumIn1, sumIn2, sumIn3;
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn0, blockExcusiveUbRPtr, VF_LEN_B32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn1, blockExcusiveUbRPtr, VF_LEN_B32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn2, blockExcusiveUbRPtr, VF_LEN_B32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn3, blockExcusiveUbRPtr, VF_LEN_B32);
        // Accumulate the current tile's exclusive cumsum into this core's block-level bin totals.
        MicroAPI::Add(sumIn0, sumIn0, sum0, maskB32);
        MicroAPI::Add(sumIn1, sumIn1, sum1, maskB32);
        MicroAPI::Add(sumIn2, sumIn2, sum2, maskB32);
        MicroAPI::Add(sumIn3, sumIn3, sum3, maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn0, VF_LEN_B32,
                                                                              maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn1, VF_LEN_B32,
                                                                              maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn2, VF_LEN_B32,
                                                                              maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn3, VF_LEN_B32,
                                                                              maskB32);
    } else {
        MicroAPI::MaskReg maskB64 = MicroAPI::CreateMask<int64_t>();
        MicroAPI::RegTensor<int64_t> sum0Int64, sum1Int64, sum2Int64, sum3Int64;
        MicroAPI::RegTensor<int64_t> sum4Int64, sum5Int64, sum6Int64, sum7Int64;
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t>&)sum0Int64, (MicroAPI::RegTensor<uint32_t>&)sum1Int64, sum0,
                             (MicroAPI::RegTensor<uint32_t>&)zeroReg);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t>&)sum2Int64, (MicroAPI::RegTensor<uint32_t>&)sum3Int64, sum1,
                             (MicroAPI::RegTensor<uint32_t>&)zeroReg);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t>&)sum4Int64, (MicroAPI::RegTensor<uint32_t>&)sum5Int64, sum2,
                             (MicroAPI::RegTensor<uint32_t>&)zeroReg);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t>&)sum6Int64, (MicroAPI::RegTensor<uint32_t>&)sum7Int64, sum3,
                             (MicroAPI::RegTensor<uint32_t>&)zeroReg);
        MicroAPI::RegTensor<int64_t> int64In0, int64In1, int64In2, int64In3;
        MicroAPI::RegTensor<int64_t> int64In4, int64In5, int64In6, int64In7;
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In0, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In1, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In2, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In3, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In4, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In5, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In6, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In7, blockExcusiveUbRPtr, VF_LEN_B64);
        // Accumulate the current tile's exclusive cumsum into this core's block-level bin totals.
        MicroAPI::Add(int64In0, int64In0, sum0Int64, maskB64);
        MicroAPI::Add(int64In1, int64In1, sum1Int64, maskB64);
        MicroAPI::Add(int64In2, int64In2, sum2Int64, maskB64);
        MicroAPI::Add(int64In3, int64In3, sum3Int64, maskB64);
        MicroAPI::Add(int64In4, int64In4, sum4Int64, maskB64);
        MicroAPI::Add(int64In5, int64In5, sum5Int64, maskB64);
        MicroAPI::Add(int64In6, int64In6, sum6Int64, maskB64);
        MicroAPI::Add(int64In7, int64In7, sum7Int64, maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In0, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In1, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In2, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In3, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In4, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In5, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In6, VF_LEN_B64,
                                                                             maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In7, VF_LEN_B64,
                                                                             maskB64);
    }
}

template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB32(LocalTensor<UT>& inputX, LocalTensor<T3>& blockExcusiveUb,
                                               LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& histCumsumUb,
                                               LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint32_t* inXPtr = (__ubuf__ uint32_t*)inputX.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbWPtr = (__ubuf__ T3*)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __local_mem__ uint16_t* histCumsumUbPtr = (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr();
    __local_mem__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> in0, in1, in2, in3;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::MaskReg histMask;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // Load four B32 vectors, shift out the current radix byte, then compact it to B8 for histogram/scatter.
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B32);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in1, inXPtr, VF_LEN_B32);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in2, inXPtr, VF_LEN_B32);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in3, inXPtr, VF_LEN_B32);
            MicroAPI::RegTensor<uint32_t> shift0, shift1, shift2, shift3;
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift0, in0, bitOffset, maskB32);
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift1, in1, bitOffset, maskB32);
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift2, in2, bitOffset, maskB32);
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift3, in3, bitOffset, maskB32);
            MicroAPI::RegTensor<uint16_t> deInter0, deInter1, deInter2, deInter3;
            MicroAPI::DeInterleave(deInter0, deInter1, (MicroAPI::RegTensor<uint16_t>&)shift0,
                                   (MicroAPI::RegTensor<uint16_t>&)shift1);
            MicroAPI::DeInterleave(deInter2, deInter3, (MicroAPI::RegTensor<uint16_t>&)shift2,
                                   (MicroAPI::RegTensor<uint16_t>&)shift3);

            MicroAPI::RegTensor<uint8_t> deInter0B8, deInter1B8;
            MicroAPI::DeInterleave(deInter0B8, deInter1B8, (MicroAPI::RegTensor<uint8_t>&)deInter0,
                                   (MicroAPI::RegTensor<uint8_t>&)deInter2);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8,
                                                                                 histMask);
            // Build both the frequency histogram and the inclusive cumulative histogram for this tile.
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
                            histUbPtr, histCumsumUbPtr);
    }
}

template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB64(LocalTensor<UT>& inputX, LocalTensor<T3>& blockExcusiveUb,
                                               LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& histCumsumUb,
                                               LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint64_t* inXPtr = (__ubuf__ uint64_t*)inputX.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbWPtr = (__ubuf__ T3*)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __local_mem__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    __local_mem__ uint16_t* histCumsumUbPtr = (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint64_t> in0, in1, in2, in3, in4, in5, in6, in7;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::MaskReg histMask;
        MicroAPI::MaskReg maskB64 = MicroAPI::CreateMask<uint64_t>();
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // Load eight B64 vectors, shift out the current radix byte, then compact it to B8 for histogram/scatter.
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in1, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in2, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in3, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in4, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in5, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in6, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in7, inXPtr, VF_LEN_B64);
            MicroAPI::RegTensor<uint64_t> shift0, shift1, shift2, shift3, shift4, shift5, shift6, shift7;
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift0, in0, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift1, in1, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift2, in2, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift3, in3, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift4, in4, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift5, in5, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift6, in6, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift7, in7, bitOffset, maskB64);
            MicroAPI::RegTensor<uint32_t> deInter0, deInter1, deInter2, deInter3, deInter4, deInter5, deInter6,
                deInter7;
            MicroAPI::DeInterleave(deInter0, deInter1, (MicroAPI::RegTensor<uint32_t>&)shift0,
                                   (MicroAPI::RegTensor<uint32_t>&)shift1);
            MicroAPI::DeInterleave(deInter2, deInter3, (MicroAPI::RegTensor<uint32_t>&)shift2,
                                   (MicroAPI::RegTensor<uint32_t>&)shift3);
            MicroAPI::DeInterleave(deInter4, deInter5, (MicroAPI::RegTensor<uint32_t>&)shift4,
                                   (MicroAPI::RegTensor<uint32_t>&)shift5);
            MicroAPI::DeInterleave(deInter6, deInter7, (MicroAPI::RegTensor<uint32_t>&)shift6,
                                   (MicroAPI::RegTensor<uint32_t>&)shift7);
            MicroAPI::RegTensor<uint16_t> deInter0B16, deInter1B16, deInter2B16, deInter3B16;
            MicroAPI::DeInterleave(deInter0B16, deInter1B16, (MicroAPI::RegTensor<uint16_t>&)deInter0,
                                   (MicroAPI::RegTensor<uint16_t>&)deInter2);
            MicroAPI::DeInterleave(deInter2B16, deInter3B16, (MicroAPI::RegTensor<uint16_t>&)deInter4,
                                   (MicroAPI::RegTensor<uint16_t>&)deInter6);
            MicroAPI::RegTensor<uint8_t> deInter0B8, deInter1B8;
            MicroAPI::DeInterleave(deInter0B8, deInter1B8, (MicroAPI::RegTensor<uint8_t>&)deInter0B16,
                                   (MicroAPI::RegTensor<uint8_t>&)deInter2B16);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8,
                                                                                 histMask);
            // Build both the frequency histogram and the inclusive cumulative histogram for this tile.
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
                            histUbPtr, histCumsumUbPtr);
    }
}

template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB16(LocalTensor<UT>& inputX, LocalTensor<T3>& blockExcusiveUb,
                                               LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& histCumsumUb,
                                               LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint16_t* inputXValuePtr = (__ubuf__ uint16_t*)inputX.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbWPtr = (__ubuf__ T3*)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __local_mem__ uint16_t* histCumsumUbPtr = (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr();
    __local_mem__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg histMask;
        MicroAPI::RegTensor<uint16_t> in0, in1;
        MicroAPI::RegTensor<uint16_t> shift0, shift1;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // Load two B16 vectors, shift out the current radix byte, then compact it to B8 for histogram/scatter.
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inputXValuePtr, VF_LEN_B16);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in1, inputXValuePtr, VF_LEN_B16);
            MicroAPI::ShiftRights<uint16_t, int16_t>(shift0, in0, bitOffset, maskB16);
            MicroAPI::ShiftRights<uint16_t, int16_t>(shift1, in1, bitOffset, maskB16);
            MicroAPI::RegTensor<uint8_t> deInter0B8, deInter1B8;
            MicroAPI::DeInterleave(deInter0B8, deInter1B8, (MicroAPI::RegTensor<uint8_t>&)shift0,
                                   (MicroAPI::RegTensor<uint8_t>&)shift1);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8,
                                                                                 histMask);
            // Build both the frequency histogram and the inclusive cumulative histogram for this tile.
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
                            histUbPtr, histCumsumUbPtr);
    }
}

template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB8(LocalTensor<UT>& inputX, LocalTensor<T3>& blockExcusiveUb,
                                              LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& histCumsumUb,
                                              LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint8_t* inXPtr = (__ubuf__ uint8_t*)inputX.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbWPtr = (__ubuf__ T3*)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3* blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __local_mem__ uint16_t* histCumsumUbPtr = (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr();
    __local_mem__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg histMask;
        MicroAPI::RegTensor<uint8_t> in0;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::RegTensor<uint16_t> chistVectorZero, chistVectorOne, zeroReg;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // B8 input already is the radix byte for this round; save it and build histogram/cumsum directly.
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, in0, VF_LEN_B8,
                                                                                 histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist0, in0, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::FREQUENCY>(hist1, in0, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist0, in0, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(chist1, in0, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
                            histUbPtr, histCumsumUbPtr);
    }
}

template <typename T3>
__simt_vf__ LAUNCH_BOUND(RADIX_SORT_NUM) __aicore__
    void SimtGlobalOffset(uint64_t excusiveBinOffset, __gm__ T3* excusiveBinsGm, __ubuf__ T3* blockExcusiveBuffer,
                          __gm__ T3* outGM_)
{
    for (int32_t i = threadIdx.x; i < RADIX_SORT_NUM; i += RADIX_SORT_NUM) {
        int32_t offset = i;
        T3 srcData = blockExcusiveBuffer[offset];
        asc_atomic_add(excusiveBinsGm + excusiveBinOffset + offset, srcData);
    }
}

template <typename T>
__aicore__ inline void CopyGlobalDataIn(GlobalTensor<T> inputX, LocalTensor<T>& xLocal, uint32_t tileOffset,
                                        uint32_t currTileSize)
{
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T)) / sizeof(T);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
}

template <typename T>
__aicore__ inline void CopyIndexDataIn(GlobalTensor<uint32_t> inputIndex, LocalTensor<uint32_t>& xLocal,
                                       uint64_t tileOffset, uint32_t currTileSize)
{
    DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputIndex[tileOffset], dataCopyParam, padParams);
}

template <typename T1, typename T2, typename T3, typename IdxT, int32_t round>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__
    void CopyOutGm(T3 tileDataStart, uint32_t cureTileSize, uint64_t outputXUnsortedAxisOffset, uint64_t unSortIdOffset,
                   __ubuf__ uint16_t* blockExcusiveSumAddr, __gm__ volatile T3* excusiveBinsGmAddr,
                   __ubuf__ T3* blockDataInGlobalPosAddr, __ubuf__ uint32_t* sortedIndexLocalAddr,
                   __ubuf__ T3* xInputIndexLocalAddr, __ubuf__ uint8_t* sortedValueLocalAddr,
                   __ubuf__ T1* xInputValueLocalAddr, __ubuf__ T3* blockHistFlagAddr, __ubuf__ uint16_t* blockHistAddr,
                   __gm__ volatile IdxT* indexDoubleBufferGmAddr, __gm__ volatile T1* inputXDoubleBufferAddr)
{
    for (int i = threadIdx.x; i < RADIX_SORT_NUM; i += THREAD_DIM_NUM) {
        // Per-key scatter base:
        //   globalKeyOffsetVal   = number of all previous radix keys in this pass
        //   blockHistCumsumVal   = number of this key in previous tiles, after lookback
        //   blockHistVal         = number of this key in the current tile
        //   blockExcusiveSumVal  = number of smaller positions inside the current tile for this key
        // The result is the starting GM position for key i in this tile; the per-element loop below adds its
        // in-key sorted rank.
        T3 blockHistCumsumVal = blockHistFlagAddr[i];
        if constexpr (IsSameType<T3, uint32_t>::value) {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK;
        } else {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK_B64;
        }
        uint32_t blockExcusiveSumVal = blockExcusiveSumAddr[i];
        uint32_t blockHistVal = blockHistAddr[i];
        T3 globalKeyOffsetVal = excusiveBinsGmAddr[unSortIdOffset + i];
        T3 finalpos = globalKeyOffsetVal + blockHistCumsumVal - blockHistVal - blockExcusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    asc_syncthreads();
    for (int i = threadIdx.x; i < cureTileSize; i += THREAD_DIM_NUM) {
        T3 localDataIndex = static_cast<T3>(sortedIndexLocalAddr[i]);
        T3 dataInitIndex = 0;
        // Round 0 starts from the original tile-local index. Later rounds carry the original index through the
        // double-buffered index workspace.
        if constexpr (round != 0) {
            dataInitIndex = xInputIndexLocalAddr[localDataIndex];
        } else {
            dataInitIndex = tileDataStart + localDataIndex;
        }
        T3 dataFinalGlobalPos = blockDataInGlobalPosAddr[sortedValueLocalAddr[i]] + i;
        inputXDoubleBufferAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = xInputValueLocalAddr[localDataIndex];
        indexDoubleBufferGmAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = static_cast<IdxT>(dataInitIndex);
    }
}

// ======================== CRTP Base Class ========================

/**
 * @brief CRTP base class for radix_more_core kernels.
 *        Contains shared workspace tensors, radix preprocessing, histogram/prefix-sum, and copy helpers used by
 *        sort and kth_value. Derived classes provide operator-specific initialization, scheduling, and output
 *        extraction.
 * @tparam Derived CRTP derived type
 * @tparam T1 Input/output storage data type
 * @tparam T2 Output index data type used by derived kernels
 * @tparam UT Unsigned/key representation used by radix preprocessing
 * @tparam T3 Workspace counter type for histogram and exclusive-prefix data
 * @tparam isDescend Sort order flag: 1 for descending, 0 for ascending
 */
template <typename Derived, typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
class RadixMoreCoreBase {
public:
    GlobalTensor<T1> inputXGm_;
    GlobalTensor<T1> outValueGm_;
    GlobalTensor<uint32_t> outIdxGm_;
    GlobalTensor<T1> outValueDbWK_;
    GlobalTensor<uint32_t> outIdxDbWK_;
    GlobalTensor<uint8_t> xB8GmWk_;
    GlobalTensor<uint16_t> histTileGmWk_;
    GlobalTensor<uint16_t> histCumsumTileGmWk_;
    GlobalTensor<uint32_t> globalHistGmWk_;
    GlobalTensor<T3> globalHistGmWkTmp_;
    GlobalTensor<uint32_t> excusiveBinsGmWk_;
    GlobalTensor<T3> excusiveBinsGmWkTmp_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueIndex_;
    TQue<QuePosition::VECIN, 1> inQueueGlobalHist_;
    TQue<QuePosition::VECIN, 1> blockExcusiveInQue_;
    TQue<QuePosition::VECIN, 1> blockHistInQue_;
    TBuf<TPosition::VECCALC> tmpUb_;
    TQue<QuePosition::VECOUT, 1> blockUbFlagQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputB8Que_;
    TQue<QuePosition::VECOUT, 1> outIdxQueue_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    TQue<QuePosition::VECOUT, 1> blockHistFlagUbQue_;
    DoubleBufferSimd<T1> inputXDbGm_;
    DoubleBufferSimd<uint32_t> idxDbGm_;
    static constexpr SortConfig sortConfigMuti{SortType::RADIX_SORT, false};
    int64_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    int64_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t lastDimTileNum_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t lastDimRealCore_ = 0;
    uint32_t tmpUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t realCoreNum_ = 0;
    uint32_t clearCore1_ = 0;
    uint32_t clearCore0_ = 0;
    uint32_t clearSize_ = 0;
    uint32_t clearCout_ = 0;
    uint32_t clearCoreSize0_ = 0;
    uint32_t clearCoreSize1_ = 0;
    uint32_t factor_ = 1;
    DataCopyExtParams copyParams{1, 1, 0, 0, 0};
    uint64_t oneBlock_ = Ops::Base::GetUbBlockSize();

    __aicore__ inline void ClearWorkSapce()
    {
        if (this->blockIdx_ < this->clearCore1_) {
            AscendC::LocalTensor<uint32_t> tmpUb = this->tmpUb_.template Get<uint32_t>();
            if constexpr (sizeof(T3) == sizeof(uint32_t)) {
                Duplicate(tmpUb, static_cast<uint32_t>(0), this->clearSize_);
            } else {
                Duplicate(tmpUb, static_cast<uint32_t>(0), this->clearSize_ * this->factor_);
            }
            event_t eventIdMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdMte3);
            if (this->blockIdx_ < this->clearCore0_) {
                this->copyParams.blockLen = this->clearCoreSize0_ * sizeof(T3);
                uint64_t offset = static_cast<uint64_t>(this->blockIdx_) * this->clearCoreSize0_ * this->factor_;
                DataCopyPad(this->excusiveBinsGmWk_[offset], tmpUb, this->copyParams);
                for (uint32_t i = 0; i < this->clearCout_; i++) {
                    uint64_t histGmOffset = (static_cast<uint64_t>(this->blockIdx_) * this->clearCoreSize1_ +
                                             static_cast<uint64_t>(i) * this->clearSize_) *
                                            this->factor_;
                    this->copyParams.blockLen = this->clearSize_ * sizeof(T3);
                    DataCopyPad(this->globalHistGmWk_[histGmOffset], tmpUb, this->copyParams);
                }
            } else {
                for (uint32_t i = 0; i < this->clearCout_; i++) {
                    uint64_t histGmOffset = (static_cast<uint64_t>(this->blockIdx_) * this->clearCoreSize1_ +
                                             static_cast<uint64_t>(i) * this->clearSize_) *
                                            this->factor_;
                    this->copyParams.blockLen = this->clearSize_ * sizeof(T3);
                    DataCopyPad(this->globalHistGmWk_[histGmOffset], tmpUb, this->copyParams);
                }
            }
        }
    }

    __aicore__ inline LocalTensor<UT> PreProcess(LocalTensor<T1> inputX, uint32_t numTileData)
    {
        LocalTensor<UT> inputXCopy = inputX.template ReinterpretCast<UT>();
        if constexpr (IsSameType<int32_t, T1>::value) {
            TwiddleInB32<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<half, T1>::value || IsSameType<bfloat16_t, T1>::value) {
            TwiddleInFp16<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<float, T1>::value) {
            TwiddleInFp32<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<int16_t, T1>::value) {
            TwiddleInB16<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<int8_t, T1>::value) {
            TwiddleInB8<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<int64_t, T1>::value) {
            TwiddleInB64<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else {
            if (isDescend == 1) {
                ReverseInputData<T1, UT>(inputX, inputXCopy, numTileData);
            }
        }
        return inputXCopy;
    }

    __aicore__ inline void PreGlobalExcusiveSum(LocalTensor<UT>& inputXCopy, LocalTensor<T3>& blockExcusiveUb,
                                                LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& histCumsumUb,
                                                LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round,
                                                uint32_t tileId)
    {
        // Extract the current radix byte, build the tile histogram, and update this core's per-bin exclusive totals.
        if constexpr (sizeof(T1) == sizeof(int32_t)) {
            GetGlobalExcusiveSumB32<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
                                            round);
        } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
            GetGlobalExcusiveSumB16<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
                                            round);
        } else if constexpr (sizeof(T1) == sizeof(int8_t)) {
            GetGlobalExcusiveSumB8<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
                                           round);
        } else if constexpr (sizeof(T1) == sizeof(int64_t)) {
            GetGlobalExcusiveSumB64<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
                                            round);
        }
    }

    __aicore__ inline void GetGlobalExcusiveSum(uint32_t round, uint32_t sortLoopRound, GlobalTensor<T1> inputX)
    {
        uint32_t startTileId = this->blockIdx_ % this->lastDimRealCore_;
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        uint64_t unsortedDimIndex = unSortId + static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_;
        int64_t xUnsortOffset = static_cast<int64_t>(unSortId) * this->totalDataNum_;
        if (unsortedDimIndex < static_cast<uint64_t>(this->unsortedDimNum_)) {
            LocalTensor<uint32_t> blockExcusiveUb = this->blockUbFlagQue_.template AllocTensor<uint32_t>();

            uint64_t excusiveBinOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) +
                                         round * RADIX_SORT_NUM;
            if constexpr (sizeof(T3) == sizeof(uint32_t)) {
                Duplicate(blockExcusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM);
            } else {
                Duplicate(blockExcusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM * RADIX_HIST_BUFFER_NUM);
            }
            for (uint32_t tileId = startTileId; tileId < this->lastDimTileNum_; tileId += this->lastDimRealCore_) {
                // tileOffset may exceed int32 range for large rows, so keep the address arithmetic in uint64.
                uint64_t tileOffset = static_cast<uint64_t>(tileId) * this->numTileData_;
                uint64_t remainTileDataNum = this->totalDataNum_ - tileOffset;
                if (this->totalDataNum_ < tileOffset) {
                    break;
                }
                uint32_t currTileSize = static_cast<uint32_t>(
                    remainTileDataNum < static_cast<uint64_t>(this->numTileData_) ? remainTileDataNum :
                                                                                    this->numTileData_);
                LocalTensor<T1> xLocal = this->inQueueX_.template AllocTensor<T1>();

                // Round 0 reads original input. Later rounds read the double-buffered output from the previous pass.
                if (round == 0) {
                    CopyDataIn<T1>(inputX[xUnsortOffset], xLocal, tileOffset, currTileSize);
                } else {
                    CopyDataIn<T1>(this->inputXDbGm_.Current()[xUnsortOffset], xLocal, tileOffset, currTileSize);
                }
                this->inQueueX_.EnQue(xLocal);
                xLocal = this->inQueueX_.template DeQue<T1>();
                // Convert signed/floating values to unsigned radix keys before byte extraction.
                LocalTensor<UT> xUbCopy = PreProcess(xLocal, currTileSize);
                LocalTensor<uint8_t> inputB8Ub = this->inputB8Que_.template AllocTensor<uint8_t>();
                LocalTensor<uint16_t> histUb = this->outIdxQueue_.template AllocTensor<uint16_t>();
                LocalTensor<uint16_t> histCumsumUb = this->outValueQueue_.template AllocTensor<uint16_t>();
                LocalTensor<T3> blockExcusiveUbTmp = blockExcusiveUb.template ReinterpretCast<T3>();
                PreGlobalExcusiveSum(xUbCopy, blockExcusiveUbTmp, histUb, histCumsumUb, inputB8Ub, currTileSize, round,
                                     tileId);
                this->inQueueX_.FreeTensor(xLocal);
                this->inputB8Que_.template EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(inputB8Ub);
                inputB8Ub = this->inputB8Que_.template DeQue<QuePosition::VECCALC, QuePosition::VECOUT, uint8_t>();
                // Save the extracted radix byte for the scatter phase to avoid recomputing it.
                this->copyParams.blockCount = 1;
                this->copyParams.blockLen = currTileSize * sizeof(uint8_t);
                this->copyParams.srcStride = 0;
                this->copyParams.dstStride = 0;
                DataCopyPad(this->xB8GmWk_[xUnsortOffset + tileOffset], inputB8Ub, this->copyParams);
                this->inputB8Que_.FreeTensor(inputB8Ub);

                this->outIdxQueue_.EnQue(histUb);
                histUb = this->outIdxQueue_.template DeQue<uint16_t>();
                // Save each tile's 256-bin histogram. Lookback uses these counts for inter-tile prefix offsets.
                this->copyParams.blockCount = 1;
                this->copyParams.blockLen = RADIX_SORT_NUM * sizeof(uint16_t);
                this->copyParams.srcStride = 0;
                this->copyParams.dstStride = 0;
                DataCopyPad(
                    this->histTileGmWk_[static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * this->lastDimTileNum_ +
                                        static_cast<uint64_t>(tileId) * RADIX_SORT_NUM],
                    histUb, this->copyParams);
                this->outIdxQueue_.FreeTensor(histUb);

                this->outValueQueue_.EnQue(histCumsumUb);
                histCumsumUb = this->outValueQueue_.template DeQue<uint16_t>();
                // Save each tile's intra-tile exclusive cumsum for the final scatter address calculation.
                DataCopyPad(
                    this->histCumsumTileGmWk_[static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * this->lastDimTileNum_ +
                                              static_cast<uint64_t>(tileId) * RADIX_SORT_NUM],
                    histCumsumUb, this->copyParams);
                this->outValueQueue_.FreeTensor(histCumsumUb);
            }
            this->blockUbFlagQue_.EnQue(blockExcusiveUb);
            blockExcusiveUb = this->blockUbFlagQue_.template DeQue<uint32_t>();
            if constexpr (sizeof(T3) == sizeof(uint32_t)) {
                this->copyParams.blockCount = 1;
                this->copyParams.blockLen = RADIX_SORT_NUM * sizeof(uint32_t);
                this->copyParams.srcStride = 0;
                this->copyParams.dstStride = 0;

                SetAtomicAdd<int32_t>();
                DataCopyPad(this->excusiveBinsGmWk_[excusiveBinOffset], blockExcusiveUb, this->copyParams);
                SetAtomicNone();
            } else {
                // int64 counters can exceed the int32 atomic path, so use the SIMT int64 atomic-add helper.
                asc_vf_call<SimtGlobalOffset<T3>>(
                    dim3(RADIX_SORT_NUM), excusiveBinOffset, (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                    (__ubuf__ T3*)(blockExcusiveUb.GetPhyAddr()), (__gm__ T3*)(this->outIdxGm_.GetPhyAddr()));
            }
            this->blockUbFlagQue_.FreeTensor(blockExcusiveUb);
        }
    }

    __aicore__ inline void ScatterBlockHist2Global(LocalTensor<uint16_t> blockHist, LocalTensor<T3> blockHistWithFlag,
                                                   GlobalTensor<T3> allblockHistToGm, uint32_t tileId, uint32_t round)
    {
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        int64_t unSortIdOffset = static_cast<int64_t>(unSortId) * this->lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) +
                                 static_cast<int64_t>(round) * RADIX_SORT_NUM * this->lastDimTileNum_;
        __local_mem__ uint16_t* blockHistPtr = (__ubuf__ uint16_t*)blockHist.GetPhyAddr();
        __local_mem__ T3* blockHistWithFlagPtr = (__ubuf__ T3*)blockHistWithFlag.GetPhyAddr();
        // Soft synchronization state is packed in the highest two bits of each bin value:
        //   0: not initialized, retry
        //   1: aggregate ready, this tile's local histogram can be accumulated
        //   2: prefix ready, accumulated prefix is complete and lookback can stop
        //   3: reserved
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
            MicroAPI::RegTensor<uint16_t> blockHistZero, blockHistOne;
            MicroAPI::RegTensor<uint16_t> lookaheadOutZero, lookaheadOutOne, lookaheadOutTwo, lookaheadOutThree;
            MicroAPI::RegTensor<uint16_t> zeroVector;
            MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistZero, blockHistPtr,
                                                                                  VF_LEN_B16);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistOne, blockHistPtr,
                                                                                  VF_LEN_B16);
            // Widen the uint16 histogram to T3 and set the aggregate-ready state bit before publishing to GM.
            MicroAPI::Interleave(lookaheadOutZero, lookaheadOutOne, blockHistZero, zeroVector);
            MicroAPI::Interleave(lookaheadOutTwo, lookaheadOutThree, blockHistOne, zeroVector);
            if constexpr (IsSameType<T3, uint32_t>::value) {
                MicroAPI::RegTensor<uint32_t> aggregateReadyMask;
                MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint32_t>();
                MicroAPI::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK, predicateDefault);
                MicroAPI::RegTensor<uint32_t> lookaheadOutZeroMask, lookaheadOutOneMask, lookaheadOutTwoMask,
                    lookaheadOutThreeMask;
                MicroAPI::Or(lookaheadOutZeroMask, (MicroAPI::RegTensor<uint32_t>&)lookaheadOutZero, aggregateReadyMask,
                             predicateDefault);
                MicroAPI::Or(lookaheadOutOneMask, (MicroAPI::RegTensor<uint32_t>&)lookaheadOutOne, aggregateReadyMask,
                             predicateDefault);
                MicroAPI::Or(lookaheadOutTwoMask, (MicroAPI::RegTensor<uint32_t>&)lookaheadOutTwo, aggregateReadyMask,
                             predicateDefault);
                MicroAPI::Or(lookaheadOutThreeMask, (MicroAPI::RegTensor<uint32_t>&)lookaheadOutThree,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutZeroMask, VF_LEN_B32, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutOneMask, VF_LEN_B32, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutTwoMask, VF_LEN_B32, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutThreeMask, VF_LEN_B32, predicateDefault);
            } else {
                MicroAPI::RegTensor<int64_t> aggregateReadyMask;
                MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<int64_t>();
                MicroAPI::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK_B64, predicateDefault);
                MicroAPI::RegTensor<uint16_t> lookaheadOutZeroB64A, lookaheadOutZeroB64B;
                MicroAPI::RegTensor<uint16_t> lookaheadOutOneB64A, lookaheadOutOneB64B;
                MicroAPI::RegTensor<uint16_t> lookaheadOutTwoB64A, lookaheadOutTwoB64B;
                MicroAPI::RegTensor<uint16_t> lookaheadOutThreeB64A, lookaheadOutThreeB64B;
                MicroAPI::Interleave(lookaheadOutZeroB64A, lookaheadOutZeroB64B, lookaheadOutZero, zeroVector);
                MicroAPI::Interleave(lookaheadOutOneB64A, lookaheadOutOneB64B, lookaheadOutOne, zeroVector);
                MicroAPI::Interleave(lookaheadOutTwoB64A, lookaheadOutTwoB64B, lookaheadOutTwo, zeroVector);
                MicroAPI::Interleave(lookaheadOutThreeB64A, lookaheadOutThreeB64B, lookaheadOutThree, zeroVector);
                MicroAPI::RegTensor<int64_t> lookaheadOutZeroMaskB64A, lookaheadOutZeroMaskB64B;
                MicroAPI::RegTensor<int64_t> lookaheadOutOneMaskB64A, lookaheadOutOneMaskB64B;
                MicroAPI::RegTensor<int64_t> lookaheadOutTwoMaskB64A, lookaheadOutTwoMaskB64B;
                MicroAPI::RegTensor<int64_t> lookaheadOutThreeMaskB64A, lookaheadOutThreeMaskB64B;
                MicroAPI::Or(lookaheadOutZeroMaskB64A, (MicroAPI::RegTensor<int64_t>&)lookaheadOutZeroB64A,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutZeroMaskB64B, (MicroAPI::RegTensor<int64_t>&)lookaheadOutZeroB64B,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutOneMaskB64A, (MicroAPI::RegTensor<int64_t>&)lookaheadOutOneB64A,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutOneMaskB64B, (MicroAPI::RegTensor<int64_t>&)lookaheadOutOneB64B,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutTwoMaskB64A, (MicroAPI::RegTensor<int64_t>&)lookaheadOutTwoB64A,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutTwoMaskB64B, (MicroAPI::RegTensor<int64_t>&)lookaheadOutTwoB64B,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutThreeMaskB64A, (MicroAPI::RegTensor<int64_t>&)lookaheadOutThreeB64A,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::Or(lookaheadOutThreeMaskB64B, (MicroAPI::RegTensor<int64_t>&)lookaheadOutThreeB64B,
                             aggregateReadyMask, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutZeroMaskB64A, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutZeroMaskB64B, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutOneMaskB64A, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutOneMaskB64B, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutTwoMaskB64A, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutTwoMaskB64B, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutThreeMaskB64A, VF_LEN_B64, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    blockHistWithFlagPtr, lookaheadOutThreeMaskB64B, VF_LEN_B64, predicateDefault);
            }
        }
        this->blockHistFlagUbQue_.EnQue(blockHistWithFlag);
        blockHistWithFlag = this->blockHistFlagUbQue_.template DeQue<T3>();
        if (tileId < (this->lastDimTileNum_ - 1)) {
            DataCopyExtParams dataCopyParam;
            dataCopyParam.blockCount = 1;
            dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(T3);
            dataCopyParam.srcStride = 0;
            dataCopyParam.dstStride = 0;
            // The last tile has no successor to look back from, so only non-last tiles publish their histogram state.
            DataCopyPad(allblockHistToGm[unSortIdOffset + RADIX_SORT_NUM * tileId], blockHistWithFlag, dataCopyParam);
        }
        this->blockHistFlagUbQue_.FreeTensor(blockHistWithFlag);
    }

    __aicore__ inline void LookbackGlobal(LocalTensor<T3> nowTileHistBuffer, GlobalTensor<T3> allTileHistBuffer,
                                          LocalTensor<uint32_t> ubFlagTensor, uint32_t tileId, uint32_t round)
    {
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        int64_t unSortIdOffset = static_cast<int64_t>(unSortId) * this->lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) +
                                 static_cast<int64_t>(round) * RADIX_SORT_NUM * this->lastDimTileNum_;
        __local_mem__ T3* nowTileHistBufferPtr = (__ubuf__ T3*)nowTileHistBuffer.GetPhyAddr();

        uint16_t repeatTime;
        if constexpr (IsSameType<T3, uint32_t>::value) {
            repeatTime = RADIX_SORT_NUM / VF_LEN_B32;
        } else {
            repeatTime = RADIX_SORT_NUM / VF_LEN_B64;
        }

        // Look back over prior tiles. Aggregate-ready tiles contribute their local histograms and keep scanning.
        // Prefix-ready tiles already include all earlier tiles, so add them and stop.
        for (int i = tileId - 1; i >= 0; --i) {
            int mode = -1;
            uint32_t histTileOffset = RADIX_SORT_NUM * i;
            __local_mem__ uint32_t* ubFlagTensorPtr = (__ubuf__ uint32_t*)ubFlagTensor.GetPhyAddr();
            __local_mem__ T3* tilePrevHistValuePtrCopy = nullptr;
            while (true) {
                // Poll until the previous tile publishes a full 256-bin aggregate-ready or prefix-ready state.
                LocalTensor<T3> xLocal = this->inQueueGlobalHist_.template AllocTensor<T3>();
                CopyGlobalDataIn<T3>(allTileHistBuffer[unSortIdOffset], xLocal, histTileOffset, RADIX_SORT_NUM);
                this->inQueueGlobalHist_.EnQue(xLocal);
                LocalTensor<T3> tilePrevHistValue = this->inQueueGlobalHist_.template DeQue<T3>();
                __local_mem__ T3* tilePrevHistValuePtr = (__ubuf__ T3*)tilePrevHistValue.GetPhyAddr();
                tilePrevHistValuePtrCopy = tilePrevHistValuePtr;
                __VEC_SCOPE__
                {
                    MicroAPI::MaskReg pRegSelect;
                    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
                    MicroAPI::RegTensor<uint32_t> notInitCount, aggReadyCount, prefixReadyCount;
                    MicroAPI::Duplicate(notInitCount, 0, maskB32);
                    MicroAPI::Duplicate(aggReadyCount, 0, maskB32);
                    MicroAPI::Duplicate(prefixReadyCount, 0, maskB32);
                    MicroAPI::RegTensor<uint32_t> onesVector, zerosVector;
                    MicroAPI::Duplicate(onesVector, 1, maskB32);
                    MicroAPI::Duplicate(zerosVector, 0, maskB32);
                    for (uint16_t i = 0; i < repeatTime; i++) {
                        MicroAPI::RegTensor<T3> prevTileHistValue;
                        MicroAPI::RegTensor<uint32_t> stateBitValue;
                        if constexpr (IsSameType<T3, uint32_t>::value) {
                            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                                prevTileHistValue, tilePrevHistValuePtr, VF_LEN_B32);
                            // The highest two bits carry the soft-sync state, the remaining bits carry the bin count.
                            MicroAPI::ShiftRights<uint32_t, int16_t>(stateBitValue, prevTileHistValue,
                                                                     STATE_BIT_SHF_VALUE, maskB32);
                            pRegSelect = maskB32;
                        } else {
                            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                                prevTileHistValue, tilePrevHistValuePtr, VF_LEN_B64);
                            MicroAPI::MaskReg maskB64 = MicroAPI::CreateMask<int64_t>();
                            MicroAPI::RegTensor<uint64_t> stateTmp;
                            // The highest two bits carry the soft-sync state, the remaining bits carry the bin count.
                            MicroAPI::ShiftRights<uint64_t, int16_t>(stateTmp,
                                                                     (MicroAPI::RegTensor<uint64_t>&)prevTileHistValue,
                                                                     STATE_BIT_SHF_VALUE_B64, maskB64);
                            MicroAPI::Pack(stateBitValue, stateTmp);
                            pRegSelect = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::H>();
                        }
                        MicroAPI::MaskReg maskNotInit;
                        MicroAPI::RegTensor<uint32_t> maskNotInitCount;
                        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskNotInit, stateBitValue, NOT_INIT_MODE,
                                                                       pRegSelect);
                        MicroAPI::Select(maskNotInitCount, onesVector, zerosVector, maskNotInit);
                        MicroAPI::Add(notInitCount, notInitCount, maskNotInitCount, maskNotInit);
                        MicroAPI::MaskReg maskAggReady;
                        MicroAPI::RegTensor<uint32_t> maskAggReadyCount;
                        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskAggReady, stateBitValue, AGG_READY_MODE,
                                                                       pRegSelect);
                        MicroAPI::Select(maskAggReadyCount, onesVector, zerosVector, maskAggReady);
                        MicroAPI::Add(aggReadyCount, aggReadyCount, maskAggReadyCount, maskAggReady);
                        MicroAPI::MaskReg maskPrefixReady;
                        MicroAPI::RegTensor<uint32_t> maskPrefixReadyCount;
                        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskPrefixReady, stateBitValue,
                                                                       PREFIX_READY_MODE, pRegSelect);
                        MicroAPI::Select(maskPrefixReadyCount, onesVector, zerosVector, maskPrefixReady);
                        MicroAPI::Add(prefixReadyCount, prefixReadyCount, maskPrefixReadyCount, maskPrefixReady);
                    }
                    MicroAPI::ReduceSum(notInitCount, notInitCount, maskB32);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                       MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, notInitCount,
                                                                                    HIST_MASK_OUT_LEN, maskB32);
                    MicroAPI::ReduceSum(aggReadyCount, aggReadyCount, maskB32);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                       MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, aggReadyCount,
                                                                                    HIST_MASK_OUT_LEN, maskB32);
                    MicroAPI::ReduceSum(prefixReadyCount, prefixReadyCount, maskB32);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                       MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, prefixReadyCount,
                                                                                    HIST_MASK_OUT_LEN, maskB32);
                }
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                uint32_t notInitCountScalar = ubFlagTensorPtr[NOT_INIT_COUNT_INDEX];
                uint32_t aggReadyCountScalar = ubFlagTensorPtr[AGG_READY_COUNT_INDEX];
                uint32_t PrefixReadyCountScalar = ubFlagTensorPtr[PREFIX_READY_COUNT_INDEX];
                if (aggReadyCountScalar == RADIX_SORT_NUM) {
                    mode = AGGREGATE_READY_FLAG;
                    this->inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                    break;
                }
                if (PrefixReadyCountScalar == RADIX_SORT_NUM) {
                    mode = PREFIX_READY_FLAG;
                    this->inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                    break;
                }
                this->inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
            }
            __local_mem__ T3* nowTileHistBufferPtrCopy = nowTileHistBufferPtr;
            event_t eventIdWaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIdWaitV);
            WaitFlag<HardEvent::S_V>(eventIdWaitV);
            __VEC_SCOPE__
            {
                MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint16_t>();
                MicroAPI::RegTensor<T3> histMask;
                if constexpr (IsSameType<T3, uint32_t>::value) {
                    MicroAPI::Duplicate(histMask, VALUE_MASK, predicateDefault);
                    for (uint16_t i = 0; i < repeatTime; i++) {
                        MicroAPI::RegTensor<uint32_t> nowTileHistVal, prevTileHistVal;
                        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            nowTileHistVal, nowTileHistBufferPtr, VF_LEN_B32);
                        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            prevTileHistVal, tilePrevHistValuePtrCopy, VF_LEN_B32);
                        // Strip the state bits before adding the previous tile's histogram value.
                        MicroAPI::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
                        MicroAPI::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
                        MicroAPI::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
                        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            nowTileHistBufferPtrCopy, nowTileHistVal, VF_LEN_B32, predicateDefault);
                    }
                } else {
                    MicroAPI::Duplicate(histMask, VALUE_MASK_B64, predicateDefault);
                    for (uint16_t i = 0; i < repeatTime; i++) {
                        MicroAPI::RegTensor<int64_t> nowTileHistVal, prevTileHistVal;
                        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            nowTileHistVal, nowTileHistBufferPtr, VF_LEN_B64);
                        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            prevTileHistVal, tilePrevHistValuePtrCopy, VF_LEN_B64);
                        // Strip the state bits before adding the previous tile's histogram value.
                        MicroAPI::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
                        MicroAPI::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
                        MicroAPI::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
                        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            nowTileHistBufferPtrCopy, nowTileHistVal, VF_LEN_B64, predicateDefault);
                    }
                }
            }
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
            if (mode == PREFIX_READY_FLAG) {
                break;
            }
        }
    }

    __aicore__ inline void AddPrevfixMask(LocalTensor<T3>& blockHistWithFlag, GlobalTensor<T3> blockHistToGm,
                                          uint32_t tileId, uint32_t round)
    {
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        int64_t unSortIdOffset = static_cast<int64_t>(unSortId) * this->lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) +
                                 static_cast<int64_t>(round) * RADIX_SORT_NUM * this->lastDimTileNum_;
        __local_mem__ T3* histCumsumPtr = (__ubuf__ T3*)blockHistWithFlag.GetPhyAddr();
        __local_mem__ T3* histCumsumPtrCopy = histCumsumPtr;

        uint16_t repeatTime;
        if constexpr (IsSameType<T3, uint32_t>::value) {
            repeatTime = RADIX_SORT_NUM / VF_LEN_B32;
        } else {
            repeatTime = RADIX_SORT_NUM / VF_LEN_B64;
        }
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T3> prefixReadyMask, prefixRemainMask;
            MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<T3>();
            if constexpr (IsSameType<T3, uint32_t>::value) {
                MicroAPI::Duplicate(prefixReadyMask, PREFIX_READY_MASK, predicateDefault);
                MicroAPI::Duplicate(prefixRemainMask, VALUE_MASK, predicateDefault);
                for (uint16_t repate = 0; repate < repeatTime; repate++) {
                    MicroAPI::RegTensor<uint32_t> keyCumsumValue;
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue, histCumsumPtr,
                                                                                          VF_LEN_B32);
                    // Preserve the accumulated histogram value and publish the prefix-ready state bit.
                    MicroAPI::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
                    MicroAPI::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        histCumsumPtrCopy, keyCumsumValue, VF_LEN_B32, predicateDefault);
                }
            } else {
                MicroAPI::Duplicate(prefixReadyMask, PREFIX_READY_MASK_B64, predicateDefault);
                MicroAPI::Duplicate(prefixRemainMask, VALUE_MASK_B64, predicateDefault);
                for (uint16_t repate = 0; repate < repeatTime; repate++) {
                    MicroAPI::RegTensor<int64_t> keyCumsumValue;
                    MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue, histCumsumPtr,
                                                                                         VF_LEN_B64);
                    // Preserve the accumulated histogram value and publish the prefix-ready state bit.
                    MicroAPI::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
                    MicroAPI::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
                    MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        histCumsumPtrCopy, keyCumsumValue, VF_LEN_B64, predicateDefault);
                }
            }
        }
        this->blockHistFlagUbQue_.EnQue(blockHistWithFlag);
        blockHistWithFlag = this->blockHistFlagUbQue_.template DeQue<T3>();
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(T3);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(blockHistToGm[unSortIdOffset + RADIX_SORT_NUM * tileId], blockHistWithFlag, dataCopyParam);
    }

    __aicore__ inline void DataCopyWorkSpaceToUb(LocalTensor<uint8_t> inputX8Ub, LocalTensor<uint16_t> blockExcusiveUb,
                                                 LocalTensor<uint16_t> blockHistUb, uint32_t unSortId, uint32_t tileId,
                                                 uint32_t curSize)
    {
        uint64_t histTileBase = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * this->lastDimTileNum_ +
                                static_cast<uint64_t>(tileId) * RADIX_SORT_NUM;
        DataCopyPadExtParams<uint16_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(uint16_t);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(blockExcusiveUb, this->histCumsumTileGmWk_[histTileBase], dataCopyParam, padParams);
        DataCopyPad(blockHistUb, this->histTileGmWk_[histTileBase], dataCopyParam, padParams);

        DataCopyPadExtParams<uint8_t> padParamsB8{false, 0, 0, 0};
        DataCopyExtParams dataCopyParam1;
        dataCopyParam1.blockCount = 1;
        dataCopyParam1.blockLen = curSize * sizeof(uint8_t);
        dataCopyParam1.srcStride = 0;
        dataCopyParam1.dstStride = 0;
        DataCopyPad(inputX8Ub,
                    this->xB8GmWk_[static_cast<uint64_t>(unSortId) * this->totalDataNum_ +
                                   static_cast<uint64_t>(tileId) * this->numTileData_],
                    dataCopyParam1, padParamsB8);
    }

    __aicore__ inline void ComputeOnePass(uint32_t round, uint32_t sortLoopRound, GlobalTensor<T1> inputXGm)
    {
        uint32_t startId = this->blockIdx_ % this->lastDimRealCore_;
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        uint64_t unsortedDimIndex = unSortId + static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_;
        int64_t xUnsortOffset = static_cast<int64_t>(unSortId) * this->totalDataNum_;
        if (unsortedDimIndex < static_cast<uint64_t>(this->unsortedDimNum_)) {
            for (uint32_t tileId = startId; tileId < this->lastDimTileNum_; tileId += this->lastDimRealCore_) {
                uint64_t tileOffset = static_cast<uint64_t>(tileId) * this->numTileData_;
                uint64_t tileDataStart = static_cast<uint64_t>(tileId) * this->numTileData_;
                uint64_t remainTileDataNum = this->totalDataNum_ - tileDataStart;
                if (this->totalDataNum_ < tileDataStart) {
                    break;
                }
                uint32_t currTileSize = static_cast<uint32_t>(
                    remainTileDataNum < static_cast<uint64_t>(this->numTileData_) ? remainTileDataNum :
                                                                                    this->numTileData_);
                LocalTensor<T1> xLocal = this->inQueueX_.template AllocTensor<T1>();
                if (round == 0) {
                    CopyDataIn<T1>(inputXGm[xUnsortOffset], xLocal, tileOffset, currTileSize);
                } else {
                    CopyDataIn<T1>(this->inputXDbGm_.Current()[xUnsortOffset], xLocal, tileOffset, currTileSize);
                }
                this->inQueueX_.EnQue(xLocal);
                xLocal = this->inQueueX_.template DeQue<T1>();
                LocalTensor<uint8_t> inputX8Ub = this->inputB8Que_.template AllocTensor<uint8_t>();
                LocalTensor<uint16_t> blockExcusiveUb = this->blockExcusiveInQue_.template AllocTensor<uint16_t>();
                LocalTensor<uint16_t> blockHistUb = this->blockHistInQue_.template AllocTensor<uint16_t>();
                DataCopyWorkSpaceToUb(inputX8Ub, blockExcusiveUb, blockHistUb, unSortId, tileId, currTileSize);
                this->blockHistInQue_.EnQue(blockHistUb);
                this->blockExcusiveInQue_.EnQue(blockExcusiveUb);
                this->inputB8Que_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(inputX8Ub);
                inputX8Ub = this->inputB8Que_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, uint8_t>();
                blockHistUb = this->blockHistInQue_.template DeQue<uint16_t>();

                LocalTensor<T3> blockHistFlagUb = this->blockHistFlagUbQue_.template AllocTensor<T3>();
                ScatterBlockHist2Global(blockHistUb, blockHistFlagUb, this->globalHistGmWkTmp_, tileId, round);

                LocalTensor<uint8_t> shareTmpBuffer = this->tmpUb_.template Get<uint8_t>();
                AscendC::LocalTensor<uint32_t> sortedValueIndexLocal = this->outIdxQueue_
                                                                           .template AllocTensor<uint32_t>();
                AscendC::LocalTensor<uint8_t> sortedValueLocal = this->outValueQueue_.template AllocTensor<uint8_t>();
                AscendC::Sort<uint8_t, false, sortConfigMuti>(sortedValueLocal, sortedValueIndexLocal, inputX8Ub,
                                                              shareTmpBuffer, static_cast<uint32_t>(currTileSize));
                this->outValueQueue_.template EnQue<uint8_t>(sortedValueLocal);
                this->outIdxQueue_.template EnQue<uint32_t>(sortedValueIndexLocal);
                this->inputB8Que_.FreeTensor(inputX8Ub);
                AscendC::LocalTensor<uint32_t> ubFlagTensor = this->blockUbFlagQue_.template AllocTensor<uint32_t>();
                LocalTensor<T3> blockHistFlagUb1 = this->blockHistFlagUbQue_.template AllocTensor<T3>();
                LocalTensor<uint32_t> blockHistFlagUb1Tmp = blockHistFlagUb1.template ReinterpretCast<uint32_t>();
                if (tileId > 0) {
                    LookbackGlobal(blockHistFlagUb1, this->globalHistGmWkTmp_, ubFlagTensor, tileId, round);
                }
                this->blockUbFlagQue_.FreeTensor(ubFlagTensor);
                if (tileId < (this->lastDimTileNum_ - 1)) {
                    AddPrevfixMask(blockHistFlagUb1, this->globalHistGmWkTmp_, tileId, round);
                }
                this->blockHistFlagUbQue_.FreeTensor(blockHistFlagUb1);
                LocalTensor<uint32_t> xIndexLocal;
                if (round != 0) {
                    xIndexLocal = this->inQueueIndex_.template AllocTensor<uint32_t>();
                    CopyIndexDataIn<T3>(this->idxDbGm_.Current()[xUnsortOffset * this->factor_], xIndexLocal,
                                        tileOffset * this->factor_, currTileSize);
                    this->inQueueIndex_.EnQue(xIndexLocal);
                    xIndexLocal = this->inQueueIndex_.template DeQue<uint32_t>();
                }
                sortedValueIndexLocal = this->outIdxQueue_.template DeQue<uint32_t>();
                sortedValueLocal = this->outValueQueue_.template DeQue<uint8_t>();
                AscendC::LocalTensor<T3> blockDataInGlobalPos = this->blockUbFlagQue_.template AllocTensor<T3>();
                blockExcusiveUb = this->blockExcusiveInQue_.template DeQue<uint16_t>();
                LocalTensor<uint32_t> blockHistFlagUb2 = this->blockHistFlagUbQue_.template AllocTensor<uint32_t>();
                static_cast<Derived*>(this)->ScatterKeysGlobal(
                    xLocal, sortedValueIndexLocal, xIndexLocal, sortedValueLocal, blockExcusiveUb, blockDataInGlobalPos,
                    blockHistFlagUb2, blockHistUb, round, tileDataStart, currTileSize, sortLoopRound);
                if (round != 0) {
                    this->inQueueIndex_.FreeTensor(xIndexLocal);
                }
                this->blockHistFlagUbQue_.FreeTensor(blockHistFlagUb2);
                this->inQueueX_.FreeTensor(xLocal);
                this->blockHistInQue_.FreeTensor(blockHistUb);
                this->blockUbFlagQue_.FreeTensor(blockDataInGlobalPos);
                this->blockExcusiveInQue_.FreeTensor(blockExcusiveUb);
                this->outIdxQueue_.FreeTensor(sortedValueIndexLocal);
                this->outValueQueue_.FreeTensor(sortedValueLocal);
            }
            this->idxDbGm_.selector_ ^= 1;
            this->inputXDbGm_.selector_ ^= 1;
        }
    }

    __aicore__ inline void ProcessRadix(GlobalTensor<T1> inputXGm, int64_t gmOffset, uint32_t sortLoopRound)
    {
        this->ClearWorkSapce();
        SyncAll();

        if constexpr (sizeof(T2) == sizeof(uint32_t)) {
            if constexpr (sizeof(T1) == sizeof(int8_t)) {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueDbWK_, this->outValueGm_[gmOffset]);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxDbWK_, this->outIdxGm_[gmOffset]);
            } else {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueGm_[gmOffset], this->outValueDbWK_);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxGm_[gmOffset], this->outIdxDbWK_);
            }
        } else {
            if constexpr (sizeof(T1) == sizeof(int8_t)) {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueDbWK_, this->outValueGm_[gmOffset]);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxDbWK_, this->outIdxGm_[gmOffset * INT64_INDEX_SCALE]);
            } else {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueGm_[gmOffset], this->outValueDbWK_);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxGm_[gmOffset * INT64_INDEX_SCALE], this->outIdxDbWK_);
            }
        }
        for (uint32_t round = 0; round < static_cast<uint32_t>(sizeof(T1)); round++) {
            this->GetGlobalExcusiveSum(round, sortLoopRound, inputXGm);
            SyncAll();
            this->ComputeOnePass(round, sortLoopRound, inputXGm);
            SyncAll();
        }
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx_ >= this->realCoreNum_) {
            return;
        }
        for (uint32_t i = 0; i < this->sortLoopTimes_; i++) {
            int64_t loopOffset = static_cast<int64_t>(i) * this->unsortedDimParallel_ * this->totalDataNum_;
            this->ProcessRadix(this->inputXGm_[loopOffset], loopOffset, i);
        }
    }
};

} // namespace RadixSortCommon

#endif // RADIX_MORE_CORE_BASE_H
