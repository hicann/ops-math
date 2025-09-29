/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/* !
 * \file mirror_pad_common.h
 * \brief mirror_pad_common
 */

#ifndef ASCENDC_MIRROR_PAD_COMMON_H
#define ASCENDC_MIRROR_PAD_COMMON_H

#include <cmath>
#include <cstdint>
#include "kernel_operator.h"
#include "../../inc/platform.h"

namespace MirrorPad {
using namespace AscendC;

constexpr int64_t TILING_NDDMA_LEN = 5;
constexpr uint32_t NDDMA_MAX_PADDING = 255;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UINT_32_ONE = 1;
constexpr uint8_t UINT_8_ZERO = 0;
constexpr int64_t NDDMA_DIM = 5;
constexpr uint32_t TWO_DIM = 2;
constexpr uint32_t THREE_DIM = 3;
constexpr uint32_t FOUR_DIM = 4;
constexpr uint32_t FIVE_DIM = 5;
constexpr uint32_t SIX_DIM = 6;
constexpr uint32_t THREAD_DIM = 1024;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename P>
__aicore__ inline void CopyTilingArray(P curArray[8], P tilingArray[8], uint32_t dim)
{
    for (uint16_t i = 0; i < dim; ++i) {
        curArray[i] = tilingArray[i];
    }
}

template <typename T>
__aicore__ inline MultiCopyParams<T, NDDMA_DIM> InitPadCopyParams(uint32_t loopSize, T value)
{
    return {
        {{UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, UINT_32_ONE},  // srcstride
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, UINT_32_ONE},  // dststride
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, loopSize},     // loopsize
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO},  // leftpad
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO}}, // rightpad
        value};
}

template <typename T>
__aicore__ inline MultiCopyParams<T, NDDMA_DIM> InitFirstDataPieceParams(
    uint32_t srcStride, uint32_t dstStride, uint32_t loopSize, T value)
{
    return {
        {{UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, srcStride, UINT_32_ONE},
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, dstStride, UINT_32_ONE},
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, loopSize, srcStride},
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO},
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO}},
        value};
}

template <typename T>
__aicore__ inline MultiCopyParams<T, NDDMA_DIM> InitMainDataPieceParams(
    uint32_t srcStride, uint32_t dstStride, uint32_t loopSize, T value)
{
    return {
        {{UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, srcStride, UINT_32_ONE},
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, dstStride, UINT_32_ONE},
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, loopSize, srcStride},
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO},
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO}},
        value};
}

template <typename T>
__aicore__ inline MultiCopyParams<T, NDDMA_DIM> InitLastDataPieceParams(
    uint32_t srcStride, uint32_t dstStride, uint32_t loopSize, T value)
{
    return {
        {{UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, srcStride, UINT_32_ONE},
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, dstStride, UINT_32_ONE},
         {UINT_32_ONE, UINT_32_ONE, UINT_32_ONE, loopSize, srcStride},
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO},
         {UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO, UINT_8_ZERO}},
        value};
}

__aicore__ inline void GetCurrentPosition(uint32_t counts[], uint64_t idx, uint64_t strides[], uint32_t dim)
{
    idx *= strides[dim - THREE_DIM];
    for (uint32_t i = 0; i < dim - TWO_DIM; ++i) {
        counts[i] = idx / strides[i];
        idx = idx - (counts[i] * strides[i]);
    }
}

template <typename T>
__aicore__ inline void OneRowMoveAlignCopy(
    uint32_t frontPad, uint32_t rowLength, uint64_t srcOffset, const LocalTensor<T>& outBuffer,
    const AscendC::GlobalTensor<T> inputGm)
{
    uint32_t alignNum = ((frontPad * sizeof(T)) % BLOCK_SIZE) / sizeof(T);
    uint32_t dstOffset = (frontPad * sizeof(T) / BLOCK_SIZE) * (BLOCK_SIZE / sizeof(T));
    DataCopyExtParams extParams{static_cast<uint16_t>(1), static_cast<uint32_t>(rowLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, static_cast<uint8_t>(alignNum), 0, static_cast<T>(0)};
    DataCopyPad(outBuffer[dstOffset], inputGm[srcOffset], extParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOut2MultiPos0Dim(
    const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm)
{
    DataCopyPad(yGm[offset], outBuffer, copyParams);
}

template <typename T, typename F>
__aicore__ inline void CopyOut2MultiPos1Dim(
    const uint32_t counts[], const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm, const uint32_t frontPads[], const uint32_t backPads[],
    const uint32_t inputShape[], const uint64_t afterPadStrides[])
{
    uint32_t dim1Idx[THREE_DIM] = {0, 0, 0};
    uint16_t loop1Num = 1;
    F()(dim1Idx, loop1Num, counts[0], frontPads[0], backPads[0], inputShape[0]);
    for (uint16_t a = 0; a < loop1Num; ++a) {
        uint64_t Dim1Offset = dim1Idx[a] * afterPadStrides[0];
        DataCopyPad(yGm[Dim1Offset + offset], outBuffer, copyParams);
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtRefModeCompute(
    const uint32_t rowNum, const uint32_t frontPad, const uint32_t backPad, __ubuf__ T* outBuffer, uint32_t lastLength,
    uint32_t lastLengthWithPad)
{
    uint32_t lastTotalPad = frontPad + backPad;
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < rowNum * lastTotalPad; ubIdx += Simt::GetThreadNum()) {
        uint32_t rowIdx = ubIdx / lastTotalPad;
        uint32_t colIdx = ubIdx % lastTotalPad;
        uint32_t oriColIdx = frontPad * TWO_DIM - colIdx;
        if (colIdx >= frontPad) {
            oriColIdx = lastLength + frontPad * TWO_DIM - colIdx - TWO_DIM;
            colIdx += lastLength;
        }
        outBuffer[lastLengthWithPad * rowIdx + colIdx] = outBuffer[lastLengthWithPad * rowIdx + oriColIdx];
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtSymModeCompute(
    const uint32_t rowNum, const uint32_t frontPad, const uint32_t backPad, __ubuf__ T* outBuffer, uint32_t lastLength,
    uint32_t lastLengthWithPad)
{
    uint32_t lastTotalPad = frontPad + backPad;
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < rowNum * lastTotalPad; ubIdx += Simt::GetThreadNum()) {
        uint32_t rowIdx = ubIdx / lastTotalPad;
        uint32_t colIdx = ubIdx % lastTotalPad;
        uint32_t oriColIdx = frontPad * TWO_DIM - colIdx - 1;
        if (colIdx >= frontPad) {
            oriColIdx = lastLength + frontPad * TWO_DIM - colIdx - 1;
            colIdx += lastLength;
        }
        outBuffer[lastLengthWithPad * rowIdx + colIdx] = outBuffer[lastLengthWithPad * rowIdx + oriColIdx];
    }
}

template <typename T, typename F>
__aicore__ inline void CopyOut2MultiPos2Dim(
    const uint32_t counts[], const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm, const uint32_t frontPads[], const uint32_t backPads[],
    const uint32_t inputShape[], const uint64_t afterPadStrides[])
{
    uint32_t dim1Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim2Idx[THREE_DIM] = {0, 0, 0};
    uint16_t loop1Num = 1;
    uint16_t loop2Num = 1;
    F()(dim1Idx, loop1Num, counts[0], frontPads[0], backPads[0], inputShape[0]);
    F()(dim2Idx, loop2Num, counts[1], frontPads[1], backPads[1], inputShape[1]);
    for (uint16_t a = 0; a < loop1Num; ++a) {
        uint64_t Dim1Offset = dim1Idx[a] * afterPadStrides[0];
        for (uint16_t b = 0; b < loop2Num; ++b) {
            uint64_t Dim2Offset = dim2Idx[b] * afterPadStrides[1];
            DataCopyPad(yGm[Dim1Offset + Dim2Offset + offset], outBuffer, copyParams);
        }
    }
}

template <typename T, typename F>
__aicore__ inline void CopyOut2MultiPos3Dim(
    const uint32_t counts[], const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm, const uint32_t frontPads[], const uint32_t backPads[],
    const uint32_t inputShape[], const uint64_t afterPadStrides[])
{
    uint32_t dim1Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim2Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim3Idx[THREE_DIM] = {0, 0, 0};
    uint16_t loop1Num = 1;
    uint16_t loop2Num = 1;
    uint16_t loop3Num = 1;
    F()(dim1Idx, loop1Num, counts[0], frontPads[0], backPads[0], inputShape[0]);
    F()(dim2Idx, loop2Num, counts[1], frontPads[1], backPads[1], inputShape[1]);
    F()(dim3Idx, loop3Num, counts[TWO_DIM], frontPads[TWO_DIM], backPads[TWO_DIM], inputShape[TWO_DIM]);
    for (uint16_t a = 0; a < loop1Num; ++a) {
        uint64_t Dim1Offset = dim1Idx[a] * afterPadStrides[0];
        for (uint16_t b = 0; b < loop2Num; ++b) {
            uint64_t Dim2Offset = dim2Idx[b] * afterPadStrides[1];
            for (uint16_t c = 0; c < loop3Num; ++c) {
                uint64_t Dim3Offset = dim3Idx[c] * afterPadStrides[TWO_DIM];
                DataCopyPad(yGm[Dim1Offset + Dim2Offset + Dim3Offset + offset], outBuffer, copyParams);
            }
        }
    }
}

template <typename T, typename F>
__aicore__ inline void CopyOut2MultiPos4Dim(
    const uint32_t counts[], const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm, const uint32_t frontPads[], const uint32_t backPads[],
    const uint32_t inputShape[], const uint64_t afterPadStrides[])
{
    uint32_t dim1Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim2Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim3Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim4Idx[THREE_DIM] = {0, 0, 0};
    uint16_t loop1Num = 1;
    uint16_t loop2Num = 1;
    uint16_t loop3Num = 1;
    uint16_t loop4Num = 1;
    F()(dim1Idx, loop1Num, counts[0], frontPads[0], backPads[0], inputShape[0]);
    F()(dim2Idx, loop2Num, counts[1], frontPads[1], backPads[1], inputShape[1]);
    F()(dim3Idx, loop3Num, counts[TWO_DIM], frontPads[TWO_DIM], backPads[TWO_DIM], inputShape[TWO_DIM]);
    F()(dim4Idx, loop4Num, counts[THREE_DIM], frontPads[THREE_DIM], backPads[THREE_DIM], inputShape[THREE_DIM]);
    for (uint16_t a = 0; a < loop1Num; ++a) {
        uint64_t Dim1Offset = dim1Idx[a] * afterPadStrides[0];
        for (uint16_t b = 0; b < loop2Num; ++b) {
            uint64_t Dim2Offset = dim2Idx[b] * afterPadStrides[1];
            for (uint16_t c = 0; c < loop3Num; ++c) {
                uint64_t Dim3Offset = dim3Idx[c] * afterPadStrides[TWO_DIM];
                for (uint16_t d = 0; d < loop4Num; ++d) {
                    uint64_t Dim4Offset = dim4Idx[d] * afterPadStrides[THREE_DIM];
                    DataCopyPad(yGm[Dim1Offset + Dim2Offset + Dim3Offset + Dim4Offset + offset], outBuffer, copyParams);
                }
            }
        }
    }
}

template <typename T, typename F>
__aicore__ inline void CopyOut2MultiPos5Dim(
    const uint32_t counts[], const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm, const uint32_t frontPads[], const uint32_t backPads[],
    const uint32_t inputShape[], const uint64_t afterPadStrides[])
{
    uint32_t dim1Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim2Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim3Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim4Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim5Idx[THREE_DIM] = {0, 0, 0};
    uint16_t loop1Num = 1;
    uint16_t loop2Num = 1;
    uint16_t loop3Num = 1;
    uint16_t loop4Num = 1;
    uint16_t loop5Num = 1;
    F()(dim1Idx, loop1Num, counts[0], frontPads[0], backPads[0], inputShape[0]);
    F()(dim2Idx, loop2Num, counts[1], frontPads[1], backPads[1], inputShape[1]);
    F()(dim3Idx, loop3Num, counts[TWO_DIM], frontPads[TWO_DIM], backPads[TWO_DIM], inputShape[TWO_DIM]);
    F()(dim4Idx, loop4Num, counts[THREE_DIM], frontPads[THREE_DIM], backPads[THREE_DIM], inputShape[THREE_DIM]);
    F()(dim5Idx, loop5Num, counts[FOUR_DIM], frontPads[FOUR_DIM], backPads[FOUR_DIM], inputShape[FOUR_DIM]);
    for (uint16_t a = 0; a < loop1Num; ++a) {
        uint64_t Dim1Offset = dim1Idx[a] * afterPadStrides[0];
        for (uint16_t b = 0; b < loop2Num; ++b) {
            uint64_t Dim2Offset = dim2Idx[b] * afterPadStrides[1];
            for (uint16_t c = 0; c < loop3Num; ++c) {
                uint64_t Dim3Offset = dim3Idx[c] * afterPadStrides[TWO_DIM];
                for (uint16_t d = 0; d < loop4Num; ++d) {
                    uint64_t Dim4Offset = dim4Idx[d] * afterPadStrides[THREE_DIM];
                    for (uint16_t e = 0; e < loop5Num; ++e) {
                        uint64_t Dim5Offset = dim5Idx[e] * afterPadStrides[FOUR_DIM];
                        DataCopyPad(
                            yGm[Dim1Offset + Dim2Offset + Dim3Offset + Dim4Offset + Dim5Offset + offset], outBuffer,
                            copyParams);
                    }
                }
            }
        }
    }
}

template <typename T, typename F>
__aicore__ inline void CopyOut2MultiPos6Dim(
    const uint32_t counts[], const LocalTensor<T>& outBuffer, DataCopyExtParams& copyParams, const uint64_t offset,
    const AscendC::GlobalTensor<T> yGm, const uint32_t frontPads[], const uint32_t backPads[],
    const uint32_t inputShape[], const uint64_t afterPadStrides[])
{
    uint32_t dim1Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim2Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim3Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim4Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim5Idx[THREE_DIM] = {0, 0, 0};
    uint32_t dim6Idx[THREE_DIM] = {0, 0, 0};
    uint16_t loop1Num = 1;
    uint16_t loop2Num = 1;
    uint16_t loop3Num = 1;
    uint16_t loop4Num = 1;
    uint16_t loop5Num = 1;
    uint16_t loop6Num = 1;
    F()(dim1Idx, loop1Num, counts[0], frontPads[0], backPads[0], inputShape[0]);
    F()(dim2Idx, loop2Num, counts[1], frontPads[1], backPads[1], inputShape[1]);
    F()(dim3Idx, loop3Num, counts[TWO_DIM], frontPads[TWO_DIM], backPads[TWO_DIM], inputShape[TWO_DIM]);
    F()(dim4Idx, loop4Num, counts[THREE_DIM], frontPads[THREE_DIM], backPads[THREE_DIM], inputShape[THREE_DIM]);
    F()(dim5Idx, loop5Num, counts[FOUR_DIM], frontPads[FOUR_DIM], backPads[FOUR_DIM], inputShape[FOUR_DIM]);
    F()(dim6Idx, loop6Num, counts[FIVE_DIM], frontPads[FIVE_DIM], backPads[FIVE_DIM], inputShape[FIVE_DIM]);
    for (uint16_t a = 0; a < loop1Num; ++a) {
        uint64_t Dim1Offset = dim1Idx[a] * afterPadStrides[0];
        for (uint16_t b = 0; b < loop2Num; ++b) {
            uint64_t Dim2Offset = dim2Idx[b] * afterPadStrides[1];
            for (uint16_t c = 0; c < loop3Num; ++c) {
                uint64_t Dim3Offset = dim3Idx[c] * afterPadStrides[TWO_DIM];
                for (uint16_t d = 0; d < loop4Num; ++d) {
                    uint64_t Dim4Offset = dim4Idx[d] * afterPadStrides[THREE_DIM];
                    for (uint16_t e = 0; e < loop5Num; ++e) {
                        uint64_t Dim5Offset = dim5Idx[e] * afterPadStrides[FOUR_DIM];
                        for (uint16_t f = 0; f < loop6Num; ++f) {
                            uint64_t Dim6Offset = dim6Idx[f] * afterPadStrides[FIVE_DIM];
                            DataCopyPad(
                                yGm[Dim1Offset + Dim2Offset + Dim3Offset + Dim4Offset + Dim5Offset + Dim6Offset +
                                    offset],
                                outBuffer, copyParams);
                        }
                    }
                }
            }
        }
    }
}
} // namespace MirrorPad
#endif