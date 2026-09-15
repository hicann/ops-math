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
 * \file
 * \brief
 */

#ifndef PAD_V3_GRAD_MIRROR_SIMT_H
#define PAD_V3_GRAD_MIRROR_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "pad_v3_grad_common.h"
#include "pad_v3_grad_struct.h"

constexpr int32_t MIRROR_THREAD_DIM = 2048;
constexpr int32_t MIRROR_HALF_THREAD_DIM = 1024;
constexpr int32_t MIRROR_QUARTER_THREAD_DIM = 512;
constexpr int32_t MIRROR_EIGHTH_THREAD_DIM = 256;
constexpr int32_t MIRROR_SIXTEENTH_THREAD_DIM = 128;

namespace PadV3Grad {
using namespace AscendC;

template <typename T, uint8_t KEY>
class PadV3GradMirrorSimt {
public:
    __aicore__ inline PadV3GradMirrorSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData);
    template <typename U>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> mInputGM_;
    GlobalTensor<T> mOutputGM_;
    uint32_t mBlockIdx_;                     // 核号
    const PadV3GradACTilingData* mMirrorTD_; // tilingData
};

template <typename T, uint8_t KEY>
__aicore__ inline void PadV3GradMirrorSimt<T, KEY>::Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mMirrorTD_ = tilingData;

    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <uint8_t DIM_NUM, typename U>
__simt_callee__ __aicore__ void ReflectDimOffset(IdxAndTimes<U>* mirrorReflectIdxCnt, U* inIndex, U* outIndex,
                                                 __ubuf__ U* inStrides, __ubuf__ U* outShapes, __ubuf__ U* leftPads,
                                                 __ubuf__ U* rightPads)
{
    for (uint8_t i = 0; i < DIM_NUM; i++) {
        mirrorReflectIdxCnt[i].inGmIdx[0] = inIndex[i] * inStrides[i];
        if (outIndex[i] - 1 < leftPads[i] && outIndex[i] > 0) // left
        {
            // 计算该点该维度左pad在输入GM上的偏移
            mirrorReflectIdxCnt[i].inGmIdx[mirrorReflectIdxCnt[i].cnt] = (leftPads[i] - outIndex[i]) * inStrides[i];
            mirrorReflectIdxCnt[i].cnt++;
        }
        if (outShapes[i] - outIndex[i] - 1 <= rightPads[i] && outShapes[i] - outIndex[i] - 1 > 0) // right
        {
            // 计算该点该维度右pad在输入GM上的偏移
            mirrorReflectIdxCnt[i].inGmIdx[mirrorReflectIdxCnt[i].cnt] = (2 * outShapes[i] - outIndex[i] + leftPads[i] -
                                                                          2) *
                                                                         inStrides[i];
            mirrorReflectIdxCnt[i].cnt++;
        }
    }
}

template <uint8_t DIM_NUM, typename U>
__simt_callee__ __aicore__ void SymmetricDimOffset(IdxAndTimes<U>* mirrorSymmetricIdxCnt, U* inIndex, U* outIndex,
                                                   __ubuf__ U* inStrides, __ubuf__ U* outShapes, __ubuf__ U* leftPads,
                                                   __ubuf__ U* rightPads)
{
    for (uint8_t i = 0; i < DIM_NUM; i++) {
        mirrorSymmetricIdxCnt[i].inGmIdx[0] = inIndex[i] * inStrides[i];
        if (outIndex[i] < leftPads[i]) {
            mirrorSymmetricIdxCnt[i].inGmIdx[mirrorSymmetricIdxCnt[i].cnt] = (leftPads[i] - outIndex[i] - 1) *
                                                                             inStrides[i];
            mirrorSymmetricIdxCnt[i].cnt++;
        }
        if (outShapes[i] - outIndex[i] <= rightPads[i]) {
            mirrorSymmetricIdxCnt[i].inGmIdx[mirrorSymmetricIdxCnt[i].cnt] = (2 * outShapes[i] - outIndex[i] +
                                                                              leftPads[i] - 1) *
                                                                             inStrides[i];
            mirrorSymmetricIdxCnt[i].cnt++;
        }
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                              uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                              __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                              __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                              __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> mirrorInIdxCnt1[DIM_NUM];
        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(mirrorInIdxCnt1, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(mirrorInIdxCnt1, inIndex, outIndex, inStrides, outShapes, leftPads,
                                           rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < mirrorInIdxCnt1[0].cnt; a0++) {
            if (mirrorInIdxCnt1[0].inGmIdx[a0] < 0 || mirrorInIdxCnt1[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = mirrorInIdxCnt1[0].inGmIdx[a0];
            CastType tmpVal;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                tmpVal = __bfloat162float(inputGM[a0Offset]);
            } else if constexpr (std::is_same_v<T, float16_t>) {
                tmpVal = __half2float(inputGM[a0Offset]);
            } else {
                tmpVal = inputGM[a0Offset];
            }
            total += tmpVal;
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                              uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                              __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                              __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                              __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        uint64_t yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> mirrorInIdxCnt2[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(mirrorInIdxCnt2, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(mirrorInIdxCnt2, inIndex, outIndex, inStrides, outShapes, leftPads,
                                           rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < mirrorInIdxCnt2[0].cnt; a0++) {
            if (mirrorInIdxCnt2[0].inGmIdx[a0] < 0 || mirrorInIdxCnt2[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = static_cast<uint64_t>(mirrorInIdxCnt2[0].inGmIdx[a0]);
            for (uint8_t a1 = 0; a1 < mirrorInIdxCnt2[1].cnt; a1++) {
                if (mirrorInIdxCnt2[1].inGmIdx[a1] < 0 || mirrorInIdxCnt2[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + static_cast<uint64_t>(mirrorInIdxCnt2[1].inGmIdx[a1]);
                CastType tmpVal;
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    tmpVal = __bfloat162float(inputGM[a1Offset]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    tmpVal = __half2float(inputGM[a1Offset]);
                } else {
                    tmpVal = inputGM[a1Offset];
                }
                total += tmpVal;
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                                uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                                __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                                __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                                __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> mirrorInIdxCnt3[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(mirrorInIdxCnt3, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(mirrorInIdxCnt3, inIndex, outIndex, inStrides, outShapes, leftPads,
                                           rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < mirrorInIdxCnt3[0].cnt; a0++) {
            if (mirrorInIdxCnt3[0].inGmIdx[a0] < 0 || mirrorInIdxCnt3[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = mirrorInIdxCnt3[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < mirrorInIdxCnt3[1].cnt; a1++) {
                if (mirrorInIdxCnt3[1].inGmIdx[a1] < 0 || mirrorInIdxCnt3[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + mirrorInIdxCnt3[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < mirrorInIdxCnt3[2].cnt; a2++) {
                    if (mirrorInIdxCnt3[2].inGmIdx[a2] < 0 || mirrorInIdxCnt3[2].inGmIdx[a2] >= cutBounds[2]) {
                        continue;
                    }
                    GmOffsetType a2Offset = a1Offset + mirrorInIdxCnt3[2].inGmIdx[a2];
                    CastType tmpVal;
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        tmpVal = __bfloat162float(inputGM[a2Offset]);
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        tmpVal = __half2float(inputGM[a2Offset]);
                    } else {
                        tmpVal = inputGM[a2Offset];
                    }
                    total += tmpVal;
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                               __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                               __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> mirrorInIdxCnt4[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(mirrorInIdxCnt4, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(mirrorInIdxCnt4, inIndex, outIndex, inStrides, outShapes, leftPads,
                                           rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < mirrorInIdxCnt4[0].cnt; a0++) {
            if (mirrorInIdxCnt4[0].inGmIdx[a0] < 0 || mirrorInIdxCnt4[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = mirrorInIdxCnt4[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < mirrorInIdxCnt4[1].cnt; a1++) {
                if (mirrorInIdxCnt4[1].inGmIdx[a1] < 0 || mirrorInIdxCnt4[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + mirrorInIdxCnt4[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < mirrorInIdxCnt4[2].cnt; a2++) {
                    if (mirrorInIdxCnt4[2].inGmIdx[a2] < 0 || mirrorInIdxCnt4[2].inGmIdx[a2] >= cutBounds[2]) {
                        continue;
                    }
                    GmOffsetType a2Offset = a1Offset + mirrorInIdxCnt4[2].inGmIdx[a2];
                    for (uint8_t a3 = 0; a3 < mirrorInIdxCnt4[3].cnt; a3++) {
                        if (mirrorInIdxCnt4[3].inGmIdx[a3] < 0 || mirrorInIdxCnt4[3].inGmIdx[a3] >= cutBounds[3]) {
                            continue;
                        }
                        GmOffsetType a3Offset = a2Offset + mirrorInIdxCnt4[3].inGmIdx[a3];
                        CastType tmpVal;
                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            tmpVal = __bfloat162float(inputGM[a3Offset]);
                        } else if constexpr (std::is_same_v<T, float16_t>) {
                            tmpVal = __half2float(inputGM[a3Offset]);
                        } else {
                            tmpVal = inputGM[a3Offset];
                        }
                        total += tmpVal;
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                               __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                               __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> mirrorInIdxCnt5[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(mirrorInIdxCnt5, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(mirrorInIdxCnt5, inIndex, outIndex, inStrides, outShapes, leftPads,
                                           rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < mirrorInIdxCnt5[0].cnt; a0++) {
            if (mirrorInIdxCnt5[0].inGmIdx[a0] < 0 || mirrorInIdxCnt5[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = mirrorInIdxCnt5[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < mirrorInIdxCnt5[1].cnt; a1++) {
                if (mirrorInIdxCnt5[1].inGmIdx[a1] < 0 || mirrorInIdxCnt5[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + mirrorInIdxCnt5[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < mirrorInIdxCnt5[2].cnt; a2++) {
                    if (mirrorInIdxCnt5[2].inGmIdx[a2] < 0 || mirrorInIdxCnt5[2].inGmIdx[a2] >= cutBounds[2]) {
                        continue;
                    }
                    GmOffsetType a2Offset = a1Offset + mirrorInIdxCnt5[2].inGmIdx[a2];
                    for (uint8_t a3 = 0; a3 < mirrorInIdxCnt5[3].cnt; a3++) {
                        if (mirrorInIdxCnt5[3].inGmIdx[a3] < 0 || mirrorInIdxCnt5[3].inGmIdx[a3] >= cutBounds[3]) {
                            continue;
                        }
                        GmOffsetType a3Offset = a2Offset + mirrorInIdxCnt5[3].inGmIdx[a3];
                        for (uint8_t a4 = 0; a4 < mirrorInIdxCnt5[4].cnt; a4++) {
                            if (mirrorInIdxCnt5[4].inGmIdx[a4] < 0 || mirrorInIdxCnt5[4].inGmIdx[a4] >= cutBounds[4]) {
                                continue;
                            }
                            GmOffsetType a4Offset = a3Offset + mirrorInIdxCnt5[4].inGmIdx[a4];
                            CastType tmpVal;
                            if constexpr (std::is_same_v<T, bfloat16_t>) {
                                tmpVal = __bfloat162float(inputGM[a4Offset]);
                            } else if constexpr (std::is_same_v<T, float16_t>) {
                                tmpVal = __half2float(inputGM[a4Offset]);
                            } else {
                                tmpVal = inputGM[a4Offset];
                            }
                            total += tmpVal;
                        }
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t KEY>
template <typename U>
__aicore__ inline void PadV3GradMirrorSimt<T, KEY>::Process()
{
    using CastType = PadV3GradCastType<T>;
    using GmOffsetType = PadV3GradGmOffsetType<U>;

    // 快速除参数
    __ubuf__ GmOffsetType magics[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ GmOffsetType shifts[PAD_GRAD_MAX_DIMS_NUM];
    // tiling data
    __ubuf__ U inShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U inStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U leftPads[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U rightPads[PAD_GRAD_MAX_DIMS_NUM];
    // 裁剪边界
    __ubuf__ U cutBounds[PAD_GRAD_MAX_DIMS_NUM];

    uint32_t blockNum = AscendC::GetBlockNum(); // 获取到核数
    GmOffsetType outputSize = 0;
    if (!PrepareSimtGradArrays<U, GmOffsetType>(mMirrorTD_, blockNum, magics, shifts, inShapes, outShapes, inStrides,
                                                outStrides, leftPads, rightPads, cutBounds, outputSize)) {
        return;
    }

    if (mMirrorTD_->dimNum == 1) {
        asc_vf_call<SimtComputeMirrorOne<T, 1, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mMirrorTD_->dimNum == 2) {
        asc_vf_call<SimtComputeMirrorTwo<T, 2, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mMirrorTD_->dimNum == 3) {
        asc_vf_call<SimtComputeMirrorThree<T, 3, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mMirrorTD_->dimNum == 4) {
        asc_vf_call<SimtComputeMirrorFour<T, 4, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mMirrorTD_->dimNum == 5) {
        asc_vf_call<SimtComputeMirrorFive<T, 5, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    }
}

} // namespace PadV3Grad
#endif
