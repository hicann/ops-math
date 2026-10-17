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
 * \file transpose_b4.h
 * \brief transpose template for HIF4 (b4/fp4) data type — dual-cut tiling (TilingKey 10009)
 *
 * TilingData layout (HIF4DualCutTilingData, must match host side exactly):
 *   gmShape[8] / gmInStride[8] / gmOutStride[8]  — outer loop axes
 *   ubShape[4] / ubInStride[4] / ubOutStride[4]  — inner (UB) loop axes
 *   cutAxisIdx01/02 + cutAxisTail01/02            — tail handling for cut axes
 *   outerAxisNums / innerAxisNums                 — active axis counts
 *   dimsPerCore / useCoreNum / dimsPerCoreTail    — multi-core split
 *
 * Kernel execution flow per core:
 *   1. Compute the range of gm-iterations this core handles.
 *   2. For each gm-iteration:
 *        a. Unflatten linear iter -> multi-dim gmIdx.
 *        b. Compute GM input/output byte offsets from gmIdx & strides.
 *        c. Adjust ubShape for cut-axis tails (last outer iteration).
 *        d. CopyIn  (Compact)  HIF4 bytes from input GM -> UB.
 *        e. Cast    fp4 -> bf16.
 *        f. Transpose in bf16 domain (output-order = ubOutStride ascending).
 *        g. Cast    bf16 -> fp4.
 *        h. CopyOut   HIF4 bytes from UB -> output GM.
 */
#ifndef TRANSPOSE_B4_H
#define TRANSPOSE_B4_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};

constexpr int32_t HIF4_ELEMENTS_PER_BYTE = 2;
constexpr int32_t DUAL_CUT_GM_LEN = 8;
constexpr int32_t DUAL_CUT_UB_LEN = 4;

constexpr int32_t TAIL_TWO = 2;
constexpr int32_t TAIL_THREE = 3;
constexpr int32_t TAIL_FOUR = 4;
constexpr int32_t INNNER_AXIS_NUMS_TWO = 2;
constexpr int32_t INNNER_AXIS_NUMS_THREE = 3;
constexpr int32_t INNNER_AXIS_NUMS_FOUR = 4;
constexpr int32_t DOUBLE_BUFF = 2;
constexpr int32_t FOUR_BUFF = 4;
constexpr int32_t TQUE_DEPTH = 2;
constexpr int32_t MAIN_TWO = 2;
constexpr int32_t MAIN_THREE = 3;

// =========================================================================
// TransposeHIF4 — dual-cut kernel class
// =========================================================================
class TransposeHIF4 {
public:
    __aicore__ inline TransposeHIF4(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HIF4DualCutTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmInByteOffset, const int32_t* updatedUbInShape);
    __aicore__ inline void CastFp4ToBf16(int64_t elemCount, const LocalTensor<bfloat16_t>& castLocal);
    __aicore__ inline void SimtTranspose(int64_t elemCount, const int32_t* updatedUbInShape,
                                         const int64_t* updatedUbInStride, const int64_t* updatedUbOutStride,
                                         const LocalTensor<bfloat16_t>& castLocal,
                                         const LocalTensor<bfloat16_t>& transposeLocal);
    __aicore__ inline void CastBf16ToFp4(int64_t elemCount, const LocalTensor<bfloat16_t>& castLocal,
                                         const LocalTensor<bfloat16_t>& transposeLocal);
    __aicore__ inline void DeInterLeaveProcess(int64_t byteCount);
    __aicore__ inline void CopyOut(int64_t gmOutByteOffset, const int32_t* updatedUbOutShape,
                                   const int64_t* updatedGmInStride, const LocalTensor<bfloat16_t>& castLocal);

    // --- Process() decomposition helpers ---
    __aicore__ inline void ComputeIterRange(int32_t& startIter, int32_t& endIter);
    __aicore__ inline void UnflattenGmIdx(int32_t iter, int32_t gmBaseIdx, int32_t* gmIdx);
    __aicore__ inline void ComputeGmByteOffsets(const int32_t* gmIdx, int32_t gmBaseIdx, int64_t& gmInByteOffset,
                                                int64_t& gmOutByteOffset);
    __aicore__ inline void ComputeUpdatedUbShape(const int32_t* gmIdx, int32_t gmBaseIdx, int32_t* updatedUbInShape,
                                                 int32_t* updatedUbOutShape);
    __aicore__ inline void ComputeTileStrides(const int32_t* updatedUbInShape, const int32_t* updatedUbOutShape,
                                              int64_t& tileB16Elem, int64_t& planeEleBf16, int64_t& validBf16Elem,
                                              int64_t* updatedUbOutStride, int64_t* updatedGmInStride);
    __aicore__ inline void AlignOutStrideForB4(const int32_t* updatedUbInShape, int64_t* updatedUbOutStride,
                                               int64_t* updatedGmInStride, int64_t validBf16Elem,
                                               int64_t& validBf16ToB4Elem);
    __aicore__ inline void ComputeUbInStride(const int32_t* updatedUbInShape, int64_t planeEleBf16,
                                             int64_t* updatedUbInStride);
    __aicore__ inline void ProcessOneIteration(int32_t iter, const LocalTensor<bfloat16_t>& castLocal,
                                               const LocalTensor<bfloat16_t>& transposeLocal);
    __aicore__ inline void RunTilePipeline(int64_t gmInByteOffset, int64_t gmOutByteOffset,
                                           const int32_t* updatedUbInShape, const int32_t* updatedUbOutShape,
                                           const int64_t* updatedUbInStride, const int64_t* updatedUbOutStride,
                                           const int64_t* updatedGmInStride, int64_t tileB16Elem, int64_t validBf16Elem,
                                           int64_t validBf16ToB4Elem, const LocalTensor<bfloat16_t>& castLocal,
                                           const LocalTensor<bfloat16_t>& transposeLocal);

    constexpr static uint32_t VL_SIZE = Ops::Base::GetVRegSize();
    constexpr static uint32_t UB_BLOCK = Ops::Base::GetUbBlockSize();

    constexpr static Reg::CastTrait CAST_TRAIT_0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait CAST_TRAIT_1 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    GlobalTensor<int8_t> xGM_;
    GlobalTensor<int8_t> yGM_;
    const HIF4DualCutTilingData* tiling_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, TQUE_DEPTH> inputQue_;
    TBuf<QuePosition::VECCALC> castBuf_;
    TBuf<QuePosition::VECCALC> transposeBuf_;

    int32_t blockIdx_;
    int32_t innerAxisNums_;
    int32_t ubInshapeIdx01_ = -1;
    int32_t ubInshapeIdx02_ = -1;
    int32_t gmShapeIdx01_ = -1;
    int32_t gmShapeIdx02_ = -1;
    int32_t ubOutshapeIdx01_ = -1;
    int32_t ubOutshapeIdx02_ = -1;
    int32_t baseIdx_ = 0;
};

__aicore__ inline void TransposeHIF4::Init(GM_ADDR x, GM_ADDR y, const HIF4DualCutTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    pipe_ = pipe;
    xGM_.SetGlobalBuffer((__gm__ int8_t*)x);
    yGM_.SetGlobalBuffer((__gm__ int8_t*)y);

    innerAxisNums_ = tiling_->innerAxisNums;
    gmShapeIdx01_ = tiling_->outCutId;
    gmShapeIdx02_ = tiling_->inCutId;

    if (tiling_->cutAxisIdx01 != -1) {
        for (int16_t i = 0; i < DUAL_CUT_UB_LEN; i++) {
            if (tiling_->ubShapeId[i] == tiling_->cutAxisIdx01) {
                ubInshapeIdx01_ = i;
            }
        }
        for (int16_t idx = 0; idx < DUAL_CUT_UB_LEN; idx++) {
            if (ubInshapeIdx01_ == tiling_->ubPerm[idx]) {
                ubOutshapeIdx01_ = idx;
            }
        }
    }
    if (tiling_->cutAxisIdx02 != -1) {
        for (int16_t i = 0; i < DUAL_CUT_UB_LEN; i++) {
            if (tiling_->ubShapeId[i] == tiling_->cutAxisIdx02) {
                ubInshapeIdx02_ = i;
            }
        }
        for (int16_t idx = 0; idx < DUAL_CUT_UB_LEN; idx++) {
            if (ubInshapeIdx02_ == tiling_->ubPerm[idx]) {
                ubOutshapeIdx02_ = idx;
            }
        }
    }

    int64_t inputBufSize = tiling_->ubSize;
    int64_t castBufSize = FOUR_BUFF * inputBufSize;
    int64_t transposeBufSize = FOUR_BUFF * inputBufSize;

    pipe_->InitBuffer(inputQue_, DOUBLE_BUFF, inputBufSize);
    pipe_->InitBuffer(castBuf_, castBufSize);
    pipe_->InitBuffer(transposeBuf_, transposeBufSize);

    baseIdx_ = DUAL_CUT_UB_LEN - innerAxisNums_;
}

__aicore__ inline void TransposeHIF4::Process()
{
    if (blockIdx_ >= tiling_->useCoreNum) {
        return;
    }

    int32_t startIter = 0;
    int32_t endIter = 0;
    ComputeIterRange(startIter, endIter);

    LocalTensor<bfloat16_t> castLocal = castBuf_.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> transposeLocal = transposeBuf_.Get<bfloat16_t>();

    for (int32_t iter = startIter; iter < endIter; iter++) {
        ProcessOneIteration(iter, castLocal, transposeLocal);
    }
}

// Compute the [startIter, endIter) gm-iteration window handled by this core.
__aicore__ inline void TransposeHIF4::ComputeIterRange(int32_t& startIter, int32_t& endIter)
{
    startIter = blockIdx_ * tiling_->dimsPerCore;
    if (blockIdx_ == tiling_->useCoreNum - 1) {
        endIter = startIter + tiling_->dimsPerCoreTail;
    } else {
        endIter = startIter + tiling_->dimsPerCore;
    }
}

// Unflatten the linear gm-iteration index into a multi-dim gmIdx (gmBaseIdx offsets the active axes).
__aicore__ inline void TransposeHIF4::UnflattenGmIdx(int32_t iter, int32_t gmBaseIdx, int32_t* gmIdx)
{
    int32_t tmp = iter;
    for (int32_t i = tiling_->outerAxisNums - 1; i >= 0; i--) {
        gmIdx[i] = tmp % tiling_->gmShape[gmBaseIdx + i];
        tmp /= tiling_->gmShape[gmBaseIdx + i];
    }
}

// Compute the GM input/output byte offsets (HIF4 packs 2 elements per byte).
__aicore__ inline void TransposeHIF4::ComputeGmByteOffsets(const int32_t* gmIdx, int32_t gmBaseIdx,
                                                           int64_t& gmInByteOffset, int64_t& gmOutByteOffset)
{
    int64_t inOffset = 0;
    int64_t outOffset = 0;
    for (int32_t i = 0; i < tiling_->outerAxisNums; i++) {
        inOffset += static_cast<int64_t>(gmIdx[i]) * tiling_->gmInStride[gmBaseIdx + i];
        outOffset += static_cast<int64_t>(gmIdx[i]) * tiling_->gmOutStride[gmBaseIdx + i];
    }
    gmInByteOffset = inOffset / HIF4_ELEMENTS_PER_BYTE;
    gmOutByteOffset = outOffset / HIF4_ELEMENTS_PER_BYTE;
}

// Determine the actual ubShape for this iteration, applying cut-axis tail replacement.
__aicore__ inline void TransposeHIF4::ComputeUpdatedUbShape(const int32_t* gmIdx, int32_t gmBaseIdx,
                                                            int32_t* updatedUbInShape, int32_t* updatedUbOutShape)
{
    for (int32_t i = baseIdx_; i < DUAL_CUT_UB_LEN; i++) {
        updatedUbInShape[i] = tiling_->ubShape[i];
        updatedUbOutShape[i] = tiling_->ubShape[tiling_->ubPerm[i]];
    }
    // 获取切分输出尾块
    if (tiling_->cutAxisIdx01 != -1) {
        if (gmIdx[gmShapeIdx01_] == tiling_->gmShape[gmBaseIdx + gmShapeIdx01_] - 1) {
            updatedUbInShape[ubInshapeIdx01_] = tiling_->cutAxisTail01;
            updatedUbOutShape[ubOutshapeIdx01_] = tiling_->cutAxisTail01;
        }
    }
    // 获取切分输入尾块
    if (tiling_->cutAxisIdx02 != -1) {
        if (gmIdx[gmShapeIdx02_] == tiling_->gmShape[gmBaseIdx + gmShapeIdx02_] - 1) {
            updatedUbInShape[ubInshapeIdx02_] = tiling_->cutAxisTail02;
            updatedUbOutShape[ubOutshapeIdx02_] = tiling_->cutAxisTail02;
        }
    }
}

// Compute per-tile element counts (tileB16Elem/planeEleBf16/validBf16Elem) and the UB-out / GM-in strides.
__aicore__ inline void TransposeHIF4::ComputeTileStrides(const int32_t* updatedUbInShape,
                                                         const int32_t* updatedUbOutShape, int64_t& tileB16Elem,
                                                         int64_t& planeEleBf16, int64_t& validBf16Elem,
                                                         int64_t* updatedUbOutStride, int64_t* updatedGmInStride)
{
    int64_t probStride = 1;
    for (int32_t i = DUAL_CUT_UB_LEN - 1; i >= baseIdx_; i--) {
        // 1、计算 tileB16Elem、planeEleBf16
        if (i == DUAL_CUT_UB_LEN - TAIL_TWO) {
            tileB16Elem = CeilAlign(tileB16Elem * updatedUbInShape[i] / HIF4_ELEMENTS_PER_BYTE, UB_BLOCK);
            planeEleBf16 = tileB16Elem;
        } else {
            tileB16Elem *= updatedUbInShape[i];
        }

        // 2、计算 validBf16Elem
        validBf16Elem *= updatedUbInShape[i];

        // 3、计算UB->UB转置的dstStride
        int32_t idx = -1;
        for (int32_t k = 0; k < DUAL_CUT_UB_LEN; k++) {
            if (tiling_->ubPerm[k] == i) {
                idx = k;
            }
        }
        int64_t currentStride = 1;
        for (int32_t j = idx + 1; j < DUAL_CUT_UB_LEN; j++) {
            currentStride *= updatedUbOutShape[j];
        }
        updatedUbOutStride[i] = currentStride;

        // 4、计算 updatedGmInStride
        updatedGmInStride[i] = probStride;
        probStride *= updatedUbOutShape[i];
    }
    tileB16Elem *= HIF4_ELEMENTS_PER_BYTE;
    planeEleBf16 *= HIF4_ELEMENTS_PER_BYTE;
}

// Align the stride of the 3rd-largest (and, for 4 inner axes, 4th-largest) UB-out dimension to UB_BLOCK,
// and derive the element count to cast bf16 -> b4.
__aicore__ inline void TransposeHIF4::AlignOutStrideForB4(const int32_t* updatedUbInShape, int64_t* updatedUbOutStride,
                                                          int64_t* updatedGmInStride, int64_t validBf16Elem,
                                                          int64_t& validBf16ToB4Elem)
{
    // 找出updatedUbOutStride中第3大的位置，对齐UB_BLOCK
    if (innerAxisNums_ < INNNER_AXIS_NUMS_THREE) {
        validBf16ToB4Elem = validBf16Elem;
        return;
    }

    int32_t indices[DUAL_CUT_UB_LEN];
    int64_t values[DUAL_CUT_UB_LEN];
    int32_t validCount = 0;
    for (int32_t i = baseIdx_; i < DUAL_CUT_UB_LEN; i++) {
        indices[validCount] = i;
        values[validCount] = updatedUbOutStride[i];
        validCount++;
    }
    // 从小->大排序
    for (int32_t i = 0; i < validCount - 1; i++) {
        for (int32_t j = 0; j < validCount - 1 - i; j++) {
            if (values[j] > values[j + 1]) {
                int64_t tmpVal = values[j];
                values[j] = values[j + 1];
                values[j + 1] = tmpVal;
                int32_t tmpIdx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmpIdx;
            }
        }
    }
    int32_t thirdLargestIdx = indices[MAIN_TWO];
    int64_t originalThird = values[MAIN_TWO];
    updatedUbOutStride[thirdLargestIdx] = CeilAlign(updatedUbOutStride[thirdLargestIdx] / HIF4_ELEMENTS_PER_BYTE,
                                                    UB_BLOCK) *
                                          HIF4_ELEMENTS_PER_BYTE;
    validBf16ToB4Elem = updatedUbOutStride[thirdLargestIdx] * updatedUbInShape[thirdLargestIdx];
    updatedGmInStride[DUAL_CUT_UB_LEN - TAIL_THREE] = updatedUbOutStride[thirdLargestIdx];
    if (innerAxisNums_ == INNNER_AXIS_NUMS_FOUR) {
        int32_t fourthLargestIdx = indices[MAIN_THREE];
        int64_t originalFourth = values[MAIN_THREE];
        updatedUbOutStride[fourthLargestIdx] = (originalFourth / originalThird) * updatedUbOutStride[thirdLargestIdx];
        validBf16ToB4Elem = updatedUbOutStride[fourthLargestIdx] * updatedUbInShape[fourthLargestIdx];
        updatedGmInStride[DUAL_CUT_UB_LEN - TAIL_FOUR] = updatedUbOutStride[fourthLargestIdx];
    }
}

// Compute the UB->UB transpose srcStride.
__aicore__ inline void TransposeHIF4::ComputeUbInStride(const int32_t* updatedUbInShape, int64_t planeEleBf16,
                                                        int64_t* updatedUbInStride)
{
    updatedUbInStride[DUAL_CUT_UB_LEN - 1] = 1;
    updatedUbInStride[DUAL_CUT_UB_LEN - TAIL_TWO] = updatedUbInShape[DUAL_CUT_UB_LEN - 1];
    if (innerAxisNums_ >= INNNER_AXIS_NUMS_THREE) {
        updatedUbInStride[DUAL_CUT_UB_LEN - TAIL_THREE] = planeEleBf16;
    }
    if (innerAxisNums_ >= INNNER_AXIS_NUMS_FOUR) {
        updatedUbInStride[DUAL_CUT_UB_LEN - TAIL_FOUR] = updatedUbInShape[DUAL_CUT_UB_LEN - TAIL_THREE] * planeEleBf16;
    }
}

// Process a single gm-iteration: derive all tile descriptors, then run the copy/compute pipeline.
__aicore__ inline void TransposeHIF4::ProcessOneIteration(int32_t iter, const LocalTensor<bfloat16_t>& castLocal,
                                                          const LocalTensor<bfloat16_t>& transposeLocal)
{
    int32_t gmBaseIdx = DUAL_CUT_GM_LEN - tiling_->outerAxisNums;

    // ----- Unflatten linear gm-iter into multi-dim gmIdx -----
    int32_t gmIdx[DUAL_CUT_GM_LEN] = {0};
    UnflattenGmIdx(iter, gmBaseIdx, gmIdx);

    // ----- Compute GM offsets (HIF4 element units) -----
    int64_t gmInByteOffset = 0;
    int64_t gmOutByteOffset = 0;
    ComputeGmByteOffsets(gmIdx, gmBaseIdx, gmInByteOffset, gmOutByteOffset);

    // ----- Determine actual ubShape with cut-axis tail -----
    int32_t updatedUbInShape[DUAL_CUT_UB_LEN] = {1, 1, 1, 1};
    int32_t updatedUbOutShape[DUAL_CUT_UB_LEN] = {1, 1, 1, 1};
    ComputeUpdatedUbShape(gmIdx, gmBaseIdx, updatedUbInShape, updatedUbOutShape);

    // ----- Current tile element count -----
    int64_t tileB16Elem = 1;
    int64_t planeEleBf16 = 1;
    int64_t validBf16Elem = 1;
    int64_t updatedUbOutStride[DUAL_CUT_UB_LEN] = {0, 0, 0, 0};
    int64_t updatedGmInStride[DUAL_CUT_UB_LEN] = {0, 0, 0, 0};
    ComputeTileStrides(updatedUbInShape, updatedUbOutShape, tileB16Elem, planeEleBf16, validBf16Elem,
                       updatedUbOutStride, updatedGmInStride);

    int64_t validBf16ToB4Elem = 1;
    AlignOutStrideForB4(updatedUbInShape, updatedUbOutStride, updatedGmInStride, validBf16Elem, validBf16ToB4Elem);

    // 计算UB->UB转置的srcStride
    int64_t updatedUbInStride[DUAL_CUT_UB_LEN] = {0, 0, 0, 0};
    ComputeUbInStride(updatedUbInShape, planeEleBf16, updatedUbInStride);

    RunTilePipeline(gmInByteOffset, gmOutByteOffset, updatedUbInShape, updatedUbOutShape, updatedUbInStride,
                    updatedUbOutStride, updatedGmInStride, tileB16Elem, validBf16Elem, validBf16ToB4Elem, castLocal,
                    transposeLocal);
}

// CopyIn -> Cast fp4->bf16 -> transpose -> Cast bf16->fp4 -> CopyOut, with pipe syncs.
__aicore__ inline void TransposeHIF4::RunTilePipeline(
    int64_t gmInByteOffset, int64_t gmOutByteOffset, const int32_t* updatedUbInShape, const int32_t* updatedUbOutShape,
    const int64_t* updatedUbInStride, const int64_t* updatedUbOutStride, const int64_t* updatedGmInStride,
    int64_t tileB16Elem, int64_t validBf16Elem, int64_t validBf16ToB4Elem, const LocalTensor<bfloat16_t>& castLocal,
    const LocalTensor<bfloat16_t>& transposeLocal)
{
    // copyIn
    CopyIn(gmInByteOffset, updatedUbInShape);

    CastFp4ToBf16(tileB16Elem, castLocal);

    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);

    SimtTranspose(validBf16Elem, updatedUbInShape, updatedUbInStride, updatedUbOutStride, castLocal, transposeLocal);

    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId1);
    WaitFlag<HardEvent::S_V>(eventId1);

    CastBf16ToFp4(validBf16ToB4Elem, castLocal, transposeLocal);

    event_t eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId2);
    WaitFlag<HardEvent::V_MTE3>(eventId2);

    CopyOut(gmOutByteOffset, updatedUbOutShape, updatedGmInStride, castLocal);

    event_t eventId3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId3);
    WaitFlag<HardEvent::MTE3_S>(eventId3);
}

__aicore__ inline void TransposeHIF4::CopyOut(int64_t gmOutByteOffset, const int32_t* updatedUbOutShape,
                                              const int64_t* updatedGmInStride,
                                              const LocalTensor<bfloat16_t>& castLocal)
{
    LocalTensor<int8_t> tmp = (LocalTensor<int8_t>&)castLocal;
    LoopModeParams loopParams;
    loopParams.loop1Size = updatedUbOutShape[DUAL_CUT_UB_LEN - TAIL_THREE];
    loopParams.loop1SrcStride = updatedGmInStride[DUAL_CUT_UB_LEN - TAIL_THREE] / HIF4_ELEMENTS_PER_BYTE;
    loopParams.loop1DstStride = tiling_->moveUbStride[DUAL_CUT_UB_LEN - TAIL_THREE] / HIF4_ELEMENTS_PER_BYTE;

    loopParams.loop2Size = updatedUbOutShape[DUAL_CUT_UB_LEN - TAIL_FOUR];
    loopParams.loop2SrcStride = updatedGmInStride[DUAL_CUT_UB_LEN - TAIL_FOUR] / HIF4_ELEMENTS_PER_BYTE;
    loopParams.loop2DstStride = tiling_->moveUbStride[DUAL_CUT_UB_LEN - TAIL_FOUR] / HIF4_ELEMENTS_PER_BYTE;

    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = updatedUbOutShape[DUAL_CUT_UB_LEN - TAIL_TWO];
    copyOutParams.blockLen = updatedUbOutShape[DUAL_CUT_UB_LEN - 1] / HIF4_ELEMENTS_PER_BYTE;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (tiling_->moveUbStride[DUAL_CUT_UB_LEN - TAIL_TWO] -
                               updatedUbOutShape[DUAL_CUT_UB_LEN - 1]) /
                              HIF4_ELEMENTS_PER_BYTE;

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad<int8_t, PaddingMode::Compact>(yGM_[gmOutByteOffset], tmp, copyOutParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

__aicore__ inline void TransposeHIF4::CopyIn(int64_t gmInByteOffset, const int32_t* updatedUbInShape)
{
    LocalTensor<int8_t> inputLocal = inputQue_.AllocTensor<int8_t>();

    DataCopyPadExtParams<int8_t> padParams{true, 0, 0, 0};
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = updatedUbInShape[DUAL_CUT_UB_LEN - TAIL_TWO];
    copyInParams.blockLen = updatedUbInShape[DUAL_CUT_UB_LEN - 1] / HIF4_ELEMENTS_PER_BYTE;
    copyInParams.srcStride = (tiling_->ubInStride[DUAL_CUT_UB_LEN - TAIL_TWO] - updatedUbInShape[DUAL_CUT_UB_LEN - 1]) /
                             HIF4_ELEMENTS_PER_BYTE;
    copyInParams.dstStride = 0;

    LoopModeParams loopParams;
    loopParams.loop1Size = updatedUbInShape[DUAL_CUT_UB_LEN - TAIL_THREE];
    loopParams.loop1SrcStride = tiling_->ubInStride[DUAL_CUT_UB_LEN - TAIL_THREE] / HIF4_ELEMENTS_PER_BYTE;
    loopParams.loop1DstStride = CeilAlign(copyInParams.blockCount * copyInParams.blockLen, UB_BLOCK);
    loopParams.loop2Size = updatedUbInShape[DUAL_CUT_UB_LEN - TAIL_FOUR];
    loopParams.loop2SrcStride = tiling_->ubInStride[DUAL_CUT_UB_LEN - TAIL_FOUR] / HIF4_ELEMENTS_PER_BYTE;
    loopParams.loop2DstStride = loopParams.loop1Size * loopParams.loop1DstStride;

    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
#if defined(__DAV_920R1__)
    DataCopyPad<int8_t, PaddingMode::Compact>(inputLocal, xGM_[gmInByteOffset], copyInParams, padParams, true);
#else
    DataCopyPad<int8_t, PaddingMode::Compact>(inputLocal, xGM_[gmInByteOffset], copyInParams, padParams);
#endif
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    inputQue_.EnQue(inputLocal);
}

__aicore__ inline void TransposeHIF4::CastFp4ToBf16(int64_t elemCount, const LocalTensor<bfloat16_t>& castLocal)
{
    LocalTensor<int8_t> inputLocal = inputQue_.DeQue<int8_t>();
    __ubuf__ fp4x2_e2m1_t* inputAddr = reinterpret_cast<__ubuf__ fp4x2_e2m1_t*>(inputLocal.GetPhyAddr());
    __ubuf__ bfloat16_t* castAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(castLocal.GetPhyAddr());

    uint32_t remainLen = elemCount;
    constexpr uint32_t VL_ELEMS_BFLOAT16 = VL_SIZE / sizeof(bfloat16_t);
    constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(float32_t);
    uint16_t loopCount = CeilDiv(elemCount, VL_ELEMS_BFLOAT16);

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<bfloat16_t> castReg;
        Reg::RegTensor<fp4x2_e2m1_t> srcReg;

        for (uint16_t i = 0; i < loopCount; i++) {
            mask = Reg::UpdateMask<bfloat16_t>(remainLen);
            Reg::LoadAlign<fp4x2_e2m1_t, Reg::LoadDist::DIST_UNPACK4_B8>(srcReg, inputAddr + i * VL_ELEMS_FLOAT);
            Reg::Cast<bfloat16_t, fp4x2_e2m1_t, CAST_TRAIT_0>(castReg, srcReg, mask);
            Reg::StoreAlign(castAddr + i * VL_ELEMS_BFLOAT16, castReg, mask);
        }
    }
    inputQue_.FreeTensor(inputLocal);
}

__aicore__ inline void TransposeHIF4::CastBf16ToFp4(int64_t elemCount, const LocalTensor<bfloat16_t>& castLocal,
                                                    const LocalTensor<bfloat16_t>& transposeLocal)
{
    __ubuf__ bfloat16_t* transposeAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(transposeLocal.GetPhyAddr());
    __ubuf__ fp4x2_e2m1_t* castAddr = reinterpret_cast<__ubuf__ fp4x2_e2m1_t*>(castLocal.GetPhyAddr());

    uint32_t remainLen = elemCount;
    constexpr uint32_t VL_ELEMS_BFLOAT16 = VL_SIZE / sizeof(bfloat16_t);
    constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(float32_t);
    uint16_t loopCount = CeilDiv(elemCount, VL_ELEMS_BFLOAT16);

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<bfloat16_t> transposeReg;
        Reg::RegTensor<fp4x2_e2m1_t> castReg;

        for (uint16_t i = 0; i < loopCount; i++) {
            mask = Reg::UpdateMask<bfloat16_t>(remainLen);
            Reg::LoadAlign(transposeReg, transposeAddr + i * VL_ELEMS_BFLOAT16);
            Reg::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_TRAIT_1>(castReg, transposeReg, mask);
            Reg::StoreAlign<fp4x2_e2m1_t, Reg::StoreDist::DIST_PACK4_B32>(castAddr + i * VL_ELEMS_FLOAT, castReg, mask);
        }
    }
}

__aicore__ inline void TransposeHIF4::SimtTranspose(int64_t elemCount, const int32_t* updatedUbInShape,
                                                    const int64_t* updatedUbInStride, const int64_t* updatedUbOutStride,
                                                    const LocalTensor<bfloat16_t>& castLocal,
                                                    const LocalTensor<bfloat16_t>& transposeLocal)
{
    int32_t shape[DUAL_CUT_UB_LEN] = {1, 1, 1, 1};
    int64_t srcStride[DUAL_CUT_UB_LEN] = {0, 0, 0, 0};
    int64_t dstStride[DUAL_CUT_UB_LEN] = {0, 0, 0, 0};
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t ubIdx = baseIdx_ + i;
        shape[i] = updatedUbInShape[ubIdx];
        srcStride[i] = updatedUbInStride[ubIdx];
        dstStride[i] = updatedUbOutStride[ubIdx];
    }

    __ubuf__ bfloat16_t* srcPtr = (__ubuf__ bfloat16_t*)(castLocal.GetPhyAddr());
    __ubuf__ bfloat16_t* dstPtr = (__ubuf__ bfloat16_t*)(transposeLocal.GetPhyAddr());
    if (innerAxisNums_ == INNNER_AXIS_NUMS_TWO) {
        for (int64_t idx = 0; idx < elemCount; idx++) {
            int64_t yIdx = idx;
            int64_t coord0 = yIdx / shape[1];
            int64_t coord1 = yIdx % shape[1];
            int64_t srcIdx = coord0 * srcStride[0] + coord1 * srcStride[1];
            int64_t dstIdx = coord0 * dstStride[0] + coord1 * dstStride[1];
            *(dstPtr + dstIdx) = *(srcPtr + srcIdx);
        }
    } else if (innerAxisNums_ == INNNER_AXIS_NUMS_THREE) {
        for (int64_t idx = 0; idx < elemCount; idx++) {
            int64_t yIdx = idx;
            int64_t coord0 = yIdx / (shape[1] * shape[MAIN_TWO]);
            int64_t coord1 = (yIdx / shape[MAIN_TWO]) % shape[1];
            int64_t coord2 = yIdx % shape[MAIN_TWO];
            int64_t srcIdx = coord0 * srcStride[0] + coord1 * srcStride[1] + coord2 * srcStride[2];
            int64_t dstIdx = coord0 * dstStride[0] + coord1 * dstStride[1] + coord2 * dstStride[2];
            *(dstPtr + dstIdx) = *(srcPtr + srcIdx);
        }
    } else {
        for (int64_t idx = 0; idx < elemCount; idx++) {
            int64_t yIdx = idx;
            int64_t coord0 = yIdx / (shape[1] * shape[MAIN_TWO] * shape[MAIN_THREE]);
            int64_t coord1 = (yIdx / (shape[MAIN_TWO] * shape[MAIN_THREE])) % shape[1];
            int64_t coord2 = (yIdx / shape[MAIN_THREE]) % shape[MAIN_TWO];
            int64_t coord3 = yIdx % shape[MAIN_THREE];
            int64_t srcIdx = coord0 * srcStride[0] + coord1 * srcStride[1] + coord2 * srcStride[2] +
                             coord3 * srcStride[3];
            int64_t dstIdx = coord0 * dstStride[0] + coord1 * dstStride[1] + coord2 * dstStride[2] +
                             coord3 * dstStride[3];
            *(dstPtr + dstIdx) = *(srcPtr + srcIdx);
        }
    }
}
} // namespace Transpose
#endif // TRANSPOSE_B4_H
