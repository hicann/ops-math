/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mirror_pad_symmetric.h
 * \brief mirror_pad_symmetric
 */

#ifndef ASCENDC_MIRROR_PAD_SYMMETRIC_H
#define ASCENDC_MIRROR_PAD_SYMMETRIC_H

#include <cmath>
#include <cstdint>
#include "kernel_operator.h"
#include "mirror_pad_common.h"
namespace MirrorPad {
using namespace AscendC;

template <typename T> class KernelMirrorPadSymmetric {
public:
    __aicore__ inline KernelMirrorPadSymmetric() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, MirrorPadACTilingData tilingData)
    {
        inputGm_.SetGlobalBuffer((__gm__ T *)(x));
        yGm_.SetGlobalBuffer((__gm__ T *)(y));

        blockId_ = GetBlockIdx();
        blockNums_ = GetBlockNum();
        eventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        inputDims_ = tilingData.inputDims;
        totalOuterLoop_ = tilingData.totalOuterLoop;
        maxLoopInCore_ = tilingData.maxLoopInCore;
        tileAxisOuter_ = tilingData.tileAxisOuter;
        tileAxisInner_ = tilingData.tileAxisInner;
        tileAxisRealLen_ = tilingData.tileAxisRealLen;
        mainFrontPad_ = tilingData.mainFrontPad;
        mainBackPad_ = tilingData.mainBackPad == tileAxisInner_ ? 0 : tilingData.mainBackPad;
        tileDim_ = tilingData.tileDim;
        ubDataNum_ = tilingData.ubDataNum;
        frontPaddingLoop_ = tilingData.frontPaddingLoop;
        backPaddingLoop_ = tilingData.backPaddingLoop;
        tailsize_ = tilingData.tailsize == 0 ? tileAxisInner_ : tilingData.tailsize;
        lastLoopBeforePad_ = tilingData.lastLoopBeforePad;
        firstMainDataNum_ = tilingData.firstMainDataNum;
        lastMainDataNum_ = tilingData.lastMainDataNum == 0 ? tileAxisInner_ : tilingData.lastMainDataNum;

        CopyTilingArray(frontPads_, tilingData.frontPads, inputDims_);
        CopyTilingArray(backPads_, tilingData.backPads, inputDims_);
        CopyTilingArray(inputShapes_, tilingData.inputShapes, inputDims_);
        CopyTilingArray(afterPadShapes_, tilingData.afterPadShapes, inputDims_);
        CopyTilingArray(inputStrides_, tilingData.inputStrides, inputDims_);
        CopyTilingArray(afterPadStrides_, tilingData.afterPadStrides, inputDims_);

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, ubDataNum_ * sizeof(T));
        InitCopyParams();
    }

    __aicore__ inline void InitCopyParams()
    {
        dmaFullPadParam_ = InitPadCopyParams(inputShapes_[inputDims_ - 1], constValue_);
        firstDataPieceParam_ = InitFirstDataPieceParams(inputShapes_[inputDims_ - 1],
            afterPadStrides_[inputDims_ - TWO_DIM], firstMainDataNum_, constValue_);
        mainDataPieceParam_ = InitMainDataPieceParams(inputShapes_[inputDims_ - 1],
            afterPadStrides_[inputDims_ - TWO_DIM], tileAxisInner_, constValue_);
        lastDataPieceParam_ = InitLastDataPieceParams(inputShapes_[inputDims_ - 1],
            afterPadStrides_[inputDims_ - TWO_DIM], lastMainDataNum_, constValue_);
        if (tileDim_ == inputDims_ - TWO_DIM) {
            ubOutputLen_ = tileAxisInner_ * afterPadShapes_[inputDims_ - 1];
            tailOutputLen_ = tailsize_ * afterPadShapes_[inputDims_ - 1];
            copyParams_ = { static_cast<uint16_t>(1), static_cast<uint32_t>(ubOutputLen_ * sizeof(T)),
                static_cast<uint32_t>(0), 0, 0 };

            copyParamsTail_ = { static_cast<uint16_t>(1), static_cast<uint32_t>(tailOutputLen_ * sizeof(T)),
                static_cast<uint32_t>(0), 0, 0 };
        }
    }

    __aicore__ inline void MoveGMBackPad2UB(uint64_t &srcStart, uint32_t dstStart, const LocalTensor<T> &outBuffer,
        uint32_t loopSize)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        for (uint32_t k = 0; k < loopSize; ++k) {
            DataCopy(outBuffer[dstStart + lastAxisFrontpad], inputGm_[srcStart], dmaFullPadParam_);
            srcStart -= inputShapes_[inputDims_ - 1];
            dstStart += afterPadShapes_[inputDims_ - 1];
        }
    }

    __aicore__ inline void MoveGMFrontPad2UB(uint64_t &srcStart, uint32_t &dstStart, const LocalTensor<T> &outBuffer,
        uint32_t loopSize)
    {
        uint64_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        for (uint32_t k = 0; k < loopSize; ++k) {
            DataCopy(outBuffer[dstStart + lastAxisFrontpad], inputGm_[srcStart], dmaFullPadParam_);
            srcStart -= inputShapes_[inputDims_ - 1];
            dstStart += afterPadShapes_[inputDims_ - 1];
        }
    }

    struct GetDimSymPadIdx {
        __aicore__ inline GetDimSymPadIdx() {}
        __aicore__ inline void operator () (uint32_t dimIdx[], uint16_t &validNum, uint32_t currentCount,
            uint32_t currentFrontPad, uint32_t currentbackPad, uint32_t currentDimShape)
        {
            dimIdx[0] = currentCount + currentFrontPad;
            if (currentCount < currentFrontPad) {
                dimIdx[validNum] = currentFrontPad - currentCount - 1;
                ++validNum;
            }
            if (currentDimShape - currentCount - 1 < currentbackPad) {
                dimIdx[validNum] = currentFrontPad + currentDimShape * TWO_DIM - currentCount - 1;
                ++validNum;
            }
        }
    };

    __aicore__ inline void CustomCopyOut(const uint32_t counts[], uint16_t highDims, const LocalTensor<T> &outBuffer,
        DataCopyExtParams &copyParams, const uint64_t offset)
    {
        switch (highDims) {
            case 0:
                CopyOut2MultiPos0Dim<T>(outBuffer, copyParams, offset, yGm_);
                break;
            case 1:
                CopyOut2MultiPos1Dim<T, GetDimSymPadIdx>(counts, outBuffer, copyParams, offset, yGm_, frontPads_,
                    backPads_, inputShapes_, afterPadStrides_);
                break;
            case TWO_DIM:
                CopyOut2MultiPos2Dim<T, GetDimSymPadIdx>(counts, outBuffer, copyParams, offset, yGm_, frontPads_,
                    backPads_, inputShapes_, afterPadStrides_);
                break;
            case THREE_DIM:
                CopyOut2MultiPos3Dim<T, GetDimSymPadIdx>(counts, outBuffer, copyParams, offset, yGm_, frontPads_,
                    backPads_, inputShapes_, afterPadStrides_);
                break;
            case FOUR_DIM:
                CopyOut2MultiPos4Dim<T, GetDimSymPadIdx>(counts, outBuffer, copyParams, offset, yGm_, frontPads_,
                    backPads_, inputShapes_, afterPadStrides_);
                break;
            case FIVE_DIM:
                CopyOut2MultiPos5Dim<T, GetDimSymPadIdx>(counts, outBuffer, copyParams, offset, yGm_, frontPads_,
                    backPads_, inputShapes_, afterPadStrides_);
                break;
            case SIX_DIM:
                CopyOut2MultiPos6Dim<T, GetDimSymPadIdx>(counts, outBuffer, copyParams, offset, yGm_, frontPads_,
                    backPads_, inputShapes_, afterPadStrides_);
                break;
            default:
                break;
        }
    }
    __aicore__ inline void FrontPadCaseSym(const uint32_t counts[], uint64_t &dstStart, uint64_t &frontPadSrcStart)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        uint32_t ubOffset = 0;
        if (tileAxisInner_ == 1) {
            OneRowMoveAlignCopy(lastAxisFrontpad, inputShapes_[inputDims_ - 1], frontPadSrcStart, srcLocal, inputGm_);
        } else {
            MoveGMFrontPad2UB(frontPadSrcStart, ubOffset, srcLocal, tileAxisInner_);
        }
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tileAxisInner_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParams_, dstStart);
        dstStart += ubOutputLen_;
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void FrontPadAndDataAndBackPadCaseSym(const uint32_t counts[], uint64_t &srcStart,
        uint64_t &dstStart, uint64_t &frontPadSrcStart, uint64_t &backPadSrcStart, uint32_t loopIdx)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        uint32_t srcLastAxisLen = inputShapes_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        uint32_t ubOffset = 0;
        MoveGMFrontPad2UB(frontPadSrcStart, ubOffset, srcLocal, mainFrontPad_);
        DataCopy(srcLocal[ubOffset + lastAxisFrontpad], inputGm_[srcStart], firstDataPieceParam_);
        MoveGMBackPad2UB(backPadSrcStart, (firstMainDataNum_ + mainFrontPad_) * afterPadShapes_[inputDims_ - 1],
            srcLocal, mainBackPad_);
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tileAxisInner_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        if (loopIdx == tileAxisOuter_ - 1) {
            CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParamsTail_, dstStart);
        } else {
            CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParams_, dstStart);
        }
        dstStart += ubOutputLen_;
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void FrontPadAndDataCaseSym(const uint32_t counts[], uint64_t &srcStart, uint64_t &dstStart,
        uint64_t &frontPadSrcStart)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        uint32_t srcLastAxisLen = inputShapes_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        uint32_t ubOffset = 0;
        MoveGMFrontPad2UB(frontPadSrcStart, ubOffset, srcLocal, mainFrontPad_);
        DataCopy(srcLocal[ubOffset + lastAxisFrontpad], inputGm_[srcStart], firstDataPieceParam_);
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tileAxisInner_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        srcStart += firstMainDataNum_ * srcLastAxisLen;
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParams_, dstStart);
        dstStart += ubOutputLen_;
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void OriginalDataCaseSym(const uint32_t counts[], uint64_t &srcStart, uint64_t &dstStart)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        uint32_t srcLastAxisLen = inputShapes_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        if (tileAxisInner_ == 1) {
            OneRowMoveAlignCopy(lastAxisFrontpad, srcLastAxisLen, srcStart, srcLocal, inputGm_);
        } else {
            DataCopy(srcLocal[lastAxisFrontpad], inputGm_[srcStart], mainDataPieceParam_);
        }
        srcStart += tileAxisInner_ * srcLastAxisLen;
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tileAxisInner_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParams_, dstStart);
        dstStart += ubOutputLen_;
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void DataAndBackPadCaseSym(const uint32_t counts[], uint64_t &srcStart, uint64_t &dstStart,
        uint64_t &backPadSrcStart, uint32_t loopIdx)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        uint32_t srcLastAxisLen = inputShapes_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        DataCopy(srcLocal[lastAxisFrontpad], inputGm_[srcStart], lastDataPieceParam_);
        srcStart += (lastMainDataNum_ * srcLastAxisLen);
        MoveGMBackPad2UB(backPadSrcStart, lastMainDataNum_ * afterPadShapes_[inputDims_ - 1], srcLocal, mainBackPad_);
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tileAxisInner_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        if (loopIdx == tileAxisOuter_ - 1) {
            CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParamsTail_, dstStart);
        } else {
            CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParams_, dstStart);
        }
        dstStart += ubOutputLen_;
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void BackPadCaseSym(const uint32_t counts[], uint64_t &dstStart, uint64_t &backPadSrcStart)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        if (tileAxisInner_ == 1) {
            OneRowMoveAlignCopy(lastAxisFrontpad, inputShapes_[inputDims_ - 1], backPadSrcStart, srcLocal, inputGm_);
        } else {
            MoveGMBackPad2UB(backPadSrcStart, 0, srcLocal, tileAxisInner_);
        }
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tileAxisInner_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParams_, dstStart);
        dstStart += ubOutputLen_;
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void BackPadTailCaseSym(const uint32_t counts[], uint64_t &dstStart, uint64_t &backPadSrcStart)
    {
        uint32_t lastAxisFrontpad = frontPads_[inputDims_ - 1];
        uint32_t lastAxisBackpad = backPads_[inputDims_ - 1];
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        if (tileAxisInner_ == 1) {
            OneRowMoveAlignCopy(lastAxisFrontpad, inputShapes_[inputDims_ - 1], backPadSrcStart, srcLocal, inputGm_);
        } else {
            MoveGMBackPad2UB(backPadSrcStart, 0, srcLocal, tailsize_);
        }
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        AscendC::Simt::VF_CALL<SimtSymModeCompute<T>>(
        AscendC::Simt::Dim3{THREAD_DIM},  tailsize_, lastAxisFrontpad, lastAxisBackpad, (__ubuf__ T*)srcLocal.GetPhyAddr(), inputShapes_[inputDims_ - 1], afterPadShapes_[inputDims_ - 1]);
        inQueue_.EnQue(srcLocal);
        LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
        CustomCopyOut(counts, inputDims_ - TWO_DIM, dstLocal, copyParamsTail_, dstStart);
        inQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void SubProcessCutHOpt()
    {
        uint64_t hwBlockSize = inputShapes_[inputDims_ - 1] * inputShapes_[inputDims_ - TWO_DIM];
        uint64_t frontPadOffset = (frontPads_[inputDims_ - TWO_DIM] - 1) * inputShapes_[inputDims_ - 1];
        uint64_t backPadOffset = (tileAxisRealLen_ - 1) * inputShapes_[inputDims_ - 1];
        uint64_t innerOffset = inputShapes_[inputDims_ - 1] * tileAxisInner_;
        uint64_t firstMainData = inputShapes_[inputDims_ - 1] * firstMainDataNum_;
        uint64_t lastMainBackPad = inputShapes_[inputDims_ - 1] * mainBackPad_;
        for (uint32_t i = 0; i < maxLoopInCore_; ++i) {
            uint64_t currentIdx = i * GetBlockNum() + GetBlockIdx();
            if (currentIdx >= totalOuterLoop_ * tileAxisOuter_) {
                break;
            }
            uint64_t hwIdx = currentIdx / tileAxisOuter_;
            uint64_t innerIdx = currentIdx % tileAxisOuter_;
            uint32_t counts[SIX_DIM] = {0,0,0,0,0,0};
            if (inputDims_ >= THREE_DIM) {
                GetCurrentPosition(counts, hwIdx, inputStrides_, inputDims_);
            }

            uint64_t srcStart = hwIdx * hwBlockSize;
            uint64_t dstStart = 0;
            uint64_t frontPadSrcStart = hwIdx * hwBlockSize + frontPadOffset;
            uint64_t backPadSrcStart = hwIdx * hwBlockSize + backPadOffset;

            if (innerIdx < frontPaddingLoop_) {
                dstStart += (ubOutputLen_ * innerIdx);
                frontPadSrcStart -= (innerOffset * innerIdx);
                FrontPadCaseSym(counts, dstStart, frontPadSrcStart);
            } else if (innerIdx == frontPaddingLoop_ && (tileAxisRealLen_ + mainFrontPad_ < tileAxisInner_)) {
                dstStart += (ubOutputLen_ * frontPaddingLoop_);
                frontPadSrcStart -= (innerOffset * frontPaddingLoop_);
                FrontPadAndDataAndBackPadCaseSym(counts, srcStart, dstStart, frontPadSrcStart, backPadSrcStart,
                    innerIdx);
            } else if (innerIdx == frontPaddingLoop_) {
                dstStart += (ubOutputLen_ * frontPaddingLoop_);
                frontPadSrcStart -= (innerOffset * frontPaddingLoop_);
                FrontPadAndDataCaseSym(counts, srcStart, dstStart, frontPadSrcStart);
            } else if ((frontPaddingLoop_ < innerIdx) && (innerIdx < lastLoopBeforePad_)) {
                dstStart += (ubOutputLen_ * innerIdx);
                srcStart += (firstMainData + (innerIdx - frontPaddingLoop_ - 1) * innerOffset);
                OriginalDataCaseSym(counts, srcStart, dstStart);
            } else if (innerIdx == lastLoopBeforePad_) {
                dstStart += (ubOutputLen_ * innerIdx);
                srcStart += (firstMainData + (innerIdx - frontPaddingLoop_ - 1) * innerOffset);
                DataAndBackPadCaseSym(counts, srcStart, dstStart, backPadSrcStart, innerIdx);
            } else if (innerIdx >= lastLoopBeforePad_ && innerIdx < tileAxisOuter_ - 1) {
                dstStart += (ubOutputLen_ * innerIdx);
                backPadSrcStart -= (lastMainBackPad + (innerIdx - lastLoopBeforePad_ - 1) * innerOffset);
                BackPadCaseSym(counts, dstStart, backPadSrcStart);
            } else if (innerIdx == tileAxisOuter_ - 1) {
                dstStart += (ubOutputLen_ * innerIdx);
                backPadSrcStart -= (lastMainBackPad + (innerIdx - lastLoopBeforePad_ - 1) * innerOffset);
                BackPadTailCaseSym(counts, dstStart, backPadSrcStart);
            }
        }
    }

    __aicore__ inline void Process()
    {
        // N, C, H, W
        SubProcessCutHOpt();
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ubBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;

    AscendC::GlobalTensor<T> inputGm_, yGm_;

    DataCopyExtParams copyParams_;
    DataCopyExtParams copyParamsTail_;

    MultiCopyParams<T, NDDMA_DIM> firstDataPieceParam_;
    MultiCopyParams<T, NDDMA_DIM> dmaFullPadParam_;
    MultiCopyParams<T, NDDMA_DIM> mainDataPieceParam_;
    MultiCopyParams<T, NDDMA_DIM> lastDataPieceParam_;

    uint32_t blockId_;
    uint32_t blockNums_;
    event_t eventId_;

    uint32_t ubDataNum_{ 1 };
    uint32_t frontPads_[8] = {0,0,0,0,0,0,0,0};
    uint32_t backPads_[8] = {0,0,0,0,0,0,0,0};
    uint32_t inputShapes_[8] = {0,0,0,0,0,0,0,0};
    uint32_t afterPadShapes_[8] = {0,0,0,0,0,0,0,0};
    uint64_t inputStrides_[8] = {1,1,1,1,1,1,1,1};
    uint64_t afterPadStrides_[8] = {1,1,1,1,1,1,1,1};
    uint32_t inputDims_{ 1 };
    int32_t tileDim_{ -1 };
    uint32_t maxLoopInCore_{ 1 };
    uint32_t totalOuterLoop_{ 1 };
    uint32_t tileAxisOuter_{ 1 };
    uint32_t tileAxisInner_{ 1 };
    uint32_t frontPaddingLoop_{ 0 };
    uint32_t backPaddingLoop_{ 0 };
    uint32_t mainFrontPad_{ 0 };
    uint32_t mainBackPad_{ 0 };
    uint32_t tileAxisRealLen_{ 1 };
    uint32_t tailsize_{ 0 };
    uint32_t lastLoopBeforePad_{ 0 };
    T constValue_{ 0 };
    uint32_t firstMainDataNum_{ 1 };
    uint32_t lastMainDataNum_{ 1 };
    uint32_t ubOutputLen_{ 1 };
    uint32_t tailOutputLen_{ 1 };
};
} // namespace end
#endif