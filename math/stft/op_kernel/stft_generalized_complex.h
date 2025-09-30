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

/*!
 * \file stft_generalized_complex.h
 * \brief
 */
#ifndef STFT_GENERALIZED_COMPLEX_H
#define STFT_GENERALIZED_COMPLEX_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace STFTND {
using namespace AscendC;
using namespace matmul;

template <typename T, int32_t bufferNum, const MatmulConfig& MM_CFG>
class STFTGeneralizedComplex {
public:
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> aType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true> bType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> cType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> biasType;

    Matmul<aType, bType, cType, biasType, MM_CFG> mm0;
    Matmul<aType, bType, cType, biasType, MM_CFG> mm1;
    Matmul<aType, bType, cType, biasType, MM_CFG> mm2;
    Matmul<aType, bType, cType, biasType, MM_CFG> mm3;
    Matmul<aType, bType, cType, biasType, MM_CFG> mm;

    static constexpr int32_t COMPLEX_COEFFICIENT = 2;
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t BLOCK_FOR_ONE_REPEAT = 8;
    static constexpr int64_t REPEAT_NUM_FOR_FP32 = 64;
    static constexpr int32_t FLOAT_SIZE = 4;
    static constexpr int32_t NEGATIVE_TWO = -2;
    static constexpr int32_t POSITIVE_TWO = 2;

    __aicore__ inline STFTGeneralizedComplex(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR window, GM_ADDR y, GM_ADDR workspace, STFTGeneralizedTilingData* tilingData, TPipe* pipeIn)
    {
        pipe = pipeIn;
        tiling = tilingData;

        inputGm.SetGlobalBuffer(
            (__gm__ T*)x, tiling->batch *
                              ((tiling->inputSize + tiling->nfft) * COMPLEX_COEFFICIENT * sizeof(T) + BLOCK_SIZE - 1) /
                              BLOCK_SIZE * BLOCK_SIZE / sizeof(T));
        size_t splitWindowWorkspaceSize = tiling->batch * tiling->matmulN * tiling->nfftAlign;
        size_t splitWindowWorkspaceSizeAlign =
            (((splitWindowWorkspaceSize * sizeof(T) * COMPLEX_COEFFICIENT + WORKSPACE_ALIGN_SIZE - 1) /
              WORKSPACE_ALIGN_SIZE) *
             WORKSPACE_ALIGN_SIZE) /
            sizeof(T);

        splitRealWindowGm.SetGlobalBuffer((__gm__ T*)workspace, splitWindowWorkspaceSize);
        splitImagWindowGm.SetGlobalBuffer((__gm__ T*)workspace + splitWindowWorkspaceSize, splitWindowWorkspaceSize);

        size_t matmulWorkspaceSize = tiling->batch * tiling->matmulM * tiling->matmulN;

        aRealGm.SetGlobalBuffer((__gm__ T*)workspace + splitWindowWorkspaceSizeAlign, matmulWorkspaceSize);
        aImagGm.SetGlobalBuffer(
            (__gm__ T*)workspace + splitWindowWorkspaceSizeAlign + matmulWorkspaceSize, matmulWorkspaceSize);
        bRealGm.SetGlobalBuffer(
            (__gm__ T*)workspace + splitWindowWorkspaceSizeAlign + matmulWorkspaceSize * DOUBLE_BUFFER,
            matmulWorkspaceSize);
        bImagGm.SetGlobalBuffer(
            (__gm__ T*)workspace + splitWindowWorkspaceSizeAlign + matmulWorkspaceSize * 3, matmulWorkspaceSize);
        outputGm.SetGlobalBuffer((__gm__ T*)y, tiling->batch * tiling->matmulM * tiling->matmulN * DOUBLE_BUFFER);
        a1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(window), tiling->matmulM * tiling->nfftAlign);
        a2Global.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(window) + tiling->matmulM * tiling->nfftAlign,
            tiling->matmulM * tiling->nfftAlign);

        size_t ubAlignBufferSize = (tiling->nFactorUbFormer * COMPLEX_COEFFICIENT + REPEAT_NUM_FOR_FP32 - 1) /
                                   REPEAT_NUM_FOR_FP32 * REPEAT_NUM_FOR_FP32 * sizeof(T);

        pipe->InitBuffer(inCopy, bufferNum, ubAlignBufferSize);
        pipe->InitBuffer(realOutCopy, bufferNum, ubAlignBufferSize);
        pipe->InitBuffer(imagOutCopy, bufferNum, ubAlignBufferSize);

        maskCount = tiling->maskUBSize / REPEAT_SIZE * REPEAT_SIZE / sizeof(int32_t); // 256B aligned
        pipe->InitBuffer(aRealPingUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(aRealPongUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(aImagPingUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(aImagPongUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(bRealPingUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(bRealPongUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(bImagPingUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(bImagPongUB, maskCount / DOUBLE_BUFFER * sizeof(T));
        pipe->InitBuffer(complexPingUB, maskCount * sizeof(T));
        pipe->InitBuffer(complexPongUB, maskCount * sizeof(T));
        pipe->InitBuffer(maskUB, tiling->maskUBSize);
        pipe->InitBuffer(tempUB, tiling->maskUBSize);
        aRealPing = aRealPingUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        aRealPong = aRealPongUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        aImagPing = aImagPingUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        aImagPong = aImagPongUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        bRealPing = bRealPingUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        bRealPong = bRealPongUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        bImagPing = bImagPingUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        bImagPong = bImagPongUB.template Get<T>(maskCount / DOUBLE_BUFFER);
        complexPing = complexPingUB.template Get<T>(maskCount);
        complexPong = complexPongUB.template Get<T>(maskCount);
    }

    __aicore__ inline void Process()
    {
        auto blockIdx = GetBlockIdx();
        uint32_t nIdx = blockIdx % tiling->matmulNCoreNum;
        uint32_t nFactor = tiling->matmulNCoreFactor;
        uint32_t nOffset = nIdx * nFactor;
        uint32_t mIdx = (blockIdx / tiling->matmulNCoreNum) % tiling->matmulMCoreNum;
        uint32_t mFactor = tiling->matmulMCoreFactor;
        uint32_t mOffset = mIdx * mFactor;
        uint32_t bIdx = (blockIdx / tiling->matmulNCoreNum / tiling->matmulMCoreNum) % tiling->batchCoreNum;
        uint32_t bFactor = tiling->batchCoreFactor;
        uint32_t bOffset = bIdx * bFactor;
        bool isTailM = false;
        bool isTailN = false;

        if (nIdx >= tiling->matmulNCoreNum - tiling->matmulNTailCoreNum) {
            nOffset = (tiling->matmulNCoreNum - tiling->matmulNTailCoreNum) * nFactor;
            nFactor = tiling->matmulN % nFactor;
            isTailN = true;
        }
        if (mIdx >= tiling->matmulMCoreNum - tiling->matmulMTailCoreNum) {
            mOffset = (tiling->matmulMCoreNum - tiling->matmulMTailCoreNum) * mFactor +
                      (mIdx + tiling->matmulMTailCoreNum - tiling->matmulMCoreNum) * (mFactor - 1);
            mFactor = mFactor - 1;
            isTailM = true;
        }
        if (bIdx >= tiling->batchCoreNum - tiling->batchTailCoreNum) {
            bOffset = (tiling->batchCoreNum - tiling->batchTailCoreNum) * bFactor +
                      (bIdx + tiling->batchTailCoreNum - tiling->batchCoreNum) * (bFactor - 1);
            bFactor = bFactor - 1;
        }

        if (!isTailM) {
            mm = !isTailN ? mm0 : mm1;
        } else {
            mm = !isTailN ? mm2 : mm3;
        }

        for (uint32_t i = 0; i < bFactor; i++) {
            int64_t inputOffset = (bOffset + i) * (tiling->inputSize + tiling->nfft) * COMPLEX_COEFFICIENT +
                                  nOffset * tiling->hopLength * COMPLEX_COEFFICIENT;
            int64_t realSplitWindowOffset = ((bOffset + i) * tiling->matmulN + nOffset) * tiling->nfftAlign;
            int64_t imagSplitWindowOffset = realSplitWindowOffset;
            int64_t outputOffset =
                (((bOffset + i) * tiling->matmulM + mOffset) * tiling->matmulN + nOffset) * DOUBLE_BUFFER;
            int64_t realOffset = (bOffset + i) * tiling->matmulM * tiling->matmulN + mOffset * tiling->matmulN +
                                 nIdx * mFactor * tiling->matmulNCoreFactor;
            int64_t imagOffset = realOffset;
            int64_t a1Offset = mOffset * tiling->nfftAlign;
            int64_t a2Offset = a1Offset;

            SplitWindows(inputOffset, realSplitWindowOffset, imagSplitWindowOffset, nFactor);

            if (i == 0) {
                GenerateGatherMask();
            }
            StftMatmul(realSplitWindowOffset, imagSplitWindowOffset, a1Offset, a2Offset, realOffset, imagOffset);
            GatherRealAndImag(realOffset, imagOffset, outputOffset, mFactor, nFactor);
        }
    }

private:
    __aicore__ inline void GenerateGatherMask()
    {
        LocalTensor<int32_t> maskTemp = maskUB.template Get<int32_t>(maskCount);

        // 生成等差数列 0, 2, 4, 6, 8, 10, 12, 14,...
        ArithProgression<int32_t>(maskTemp, (int32_t)0, (int32_t)2, maskCount);

        // 奇数index清零：0, 0, 4, 0, 8, 0, 12, 0,...
        uint64_t maskBit1[2] = {0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA};
        // float32, mask_h should be zero
        if (sizeof(T) == FLOAT_SIZE) {
            maskBit1[1] = 0;
        }

        UnaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.srcBlkStride = 1;
        // 默认一个repeat跳8个block
        repeatParams.dstRepStride = BLOCK_FOR_ONE_REPEAT;
        // 默认一个repeat跳8个block
        repeatParams.srcRepStride = BLOCK_FOR_ONE_REPEAT;
        Muls(maskTemp, maskTemp, 0, maskBit1, maskCount * sizeof(int32_t) / REPEAT_SIZE, repeatParams);

        // 生成等差数列：-2， 0， 2， 4， 6， 8， 10，12，...
        LocalTensor<int32_t> temp = tempUB.template Get<int32_t>(maskCount);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        ArithProgression<int32_t>(temp, NEGATIVE_TWO, POSITIVE_TWO, maskCount);

        // 加上地址偏移offset = imagbase-realbase：offset-2, offset, offset+2, offset+4, ..., offset+12, ...
        int32_t offset = static_cast<int32_t>(reinterpret_cast<uintptr_t>(aImagPing.GetPhyAddr())) -
                         static_cast<int32_t>(reinterpret_cast<uintptr_t>(aRealPing.GetPhyAddr()));

        Adds(temp, temp, offset, maskCount);

        // 偶数index清零： 0， offset, 0, offset+4, 0, offset+8, 0, offset+12,...
        uint64_t maskBit2[2] = {0x5555555555555555, 0x5555555555555555};
        // float32, mask_h should be zero
        if (sizeof(T) == FLOAT_SIZE) {
            maskBit2[1] = 0;
        }
        Muls(temp, temp, 0, maskBit2, maskCount / REPEAT_NUM_FOR_FP32, repeatParams);
        // 相加: 0, offset, 4, offset+4, 8, offset+8, 12, offset+12,...
        Add(maskTemp, maskTemp, temp, maskCount);
        mask = maskTemp.ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void GatherForSmallNFactorAlign(
        int64_t realOffset, int64_t imagOffset, int64_t outputOffset, uint32_t mFactor, uint32_t nFactor)
    {
        int32_t complexCount = mFactor * nFactor * DOUBLE_BUFFER;
        int32_t ubCount = tiling->maskUBSize / sizeof(int32_t) / DOUBLE_BUFFER;
        int32_t gatherCountPerLoop = complexCount > maskCount ? maskCount : complexCount;

        gatherCountPerLoop = gatherCountPerLoop - gatherCountPerLoop % (nFactor * DOUBLE_BUFFER);
        int32_t realCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;
        int32_t imagCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        int ping = 1;
        int repeats = (complexCount + gatherCountPerLoop - 1) / gatherCountPerLoop;

        for (int i = 0; i < repeats; i++) {
            event_t event_id = ping ? EVENT_ID0 : EVENT_ID1;
            auto complexUB = ping ? complexPing : complexPong;
            auto aRealUB = ping ? aRealPing : aRealPong;
            auto aImagUB = ping ? aImagPing : aImagPong;
            auto bRealUB = ping ? bRealPing : bRealPong;
            auto bImagUB = ping ? bImagPing : bImagPong;

            int32_t copyLen = realCountPerLoop * sizeof(T);

            if (i == repeats - 1) {
                copyLen = (mFactor * nFactor - realCountPerLoop * i) * sizeof(T);
            }

            int32_t nBlocks = (copyLen + BLOCK_SIZE - 1) / BLOCK_SIZE;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

            DataCopy(aRealUB, aRealGm[realOffset + i * realCountPerLoop], {1, static_cast<uint16_t>(nBlocks), 0, 0});

            DataCopy(aImagUB, aImagGm[imagOffset + i * imagCountPerLoop], {1, static_cast<uint16_t>(nBlocks), 0, 0});

            DataCopy(bRealUB, bRealGm[realOffset + i * realCountPerLoop], {1, static_cast<uint16_t>(nBlocks), 0, 0});

            DataCopy(bImagUB, bImagGm[imagOffset + i * imagCountPerLoop], {1, static_cast<uint16_t>(nBlocks), 0, 0});

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);

            aRealUB = aRealUB - bImagUB;
            aImagUB = aImagUB + bRealUB;
            PipeBarrier<PIPE_V>();

            Gather(complexUB, aRealUB, mask, 0, DOUBLE_BUFFER * copyLen / sizeof(T));
            if (tiling->normalized) {
                Muls(complexUB, complexUB, tiling->rootNfft, DOUBLE_BUFFER * copyLen / sizeof(T));
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);

            if (nFactor <= 0) {
                nFactor = 1;
            }
            int32_t loops = copyLen / sizeof(T) / nFactor;

            DataCopyExtParams copyParams{
                static_cast<uint16_t>(loops), static_cast<uint32_t>(nFactor * DOUBLE_BUFFER * sizeof(T)), 0,
                static_cast<uint32_t>(DOUBLE_BUFFER * (tiling->matmulN - nFactor) * sizeof(T)), 0};
            DataCopyPad(outputGm[outputOffset], complexUB, copyParams);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
            outputOffset += DOUBLE_BUFFER * loops * tiling->matmulN;
            ping = 1 - ping;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void GatherForSmallNFactorNonAlign(
        int64_t realOffset, int64_t imagOffset, int64_t outputOffset, uint32_t mFactor, uint32_t nFactor)
    {
        int32_t nFactorAlign = (nFactor * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / sizeof(T);
        int32_t complexCount = mFactor * nFactorAlign * DOUBLE_BUFFER;
        int32_t gatherCountPerLoop = complexCount > maskCount ? maskCount : complexCount;
        gatherCountPerLoop = gatherCountPerLoop - gatherCountPerLoop % (nFactorAlign * DOUBLE_BUFFER);
        int32_t realCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;
        int32_t imagCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        int ping = 1;
        int repeats = (complexCount + gatherCountPerLoop - 1) / gatherCountPerLoop;

        for (int i = 0; i < repeats; i++) {
            event_t event_id = ping ? EVENT_ID0 : EVENT_ID1;
            auto complexUB = ping ? complexPing : complexPong;
            auto aRealUB = ping ? aRealPing : aRealPong;
            auto aImagUB = ping ? aImagPing : aImagPong;
            auto bRealUB = ping ? bRealPing : bRealPong;
            auto bImagUB = ping ? bImagPing : bImagPong;

            int32_t copyLen = realCountPerLoop * sizeof(T);
            if (i == repeats - 1) {
                copyLen = (mFactor * nFactorAlign - realCountPerLoop * i) * sizeof(T);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
            int32_t loops = copyLen / sizeof(T) / nFactorAlign;

            DataCopyExtParams copyParams{
                static_cast<uint16_t>(loops), static_cast<uint32_t>(nFactor * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(aRealUB, aRealGm[realOffset], copyParams, padParams);
            DataCopyPad(aImagUB, aImagGm[imagOffset], copyParams, padParams);
            DataCopyPad(bRealUB, bRealGm[realOffset], copyParams, padParams);
            DataCopyPad(bImagUB, bImagGm[imagOffset], copyParams, padParams);

            realOffset += loops * nFactor;
            imagOffset += loops * nFactor;

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);

            aRealUB = aRealUB - bImagUB;
            aImagUB = aImagUB + bRealUB;
            PipeBarrier<PIPE_V>();

            int32_t count = DOUBLE_BUFFER * nFactorAlign * loops;

            Gather(complexUB, aRealUB, mask, 0, count);

            if (tiling->normalized) {
                Muls(complexUB, complexUB, tiling->rootNfft, count);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);

            uint32_t srcGap = (nFactorAlign - nFactor) * DOUBLE_BUFFER * sizeof(T) > BLOCK_SIZE ? 1 : 0;

            DataCopyPad(
                outputGm[outputOffset], complexUB,
                {static_cast<uint16_t>(loops), static_cast<uint32_t>(nFactor * DOUBLE_BUFFER * sizeof(T)),
                 static_cast<uint32_t>(srcGap),
                 static_cast<uint32_t>(DOUBLE_BUFFER * (tiling->matmulN - nFactor) * sizeof(T)), 0});

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

            outputOffset += DOUBLE_BUFFER * loops * tiling->matmulN;
            ping = 1 - ping;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void GatherForLargeNFactorAlign(
        int64_t realOffset, int64_t imagOffset, int64_t outputOffset, uint32_t mFactor, uint32_t nFactor)
    {
        int32_t gatherCountPerLoop = tiling->maskUBSize / sizeof(int32_t);
        int32_t realCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;
        int32_t imagCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        int ping = 1;
        for (int m = 0; m < mFactor; m++) {
            int repeats = (nFactor + realCountPerLoop - 1) / realCountPerLoop;
            for (int i = 0; i < repeats; i++) {
                event_t event_id = ping ? EVENT_ID0 : EVENT_ID1;
                auto complexUB = ping ? complexPing : complexPong;
                auto aRealUB = ping ? aRealPing : aRealPong;
                auto aImagUB = ping ? aImagPing : aImagPong;
                auto bRealUB = ping ? bRealPing : bRealPong;
                auto bImagUB = ping ? bImagPing : bImagPong;

                int32_t copyLen = realCountPerLoop * sizeof(T);
                if (i == repeats - 1) {
                    copyLen = (mFactor * nFactor - realCountPerLoop * i) * sizeof(T);
                }

                int32_t nBlocks = (copyLen + BLOCK_SIZE - 1) / BLOCK_SIZE;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                DataCopy(
                    aRealUB, aRealGm[realOffset + i * realCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});
                DataCopy(
                    aImagUB, aImagGm[imagOffset + i * imagCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});
                DataCopy(
                    bRealUB, bRealGm[realOffset + i * realCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});
                DataCopy(
                    bImagUB, bImagGm[imagOffset + i * imagCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);

                aRealUB = aRealUB - bImagUB;
                aImagUB = aImagUB + bRealUB;
                PipeBarrier<PIPE_V>();

                Gather(complexUB, aRealUB, mask, 0, DOUBLE_BUFFER * copyLen / sizeof(T));

                if (tiling->normalized) {
                    Muls(complexUB, complexUB, tiling->rootNfft, DOUBLE_BUFFER * copyLen / sizeof(T));
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);

                if (nFactor <= 0) {
                    nFactor = 1;
                }
                int32_t loops = copyLen / sizeof(T) / nFactor;

                DataCopyPad(
                    outputGm[outputOffset + DOUBLE_BUFFER * (i * copyLen) / sizeof(T) + DOUBLE_BUFFER * m * nFactor],
                    complexUB, {1, static_cast<uint32_t>(copyLen * DOUBLE_BUFFER), 0, 0, 0});

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                ping = 1 - ping;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void GatherForLargeNFactorNonAlign(
        int64_t realOffset, int64_t imagOffset, int64_t outputOffset, uint32_t mFactor, uint32_t nFactor)
    {
        int32_t gatherCountPerLoop = tiling->maskUBSize / sizeof(int32_t);
        int32_t realCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;
        int32_t imagCountPerLoop = gatherCountPerLoop / DOUBLE_BUFFER;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        int ping = 1;
        for (int m = 0; m < mFactor; m++) {
            int repeats = (nFactor + realCountPerLoop - 1) / realCountPerLoop;
            for (int i = 0; i < repeats; i++) {
                event_t event_id = ping ? EVENT_ID0 : EVENT_ID1;
                auto complexUB = ping ? complexPing : complexPong;
                auto aRealUB = ping ? aRealPing : aRealPong;
                auto aImagUB = ping ? aImagPing : aImagPong;
                auto bRealUB = ping ? bRealPing : bRealPong;
                auto bImagUB = ping ? bImagPing : bImagPong;

                int32_t copyLen = realCountPerLoop * sizeof(T);
                if (i == repeats - 1) {
                    copyLen = (mFactor * nFactor - realCountPerLoop * i) * sizeof(T);
                }

                int32_t nBlocks = (copyLen + BLOCK_SIZE - 1) / BLOCK_SIZE;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                DataCopy(
                    aRealUB, aRealGm[realOffset + i * realCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});
                DataCopy(
                    aImagUB, aImagGm[imagOffset + i * imagCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});
                DataCopy(
                    bRealUB, bRealGm[realOffset + i * realCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});
                DataCopy(
                    bImagUB, bImagGm[imagOffset + i * imagCountPerLoop + m * nFactor],
                    {1, static_cast<uint16_t>(nBlocks), 0, 0});

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);

                aRealUB = aRealUB - bImagUB;
                aImagUB = aImagUB + bRealUB;
                PipeBarrier<PIPE_V>();

                Gather(complexUB, aRealUB, mask, 0, DOUBLE_BUFFER * copyLen / sizeof(T));

                if (tiling->normalized) {
                    Muls(complexUB, complexUB, tiling->rootNfft, DOUBLE_BUFFER * copyLen / sizeof(T));
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);

                if (nFactor <= 0) {
                    nFactor = 1;
                }
                int32_t loops = copyLen / sizeof(T) / nFactor;

                DataCopyPad(
                    outputGm[outputOffset + DOUBLE_BUFFER * (i * copyLen) / sizeof(T) + DOUBLE_BUFFER * m * nFactor],
                    complexUB, {1, static_cast<uint32_t>(copyLen * DOUBLE_BUFFER), 0, 0, 0});

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                ping = 1 - ping;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void GatherRealAndImag(
        int64_t realOffset, int64_t imagOffset, int64_t outputOffset, uint32_t mFactor, uint32_t nFactor)
    {
        int32_t ubCount = tiling->maskUBSize / sizeof(int32_t);
        if (nFactor * sizeof(T) % BLOCK_SIZE == 0) {
            if (ubCount >= nFactor * DOUBLE_BUFFER) {
                GatherForSmallNFactorAlign(realOffset, imagOffset, outputOffset, mFactor, nFactor);
            } else {
                GatherForLargeNFactorAlign(realOffset, imagOffset, outputOffset, mFactor, nFactor);
            }
        } else {
            if (ubCount >= nFactor * DOUBLE_BUFFER) {
                GatherForSmallNFactorNonAlign(realOffset, imagOffset, outputOffset, mFactor, nFactor);
            } else {
                GatherForLargeNFactorNonAlign(realOffset, imagOffset, outputOffset, mFactor, nFactor);
            }
        }
    }

    __aicore__ inline void StftMatmul(
        int64_t realSplitWindowOffset, int64_t imagSplitWindowOffset, int64_t a1Offset, int64_t a2Offset,
        int64_t realOffset, int64_t imagOffset)
    {
        // AC
        mm.SetTensorA(a1Global[a1Offset]);
        mm.SetTensorB(splitRealWindowGm[realSplitWindowOffset], true);
        mm.IterateAll(aRealGm[realOffset]);

        // AD
        mm.SetTensorA(a2Global[a2Offset]);
        mm.SetTensorB(splitRealWindowGm[realSplitWindowOffset], true);
        mm.IterateAll(bRealGm[realOffset]);

        // BC
        mm.SetTensorA(a1Global[a1Offset]);
        mm.SetTensorB(splitImagWindowGm[imagSplitWindowOffset], true);
        mm.IterateAll(aImagGm[imagOffset]);

        // BD
        mm.SetTensorA(a2Global[a2Offset]);
        mm.SetTensorB(splitImagWindowGm[imagSplitWindowOffset], true);
        mm.IterateAll(bImagGm[imagOffset]);
    }

    __aicore__ inline void SplitWindows(
        int64_t inputOffset, int64_t realSplitWindowOffset, int64_t imagSplitWindowOffset, int64_t nFactor)
    {
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams1;
        intriParams1.blockCount = 1;
        intriParams1.blockLen = tiling->nFactorUbFormer * COMPLEX_COEFFICIENT * sizeof(T);
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyParams intriParams2;
        intriParams2.blockCount = 1;

        intriParams2.blockLen = tiling->nFactorUbFormer * sizeof(T);
        intriParams2.srcStride = 0;
        intriParams2.dstStride = 0;
        DataCopyParams intriParams3;
        intriParams3.blockCount = 1;
        intriParams3.blockLen = tiling->nFactorUbTail * COMPLEX_COEFFICIENT * sizeof(T);
        intriParams3.srcStride = 0;
        intriParams3.dstStride = 0;
        DataCopyParams intriParams4;
        intriParams4.blockCount = 1;

        intriParams4.blockLen = tiling->nFactorUbTail * sizeof(T);
        intriParams4.srcStride = 0;
        intriParams4.dstStride = 0;

        for (int32_t i = 0; i < nFactor; i++) {
            for (int32_t j = 0; j < tiling->nFactorUbLoop - 1; j++) {
                LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
                DataCopyPad(
                    inputLocal,
                    inputGm
                        [inputOffset + i * tiling->hopLength * COMPLEX_COEFFICIENT +
                         j * tiling->nFactorUbFormer * COMPLEX_COEFFICIENT],
                    intriParams1, padParams);
                inCopy.EnQue(inputLocal);
                SplitRealAndImag(tiling->nFactorUbFormer);
                LocalTensor<T> realOutputLocal = realOutCopy.template DeQue<T>();
                LocalTensor<T> imagOutputLocal = imagOutCopy.template DeQue<T>();
                DataCopyPad(
                    splitRealWindowGm[realSplitWindowOffset + i * tiling->nfftAlign + j * tiling->nFactorUbFormer],
                    realOutputLocal, intriParams2);
                DataCopyPad(
                    splitImagWindowGm[imagSplitWindowOffset + i * tiling->nfftAlign + j * tiling->nFactorUbFormer],
                    imagOutputLocal, intriParams2);
                realOutCopy.FreeTensor(realOutputLocal);
                imagOutCopy.FreeTensor(imagOutputLocal);
            }
            LocalTensor<T> inputLocal = inCopy.template AllocTensor<T>();
            DataCopyPad(
                inputLocal,
                inputGm
                    [inputOffset + i * tiling->hopLength * COMPLEX_COEFFICIENT +
                     (tiling->nFactorUbLoop - 1) * tiling->nFactorUbFormer * COMPLEX_COEFFICIENT],
                intriParams3, padParams);
            inCopy.EnQue(inputLocal);
            SplitRealAndImag(tiling->nFactorUbTail);
            LocalTensor<T> realOutputLocal = realOutCopy.template DeQue<T>();
            LocalTensor<T> imagOutputLocal = imagOutCopy.template DeQue<T>();

            DataCopyPad(
                splitRealWindowGm
                    [realSplitWindowOffset + i * tiling->nfftAlign +
                     (tiling->nFactorUbLoop - 1) * tiling->nFactorUbFormer],
                realOutputLocal, intriParams4);
            DataCopyPad(
                splitImagWindowGm
                    [imagSplitWindowOffset + i * tiling->nfftAlign +
                     (tiling->nFactorUbLoop - 1) * tiling->nFactorUbFormer],
                imagOutputLocal, intriParams4);
            realOutCopy.FreeTensor(realOutputLocal);
            imagOutCopy.FreeTensor(imagOutputLocal);
        }
    }

    __aicore__ inline void SplitRealAndImag(int64_t colNum)
    {
        LocalTensor<T> inputLocal = inCopy.template DeQue<T>();
        LocalTensor<T> realOutputLocal = realOutCopy.template AllocTensor<T>();
        LocalTensor<T> imagOutputLocal = imagOutCopy.template AllocTensor<T>();

        uint64_t rsvdCnt = 0;
        uint16_t repeatTimes = (colNum * DOUBLE_BUFFER + REPEAT_NUM_FOR_FP32 - 1) / REPEAT_NUM_FOR_FP32;

        GatherMask(
            realOutputLocal, inputLocal, 1, false, 0, {1, repeatTimes, BLOCK_FOR_ONE_REPEAT, BLOCK_FOR_ONE_REPEAT},
            rsvdCnt);
        GatherMask(
            imagOutputLocal, inputLocal, DOUBLE_BUFFER, false, 0,
            {1, repeatTimes, BLOCK_FOR_ONE_REPEAT, BLOCK_FOR_ONE_REPEAT}, rsvdCnt);
        realOutCopy.EnQue(realOutputLocal);
        imagOutCopy.EnQue(imagOutputLocal);
        inCopy.FreeTensor(inputLocal);
    }

    uint32_t BLOCK_SIZE = 32;
    uint32_t WORKSPACE_ALIGN_SIZE = 512;
    uint32_t REPEAT_SIZE = 256;
    int32_t maskCount;

    STFTGeneralizedTilingData* tiling;

    TPipe* pipe;
    TQue<QuePosition::VECIN, bufferNum> inCopy;
    TQue<QuePosition::VECOUT, bufferNum> realOutCopy;
    TQue<QuePosition::VECOUT, bufferNum> imagOutCopy;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> splitRealWindowGm;
    GlobalTensor<T> splitImagWindowGm;
    GlobalTensor<T> aRealGm;  // real part of matmul workspace
    GlobalTensor<T> aImagGm;  // imag part of matmul workspace
    GlobalTensor<T> bRealGm;  // real part of matmul workspace
    GlobalTensor<T> bImagGm;  // imag part of matmul workspace
    GlobalTensor<T> a1Global; // real part of dft matrix
    GlobalTensor<T> a2Global; // imag part of dft matrix

    TBuf<> aRealPingUB;   // real part in ub
    TBuf<> aRealPongUB;   // real part in ub
    TBuf<> aImagPingUB;   // imag part in ub
    TBuf<> aImagPongUB;   // imag part in ub
    TBuf<> bRealPingUB;   // real part in ub
    TBuf<> bRealPongUB;   // real part in ub
    TBuf<> bImagPingUB;   // imag part in ub
    TBuf<> bImagPongUB;   // imag part in ub
    TBuf<> complexPingUB; // complex in ub
    TBuf<> complexPongUB; // complex in ub
    TBuf<> maskUB;        // gather mask in ub
    TBuf<> tempUB;        // temp buffer in ub

    LocalTensor<uint32_t> mask;
    LocalTensor<T> aRealPing;
    LocalTensor<T> aRealPong;
    LocalTensor<T> aImagPing;
    LocalTensor<T> aImagPong;
    LocalTensor<T> bRealPing;
    LocalTensor<T> bRealPong;
    LocalTensor<T> bImagPing;
    LocalTensor<T> bImagPong;
    LocalTensor<T> complexPing;
    LocalTensor<T> complexPong;
};
} // namespace STFTND

#endif