/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DROP_OUT_V3_IMPL_H
#define DROP_OUT_V3_IMPL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace DropOutV3 {
using namespace AscendC;

template <typename T, typename U>
class DropOutV3Impl {
public:
    __aicore__ inline DropOutV3Impl(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR noiseShape, GM_ADDR p, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR mask,
        GM_ADDR workspace, const DropOutV3TilingData *tilingData, TPipe *pipeIn);
    __aicore__ inline void Process(const DropOutV3TilingData *tilingData);

private:
    __aicore__ inline void CopyIn(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void GenDropMask(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void DoDropMask(LocalTensor<T> ubOutput, LocalTensor<T> ubInput, uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void GenPhiloxRandom(uint32_t dataCount);
    __aicore__ inline void Uint32ToFloat(uint32_t dataCount);
    __aicore__ inline void CompareMask(uint32_t dataCount);
    __aicore__ inline void Skip(uint64_t count);
    __aicore__ inline uint64_t AlignProcessByte256(uint64_t param);
    __aicore__ inline uint64_t PadProcessInt32(uint64_t param);

private:
    TPipe *pipe_;
    constexpr static int64_t bufferNum_ = 2;
    constexpr static uint32_t ALG_KEY_SIZE = 2;
    constexpr static uint32_t ALG_COUNTER_SIZE = 4;
    constexpr static uint64_t RESULT_ELEMENT_CNT = 4;
    constexpr static float CURAND_2POW32_INV = 2.3283064365386963e-10;
    constexpr static float COEFF_2POW32_INV = 2.0f;
    constexpr static uint32_t gainCoeff = 2;

    constexpr static uint32_t uint32Uint8Ratio = 4;
    constexpr static uint32_t dropMaskUint32AlignSize = 4;
    constexpr static uint32_t byteBitRatio = 8;
    constexpr static uint8_t uint8Max = 255;
    constexpr static uint32_t ONCE_ALGN_NUM_INT32 = 8;
    constexpr static uint32_t ONCE_ALGN_NUM_BYTE_256 = 256;
    constexpr static uint32_t COUNTER_IDX2 = 2;
    constexpr static uint32_t COUNTER_IDX3 = 3;
    constexpr static uint32_t RIGHT_SHIFT_BIT = 32;

    TQue<QuePosition::VECIN, bufferNum_> xInputQueue_;
    TQue<QuePosition::VECOUT, bufferNum_> yOutputQueue_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TBuf<QuePosition::VECCALC> shareCommonBuf_;
    TBuf<QuePosition::VECCALC> floatDataBuf_;

    GlobalTensor<T> xInputGm_;
    GlobalTensor<U> probInputGm_;
    GlobalTensor<T> yOutputGm_;
    GlobalTensor<uint8_t> maskOutputGm_;

    uint64_t curNumOfCore_ = 0;
    uint64_t curPerOfCore_ = 0;
    uint64_t curTailOfCore_ = 0;
    uint64_t loopCount_ = 0;

    uint32_t countIndex0_ = 0;
    uint32_t countIndex1_ = 1;
    uint32_t countIndex2_ = 2;
    uint32_t countIndex3_ = 3;

    float prob_ = 0.0f;
    bool isPadding_ = false;
    uint64_t rightPadding_ = 0;

    uint64_t blockOffset_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN,
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT
    };
};

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Init(
    GM_ADDR x, GM_ADDR noiseShape, GM_ADDR p, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR mask, 
    GM_ADDR workspace, const DropOutV3TilingData *tilingData, TPipe *pipeIn)
{
    // Init tiling data
    for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tilingData->key[i];
    }
    for (uint32_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tilingData->counter[i];
    }

    if (GetBlockIdx() + 1 == tilingData->usedCoreNum) {
        curNumOfCore_ = tilingData->numOfTailCore;
        curPerOfCore_ = tilingData->perOfTailCore;
        curTailOfCore_ = tilingData->tailOfTailCore;
        loopCount_ = tilingData->loopOfTailCore;
    } else {
        curNumOfCore_ = tilingData->numOfPerCore;
        curPerOfCore_ = tilingData->perOfPerCore;
        curTailOfCore_ = tilingData->tailOfPerCore;
        loopCount_ = tilingData->loopOfPerCore;
    }

    // SetBuffer
    blockOffset_ = GetBlockIdx() * tilingData->numOfPerCore;
    xInputGm_.SetGlobalBuffer((__gm__ T*)x);
    probInputGm_.SetGlobalBuffer((__gm__ U*)p);
    yOutputGm_.SetGlobalBuffer((__gm__ T*)y);
    maskOutputGm_.SetGlobalBuffer((__gm__ uint8_t*)mask);

    // InitBuffer
    pipe_ = pipeIn;
    pipe_->InitBuffer(xInputQueue_, bufferNum_, AlignProcessByte256(curPerOfCore_ * sizeof(T)));
    pipe_->InitBuffer(yOutputQueue_, bufferNum_, AlignProcessByte256(curPerOfCore_ * sizeof(float)));
    pipe_->InitBuffer(philoxQueBuf_, AlignProcessByte256(curPerOfCore_ * sizeof(uint32_t)));
    pipe_->InitBuffer(shareCommonBuf_, tilingData->shareTmpUbSize * sizeof(uint8_t));
    pipe_->InitBuffer(floatDataBuf_, AlignProcessByte256(curPerOfCore_ * sizeof(float)));

    // GetValue
    U currProb = probInputGm_.GetValue(0);
    if constexpr (AscendC::IsSameType<U, half>::value) {
        prob_ = static_cast<float>(currProb);
    } else if constexpr (AscendC::IsSameType<U, float>::value) {
        prob_ = static_cast<U>(currProb);
    } else if constexpr (AscendC::IsSameType<U, bfloat16_t>::value) {
        prob_ = AscendC::ToFloat(currProb);
    }
}

template <typename T, typename U>
__aicore__ inline uint64_t DropOutV3Impl<T, U>::AlignProcessByte256(uint64_t param)
{
    return (param + ONCE_ALGN_NUM_BYTE_256 - 1) / ONCE_ALGN_NUM_BYTE_256 * ONCE_ALGN_NUM_BYTE_256;
};

template <typename T, typename U>
__aicore__ inline uint64_t DropOutV3Impl<T, U>::PadProcessInt32(uint64_t param)
{
    return (ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32);
};

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Skip(uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> RIGHT_SHIFT_BIT);

    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[COUNTER_IDX2] == 0) {
            ++counter_[COUNTER_IDX3];
        }
    }
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::CopyIn(uint32_t loopIdx, uint32_t dataCount)
{
    LocalTensor<T> xInputUb = xInputQueue_.AllocTensor<T>();
    if (dataCount * sizeof(T) % 32) {
        isPadding_ = true;
        rightPadding_ = PadProcessInt32(dataCount);
    }

    DataCopyExtParams copyParams {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(dataCount * sizeof(T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPadExtParams<T> padParams {
        isPadding_,
        static_cast<uint8_t>(0),
        static_cast<uint8_t>(rightPadding_),
        static_cast<T>(0)
    };

    DataCopyPad(xInputUb, xInputGm_[blockOffset_ + loopIdx * curPerOfCore_], copyParams, padParams);
    xInputQueue_.EnQue(xInputUb);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::GenPhiloxRandom(uint32_t dataCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    AscendC::PhiloxRandom<10>(philoxRes, { key_[0], key_[1] }, 
    { counter_[countIndex0_], counter_[countIndex1_], counter_[countIndex2_], counter_[countIndex3_] }, dataCount);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Uint32ToFloat(uint32_t dataCount)
{
    // philox result saved in philoxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int64_t *ubPhilox = (__ubuf__ int64_t *)philoxRes.GetPhyAddr();

    LocalTensor<float> caluData = floatDataBuf_.Get<float>();
    uint32_t countAlign256 = AlignProcessByte256(dataCount);
    AscendC::Duplicate(caluData, static_cast<float>(1.0), countAlign256);
    __ubuf__ float *ubOut = (__ubuf__ float *)caluData.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(int64_t);
    uint32_t repeatTimes = ops::CeilDiv(dataCount, vfLen);

    uint32_t sReg1 = static_cast<uint32_t>(dataCount) * gainCoeff;
    float sReg3 = static_cast<float>(CURAND_2POW32_INV);
    float sReg4 = static_cast<float>(CURAND_2POW32_INV / COEFF_2POW32_INV);
    int32_t offset = static_cast<int32_t>(vfLen);
    uint32_t offsetCoeff = offset / gainCoeff;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int64_t> vReg0;
        MicroAPI::RegTensor<float> vReg1;
        MicroAPI::RegTensor<float> vReg2;
        MicroAPI::RegTensor<float> vReg3;
        MicroAPI::MaskReg mask;

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<int32_t>(sReg1);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B32>(
                vReg0, ubPhilox, offsetCoeff);
            MicroAPI::Cast<float, int64_t, castTrait>(vReg1, vReg0, mask);
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg1, sReg3, mask);
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(vReg3, vReg2, sReg4, mask);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B64>(
                ubOut, vReg3, offset, mask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::CompareMask(uint32_t dataCount)
{
    LocalTensor<float> caluData = floatDataBuf_.Get<float>();
    LocalTensor<uint8_t> maskOutputUb = philoxQueBuf_.Get<uint8_t>();

    uint32_t countAlign256 = AlignProcessByte256(dataCount);
    AscendC::CompareScalar(maskOutputUb, caluData, static_cast<float>(prob_), CMPMODE::LE, countAlign256);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::GenDropMask(uint32_t loopIdx, uint32_t dataCount)
{
    GenPhiloxRandom(dataCount);
    Uint32ToFloat(dataCount);
    CompareMask(dataCount);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::DoDropMask(LocalTensor<T> yOutputUb, LocalTensor<T> xInputUb, uint32_t loopIdx, uint32_t dataCount)
{
    // 申请共享buffer空间 && 获取DropOutV3Mask的元素个数
    LocalTensor<uint8_t> apiTmpBuffer = shareCommonBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskOutputUb = philoxQueBuf_.Get<uint8_t>();

    DropOutShapeInfo DropOutShapeInfo;
    DropOutShapeInfo.firstAxis = 1;
    DropOutShapeInfo.srcLastAxis = dataCount;
    DropOutShapeInfo.maskLastAxis = AlignProcessByte256(dataCount) / byteBitRatio;

    AscendC::DropOut<T, false, 4>(yOutputUb, xInputUb, maskOutputUb, apiTmpBuffer, prob_, DropOutShapeInfo);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Compute(uint32_t loopIdx, uint32_t dataCount)
{
    LocalTensor<T> xInputUb = xInputQueue_.DeQue<T>();
    LocalTensor<T> yOutputUb = yOutputQueue_.AllocTensor<T>();

    GenDropMask(loopIdx, dataCount);
    DoDropMask(yOutputUb, xInputUb, loopIdx, dataCount);

    yOutputQueue_.EnQue<T>(yOutputUb);
    xInputQueue_.FreeTensor(xInputUb);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::CopyOut(uint32_t loopIdx, uint32_t dataCount)
{
    LocalTensor<T> yOutputUb = yOutputQueue_.DeQue<T>();
    DataCopyExtParams copyParamsYOut {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(dataCount * sizeof(T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPad(yOutputGm_[blockOffset_ + loopIdx * curPerOfCore_], yOutputUb, copyParamsYOut);
    yOutputQueue_.FreeTensor(yOutputUb);

    event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
    LocalTensor<uint8_t> maskOutputUb = philoxQueBuf_.Get<uint8_t>();
    DataCopyExtParams copyParamsMaskOut {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(ops::CeilDiv(dataCount, byteBitRatio) * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPad(maskOutputGm_[(blockOffset_ + loopIdx * curPerOfCore_) / byteBitRatio], maskOutputUb, copyParamsMaskOut);
    event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMTE3V);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Process(const DropOutV3TilingData *tilingData)
{
    if (GetBlockIdx() >= tilingData->usedCoreNum) {
        return;
    }

    auto groupCnt = ops::CeilDiv(blockOffset_, RESULT_ELEMENT_CNT);
    Skip(groupCnt);

    auto numOfCurLoop = curPerOfCore_;
    for (int64_t n = 0; n < loopCount_; n++) {
        if (n == (loopCount_ - 1) && (curNumOfCore_ % curPerOfCore_ != 0)) {
            numOfCurLoop = curTailOfCore_;
        }
        CopyIn(n, numOfCurLoop);
        Compute(n, numOfCurLoop);
        CopyOut(n, numOfCurLoop);

        groupCnt = ops::CeilDiv(numOfCurLoop, RESULT_ELEMENT_CNT);
        Skip(groupCnt);
    }
}

}  // namespace DropOutV3

#endif  // DROP_OUT_V3_IMPL_H