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
 * \file pad_cut_last.h
 * \brief pad cut last dim kernel
 */

#ifndef PAD_EDGE_HUGE_WIDTH_H_
#define PAD_EDGE_HUGE_WIDTH_H_

#include "kernel_operator.h"
#include "pad_common.h"
#include "pad_v3_struct.h"

namespace PadV3 {
using namespace AscendC;
struct PadEdgeHugeParam {
    uint32_t padWLOffset;
    uint32_t padWROffset;
};

template <typename T>
class KernelPadEdgeWithHugeWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);

    GlobalTensor<T> input_;
    GlobalTensor<T> output_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
    TPipe* pipe_ = nullptr;

    uint32_t blockIdx_;
    uint32_t additionOffsetHuge_;

    const PadACTilingData* tilingData_ = nullptr;

    uint32_t inCopyLen_;
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    int8_t ubAxis_;
    uint64_t ubFactor_;

    uint16_t needPadRight_;
    uint32_t padRightLen_;
    static constexpr uint32_t additionLen_ = VL_SIZE / sizeof(T);
    T padRightValue_;
    uint32_t padRightSrcOffset_;
    uint16_t repeatTimes_;

    uint64_t factorOfubAxis_;

public:
    __aicore__ inline KernelPadEdgeWithHugeWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();

        additionOffsetHuge_ = tilingData_->additionTileSize / sizeof(T);

        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM, tilingData_->outTileSize + tilingData_->additionTileSize);

        ubAxis_ = tilingData_->ubAxis;
        ubFactor_ = tilingData_->ubFactor;
    }

    __aicore__ inline void Process()
    {
        const uint32_t ubPerCount = tilingData_->ubPerCount;
        const uint32_t ubTotalCount = tilingData_->ubTotalCount;
        const uint8_t ubAxis = ubAxis_;
        const uint64_t ubFactor = ubFactor_;
        const uint64_t* outShape = tilingData_->outShape;
        const int64_t* leftPad = tilingData_->leftPad;
        const uint64_t* inShape = tilingData_->inShape;

        uint32_t startIdxEdgeHuge = blockIdx_ * ubPerCount;
        if (startIdxEdgeHuge >= ubTotalCount) {
            return;
        }
        uint32_t endIdxEdgeHuge = (blockIdx_ + 1L) * ubPerCount;
        endIdxEdgeHuge = (endIdxEdgeHuge < ubTotalCount ? endIdxEdgeHuge : ubTotalCount);

        factorOfubAxis_ = CeilDiv(outShape[ubAxis], ubFactor);
        const uint64_t leftPadUb = leftPad[ubAxis];
        const uint64_t inShapeUb = inShape[ubAxis];
        const uint64_t leftPlusInUb = leftPadUb + inShapeUb;

        for (uint32_t idx = startIdxEdgeHuge; idx < endIdxEdgeHuge; idx++) {
            uint32_t curIdx = idx;

            needPadRight_ = 0;
            padRightLen_ = 0;

            for (int32_t i = ubAxis; i >= 0; i--) {
                uint64_t factor = (i == ubAxis) ? factorOfubAxis_ : outShape[i];
                uint64_t outIdxI = (i == ubAxis) ? (curIdx % factor * ubFactor) : (curIdx % factor);
                outIndex_[i] = outIdxI;
                int64_t lp = leftPad[i];
                inIndex_[i] = (outIdxI < static_cast<uint64_t>(lp)) ?
                                  0 :
                                  min(outIdxI - static_cast<uint64_t>(lp), inShape[i] - 1);
                curIdx = curIdx / factor;
            }

            const uint64_t outUb = outIndex_[ubAxis];
            if (outUb < leftPlusInUb) {
                if (outUb + ubFactor <= leftPadUb) {
                    inCopyLen_ = 0;
                } else if (outUb + ubFactor < leftPlusInUb) {
                    inCopyLen_ = outUb + ubFactor - inIndex_[ubAxis] - leftPadUb;
                } else {
                    inCopyLen_ = inShapeUb - inIndex_[ubAxis];
                }
            } else {
                inCopyLen_ = 0;
            }

            ProcessOneStep(additionLen_);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(const uint32_t additionLen)
    {
        PadEdgeHugeParam padParam;

        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();

        CopyIn(srcLocal, padParam, additionLen);

        PadOneLine(srcLocal[additionOffsetHuge_], padParam, srcLocal, additionLen);

        CopyOut(srcLocal, padParam);

        inQueue_.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadEdgeHugeParam& padParam, const uint32_t additionLen)
    {
        const uint8_t dimNum = tilingData_->dimNum;
        const uint8_t ubAxis = ubAxis_;
        const uint64_t ubFactor = ubFactor_;
        const uint64_t* inStride = tilingData_->inStride;
        const int64_t* leftPad = tilingData_->leftPad;
        const uint64_t* inShape = tilingData_->inShape;
        const uint64_t* outShape = tilingData_->outShape;
        const uint64_t leftPadUb = leftPad[ubAxis];
        const uint64_t inShapeUb = inShape[ubAxis];
        const uint64_t outShapeUb = outShape[ubAxis];

        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum; i++) {
            inAddr += inIndex_[i] * inStride[i];
        }

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        if (inCopyLen_ == ubFactor) {
            padParam.padWLOffset = 0;
            padParam.padWROffset = ubFactor;
            copyInParams.blockLen = ubFactor * sizeof(T);

            DataCopyPad(src[additionOffsetHuge_], input_[inAddr], copyInParams, padParams);
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else if (inCopyLen_ == 0) {
            padParam.padWLOffset = 0;
            padParam.padWROffset = 0;

            SetEvent<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            T padValue = input_[inAddr].GetValue(0);

            Duplicate(src[additionOffsetHuge_], padValue, ubFactor);
        } else {
            if (outIndex_[ubAxis] < leftPadUb) {
                padParam.padWLOffset = leftPadUb % ubFactor;
            } else {
                padParam.padWLOffset = 0;
            }

            if (outIndex_[ubAxis] + ubFactor <= leftPadUb + inShapeUb) {
                padParam.padWROffset = ubFactor;
            } else {
                padParam.padWROffset = (leftPadUb + inShapeUb) % ubFactor;
            }

            uint32_t ubOffset = CeilAlign(padParam.padWLOffset, BLK_ELEMS);

            uint32_t copyLen = inCopyLen_;
            uint32_t remainderLen = 0;

            if (ubOffset != padParam.padWLOffset) {
                remainderLen = ubOffset - padParam.padWLOffset;
                if (inCopyLen_ <= remainderLen) {
                    copyLen = 0;
                    remainderLen = inCopyLen_;
                } else {
                    copyLen = inCopyLen_ - remainderLen;
                }
                copyInParams.blockLen = remainderLen * sizeof(T);
                DataCopyPad(src, input_[inAddr], copyInParams, padParams);
            }

            if (copyLen != 0) {
                copyInParams.blockLen = copyLen * sizeof(T);
                DataCopyPad(src[additionOffsetHuge_ + ubOffset], input_[inAddr + remainderLen], copyInParams,
                            padParams);
            }

            if (padParam.padWLOffset != 0) {
                if constexpr (sizeof(T) == B64_BYTES) {
                    SetEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    T padValue{0};
                    if (ubOffset != padParam.padWLOffset) {
                        padValue = src.GetValue(0);
                    } else {
                        padValue = src[additionOffsetHuge_].GetValue(ubOffset);
                    }
                    Duplicate(src[additionOffsetHuge_], padValue, padParam.padWLOffset);
                } else {
                    SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                    __ubuf__ T* leftSrcAddr = (ubOffset != padParam.padWLOffset) ?
                                                  (__ubuf__ T*)src.GetPhyAddr() :
                                                  (__ubuf__ T*)src[additionOffsetHuge_].GetPhyAddr() + ubOffset;
                    __ubuf__ T* leftDstAddr = (__ubuf__ T*)src[additionOffsetHuge_].GetPhyAddr();
                    uint16_t leftPadLoops = padParam.padWLOffset / additionLen;
                    uint32_t leftPadRemainder = padParam.padWLOffset % additionLen;
                    __VEC_SCOPE__
                    {
                        AscendC::Reg::RegTensor<T> vRegLeft;
                        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
                        if constexpr (sizeof(T) == 1) {
                            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B8>(vRegLeft, leftSrcAddr);
                        } else if constexpr (sizeof(T) == 2) {
                            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B16>(vRegLeft, leftSrcAddr);
                        } else {
                            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B32>(vRegLeft, leftSrcAddr);
                        }
                        for (uint16_t i = 0; i < leftPadLoops; i++) {
                            AscendC::Reg::StoreAlign(leftDstAddr + i * additionLen, vRegLeft, maskAll);
                        }
                        if (leftPadRemainder > 0) {
                            AscendC::Reg::MaskReg remMask = AscendC::Reg::UpdateMask<T>(leftPadRemainder);
                            AscendC::Reg::StoreAlign(leftDstAddr + leftPadLoops * additionLen, vRegLeft, remMask);
                        }
                    }
                }
            }
            if (padParam.padWROffset != ubFactor) {
                padRightLen_ = min(ubFactor, outShapeUb - outIndex_[ubAxis]) - padParam.padWROffset;
                if (padRightLen_ != 0) {
                    if constexpr (sizeof(T) == B64_BYTES) {
                        if (padParam.padWLOffset == 0) {
                            SetEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                        }
                        if (copyLen != 0) {
                            padRightValue_ = src[additionOffsetHuge_].GetValue(padParam.padWROffset - 1);
                        } else {
                            padRightValue_ = src.GetValue(inCopyLen_ - 1);
                        }
                    } else {
                        if (padParam.padWLOffset == 0) {
                            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                        }
                        padRightSrcOffset_ = (copyLen != 0) ? (additionOffsetHuge_ + padParam.padWROffset - 1) :
                                                              (inCopyLen_ - 1);
                    }

                    if (padRightLen_ <= additionLen) {
                        repeatTimes_ = 0;
                    } else {
                        repeatTimes_ = padRightLen_ / additionLen;
                        padRightLen_ -= repeatTimes_ * additionLen;
                    }
                    needPadRight_ = 1;
                }
            }
        }
    }

    __aicore__ inline void PadOneLine(LocalTensor<T> src, PadEdgeHugeParam& padParam, LocalTensor<T> addition,
                                      const uint32_t additionLen)
    {
        const bool padWLOffsetMode0 = padParam.padWLOffset % BLK_ELEMS == 0;
        if (padWLOffsetMode0 && needPadRight_ == 0) {
            SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            return;
        }
        if constexpr (sizeof(T) == B64_BYTES) {
            PadBothSide<AscendC::Reg::RegTraitNumTwo>(src, padParam, addition, additionLen, needPadRight_, padRightLen_,
                                                      padRightValue_, repeatTimes_, padWLOffsetMode0);
        } else {
            PadBothSide<AscendC::Reg::RegTraitNumOne>(src, padParam, addition, additionLen, needPadRight_, padRightLen_,
                                                      padRightValue_, repeatTimes_, padWLOffsetMode0);
        }
        SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadBothSide(LocalTensor<T> dst, PadEdgeHugeParam& padParam, LocalTensor<T> addition,
                                       const uint32_t additionLen, uint16_t npr, uint32_t prl, T prv, uint16_t rt,
                                       const bool padWLOffsetMode0)
    {
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* additionAddr = (__ubuf__ T*)addition.GetPhyAddr();

        const uint32_t padWLOffset = padParam.padWLOffset;
        const uint16_t needPadLeft = (padWLOffset > 0 && !padWLOffsetMode0);

        const uint32_t padLeftCeilAlign = CeilAlign(padWLOffset, BLK_ELEMS);
        const uint32_t padLeftEdge = padParam.padWROffset > padLeftCeilAlign ? padLeftCeilAlign : padParam.padWROffset;
        const uint32_t padLeftSize = padLeftEdge - padWLOffset;

        T padRightValue = prv;
        const uint16_t needPadRight = npr;
        const uint16_t needPadRightSurplus = (prl != 0);
        const uint32_t padRightLen = prl;
        const uint16_t repeatTimes = rt;
        const uint32_t padWROffset = padParam.padWROffset;
        const uint32_t padWROffsetOnRepeat = padWROffset + repeatTimes * additionLen;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T, Trait> vReg;
            AscendC::Reg::UnalignRegForStore uReg;

            for (uint16_t k = 0; k < needPadLeft; k++) {
                __ubuf__ T* outAddr = dstAddr + padWLOffset;
                AscendC::Reg::LoadAlign(vReg, additionAddr);
                AscendC::Reg::StoreUnAlign(outAddr, vReg, uReg, padLeftSize);
                AscendC::Reg::StoreUnAlignPost(outAddr, uReg, 0);
            }

            for (uint16_t j = 0; j < needPadRight; j++) {
                __ubuf__ T* outAddr = dstAddr + padWROffset;
                if constexpr (sizeof(T) == B64_BYTES) {
                    AscendC::Reg::Duplicate(vReg, padRightValue);
                } else {
                    if constexpr (sizeof(T) == 1) {
                        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B8>(
                            vReg, additionAddr + padRightSrcOffset_);
                    } else if constexpr (sizeof(T) == 2) {
                        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B16>(
                            vReg, additionAddr + padRightSrcOffset_);
                    } else {
                        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                            vReg, additionAddr + padRightSrcOffset_);
                    }
                }

                for (uint16_t k = 0; k < repeatTimes; k++) {
                    AscendC::Reg::StoreUnAlign(outAddr, vReg, uReg, additionLen);
                }

                for (uint16_t k = 0; k < needPadRightSurplus; k++) {
                    outAddr = dstAddr + padWROffsetOnRepeat;
                    AscendC::Reg::StoreUnAlign(outAddr, vReg, uReg, padRightLen);
                }
                AscendC::Reg::StoreUnAlignPost(outAddr, uReg, 0);
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, PadEdgeHugeParam& padParam)
    {
        const uint8_t dimNum = tilingData_->dimNum;
        const uint8_t ubAxis = ubAxis_;
        const uint64_t ubFactor = ubFactor_;
        const uint64_t* outStride = tilingData_->outStride;
        const uint64_t outShapeUb = tilingData_->outShape[ubAxis];

        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum; i++) {
            outAddr += outIndex_[i] * outStride[i];
        }
        uint32_t copyLen = (outIndex_[ubAxis] + ubFactor < outShapeUb ? ubFactor : outShapeUb - outIndex_[ubAxis]);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(output_[outAddr], src[additionOffsetHuge_], copyOutParams);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventIdEdgeHuge = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventIdEdgeHuge);
        WaitFlag<EVENT>(eventIdEdgeHuge);
    }
};
} // namespace PadV3
#endif
