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
 * \file pad_repl_normal_w.h
 * \brief pad cut not last dim kernel in replicate mode
 */

#ifndef PAD_REPL_NORMAL_W_H_
#define PAD_REPL_NORMAL_W_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;
struct PadReplNormalParam {
    uint32_t padH;
    uint32_t padW;
    uint32_t padStride[MAX_H_DIMS];
    uint32_t padHLOffset[MAX_H_DIMS];
    uint32_t padHROffset[MAX_H_DIMS];
    uint32_t padWLOffset;
    uint32_t padWROffset;
};

template <typename T, int32_t KEY>
class KernelPadReplWithNormalWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    static constexpr uint32_t VL_ELEMS = VL_SIZE / sizeof(T);
    static constexpr uint32_t UB_AXES = (KEY / KEY_BASE) % KEY_BASE; // TilingKey倒数第二维为UB内轴个数
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<T> constPad_;
    TBuf<TPosition::VECCALC> inQueue_;

    TPipe* pipe_ = nullptr;
    int64_t blockIdx_;
    int32_t dimNum_{0};
    int32_t dimW_{0};
    int32_t dimH_{0};
    int32_t dimC_{0};
    int32_t dimN_{0};
    T constValue_{0};
    uint32_t additionOffset_{0};
    const PadACTilingData* tilingData_ = nullptr;
    bool is2DPadding{false};
    bool is3DPadding{false};
    bool firstPad{false};
    bool lastOut_{true};
    uint32_t padHLength_{1};
    uint32_t padWLength_{0};
    uint32_t padStride_[MAX_H_DIMS] = {0, 0, 0};
    uint32_t padHLOffset_[MAX_H_DIMS] = {0, 0, 0};
    uint32_t padHROffset_[MAX_H_DIMS] = {1, 1, 1};
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t inCopyLen_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;
    using RT = std::conditional_t<sizeof(T) != sizeof(uint64_t), T, uint32_t>;

public:
    __aicore__ inline KernelPadReplWithNormalWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();
        additionOffset_ = tilingData_->additionTileSize / sizeof(T);
        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM * (tilingData_->outTileSize + tilingData_->additionTileSize));
        dimNum_ = tilingData_->dimNum;
        dimW_ = dimNum_ - 1;
        dimH_ = dimNum_ - CONST2;
        dimC_ = dimNum_ - CONST3;
        dimN_ = dimNum_ - CONST4;
        const int8_t ubAxis = tilingData_->ubAxis;
        padWLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->outShape[dimW_]), BLK_ELEMS);
        for (int8_t i = dimW_; i > ubAxis; i--) {
            inCopyLen_[i] = tilingData_->inShape[i];
            if constexpr (UB_AXES > CONST2) {
                // UB内超过两个才需赋值下标1之后的参数
                if (i != dimW_) {
                    padStride_[i - ubAxis] = padHLength_ * padWLength_;
                    padHLength_ = padHLength_ * tilingData_->outShape[i];
                    padHLOffset_[i - ubAxis] = tilingData_->leftPad[i];
                    padHROffset_[i - ubAxis] = tilingData_->inShape[i] + tilingData_->leftPad[i];
                }
            }
        }
        padStride_[0] = padHLength_ * padWLength_;
        padHLength_ = padHLength_ * tilingData_->ubFactor;
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * tilingData_->ubPerCount;
        if (startIdx >= tilingData_->ubTotalCount) {
            return;
        }
        uint32_t endIdx = (blockIdx_ + 1L) * tilingData_->ubPerCount;
        endIdx = (endIdx < tilingData_->ubTotalCount ? endIdx : tilingData_->ubTotalCount);

        uint8_t ubAxis = tilingData_->ubAxis;
        uint64_t ubFactor = tilingData_->ubFactor;
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            is2DPadding = false;
            is3DPadding = false;
            firstPad = false;
            lastOut_ = (idx == endIdx - 1);
            uint32_t curIdx = idx;
            for (int32_t i = ubAxis; i >= 0; i--) {
                uint64_t factor = tilingData_->outShape[i];
                factor = (i == ubAxis) ? CeilDiv(tilingData_->outShape[i], ubFactor) : factor;
                if (factor != 0) {
                    outIndex_[i] = (i == ubAxis ? curIdx % factor * ubFactor : curIdx % factor);
                }
                inIndex_[i] = outIndex_[i] < tilingData_->leftPad[i] ? 0 : outIndex_[i] - tilingData_->leftPad[i];
                curIdx = (factor != 0) ? curIdx / factor : curIdx;
            }
            if (outIndex_[ubAxis] < tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis]) {
                // outIndex 在右pad点的左侧
                if (outIndex_[ubAxis] + ubFactor <= tilingData_->leftPad[ubAxis]) {
                    // 输出都在左pad点左侧
                    inCopyLen_[ubAxis] = 0;
                    is2DPadding = (UB_AXES == CONST2) ? true : is2DPadding;
                    is3DPadding = (UB_AXES == CONST3) ? true : is3DPadding;
                } else if (outIndex_[ubAxis] + ubFactor < tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis]) {
                    // 输出都在右pad点左侧
                    inCopyLen_[ubAxis] = outIndex_[ubAxis] + ubFactor - inIndex_[ubAxis] - tilingData_->leftPad[ubAxis];
                } else {
                    // 输出跨过右pad点
                    inCopyLen_[ubAxis] = tilingData_->inShape[ubAxis] - inIndex_[ubAxis];
                }
            } else {
                // outIndex 在右pad点的右侧
                inCopyLen_[ubAxis] = 0;
                if (UB_AXES == CONST2) {
                    is2DPadding = true;
                } else if (UB_AXES == CONST3) {
                    is3DPadding = true;
                }
            }
            PadReplNormalParam padParam = {.padH = padHLength_,
                                           .padW = padWLength_,
                                           .padStride = {padStride_[0], padStride_[1], padStride_[2]},
                                           .padHLOffset = {0, padHLOffset_[1], padHLOffset_[2]},
                                           .padHROffset = {tilingData_->ubFactor, padHROffset_[1], padHROffset_[2]},
                                           .padWLOffset = static_cast<uint32_t>(tilingData_->leftPad[dimW_]),
                                           .padWROffset = static_cast<uint32_t>(tilingData_->inShape[dimW_])};
            ProcessOneStep(idx - startIdx, padParam);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(int32_t idx, PadReplNormalParam& padParam)
    {
        // Copy IN
        LocalTensor<T> input = inQueue_.Get<T>();
        LocalTensor<T>
            srcLocal = input[(idx & 1) * (tilingData_->outTileSize + tilingData_->additionTileSize) / sizeof(T)];
        CopyIn(srcLocal, padParam, idx);

        CopyOut(srcLocal, padParam, idx);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadReplNormalParam& padParam, int32_t idx)
    {
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        const int8_t ubAxis = tilingData_->ubAxis;

        if constexpr (UB_AXES == CONST3) {
            CopyandProcess3D(src, padParam, idx);
        } else if constexpr (UB_AXES == CONST2) {
            CopyandProcess2D(src, padParam, idx);
        } else if constexpr (UB_AXES == CONST4) {
            CopyandProcess4D(src, padParam, idx);
        }
    }

    __aicore__ inline void CopyandProcess4D(const LocalTensor<T>& src, PadReplNormalParam& padParam, int32_t idx)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t first = outIndex_[ubAxis];
        uint64_t last = outIndex_[ubAxis] + ubFactor < tilingData_->outShape[ubAxis] ? outIndex_[ubAxis] + ubFactor :
                                                                                       tilingData_->outShape[ubAxis];
        uint64_t padHW = tilingData_->outShape[dimH_] * padParam.padW;
        uint64_t padCHW = tilingData_->outShape[dimC_] * padHW;
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += (inIndex_[i] >= tilingData_->inShape[i] ? tilingData_->inShape[i] - 1 : inIndex_[i]) *
                      tilingData_->inStride[i];
        }

        uint32_t firstOffset = tilingData_->leftPad[dimH_] * padParam.padW + tilingData_->leftPad[dimC_] * padHW;
        uint32_t lastOffset = firstOffset + (inCopyLen_[dimH_] - 1) * padParam.padW;

        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = inCopyLen_[dimH_];
        copyInParams.blockLen = tilingData_->inShape[dimW_] * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (padParam.padW - tilingData_->inShape[dimW_]) / BLK_ELEMS;
        LoopModeParams loopParams;
        loopParams.loop1Size = inCopyLen_[dimC_];
        loopParams.loop1SrcStride = tilingData_->inStride[dimC_] * sizeof(T);
        loopParams.loop1DstStride = tilingData_->outShape[dimH_] * padParam.padW * sizeof(T);
        loopParams.loop2Size = inCopyLen_[dimN_];
        loopParams.loop2SrcStride = tilingData_->inStride[dimN_] * sizeof(T);
        loopParams.loop2DstStride = tilingData_->outShape[dimC_] * tilingData_->outShape[dimH_] * padParam.padW *
                                    sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(src[additionOffset_ + firstOffset], input_[inAddr], copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);

        PadLeftSideFirst(src, padParam, firstOffset);
        PadRightSide(src, padParam, firstOffset, inCopyLen_[dimH_]);
        PadLeftSideLast(src, padParam, firstOffset);
        if (firstOffset != lastOffset) {
            PadLeftSideLast(src, padParam, lastOffset);
        }
        if constexpr (sizeof(T) == B64_BYTES) {
            PadCopyDiff<AscendC::Reg::RegTraitNumTwo>(src, padParam, tilingData_->leftPad[dimC_] * padHW,
                                                      ubAxis + CONST2);
        } else {
            PadCopyDiff<AscendC::Reg::RegTraitNumOne>(src, padParam, tilingData_->leftPad[dimC_] * padHW,
                                                      ubAxis + CONST2);
        }
        PadLeftSide(src, padParam, firstOffset + padParam.padW, false, inCopyLen_[dimH_] - 1);
        if constexpr (sizeof(T) == B64_BYTES) {
            PadCopyDiff<AscendC::Reg::RegTraitNumTwo>(src, padParam, 0, ubAxis + 1);
        } else {
            PadCopyDiff<AscendC::Reg::RegTraitNumOne>(src, padParam, 0, ubAxis + 1);
        }
        PadLeftSide(src, padParam, padHW, true, 1);
    }

    __aicore__ inline void CopyandProcess3D(const LocalTensor<T>& src, PadReplNormalParam& padParam, int32_t idx)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t first = outIndex_[ubAxis];
        uint64_t last = outIndex_[ubAxis] + ubFactor < tilingData_->outShape[ubAxis] ? outIndex_[ubAxis] + ubFactor :
                                                                                       tilingData_->outShape[ubAxis];
        uint64_t padHW = tilingData_->outShape[dimH_] * padParam.padW;
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += (inIndex_[i] >= tilingData_->inShape[i] ? tilingData_->inShape[i] - 1 : inIndex_[i]) *
                      tilingData_->inStride[i];
        }

        uint32_t firstOffset = (first >= tilingData_->leftPad[ubAxis] || is3DPadding) ?
                                   0 :
                                   (tilingData_->leftPad[ubAxis] - first) * padHW;
        firstOffset = firstOffset + tilingData_->leftPad[dimH_] * padParam.padW;

        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = inCopyLen_[dimH_];
        copyInParams.blockLen = tilingData_->inShape[dimW_] * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (padParam.padW - tilingData_->inShape[dimW_]) / BLK_ELEMS;
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = is3DPadding ? 1 : inCopyLen_[dimC_];
        loopParams.loop1SrcStride = tilingData_->inStride[dimC_] * sizeof(T);
        loopParams.loop1DstStride = tilingData_->outShape[dimH_] * padParam.padW * sizeof(T);

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(src[additionOffset_ + firstOffset], input_[inAddr], copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        ProcessPad3D(src, padParam, first, last);
    }

    __aicore__ inline void CopyandProcess2D(const LocalTensor<T>& src, PadReplNormalParam& padParam, int32_t idx)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t inAddr = 0;
        uint32_t copyLen = inCopyLen_[dimH_] > 0 ? inCopyLen_[dimH_] : 1;

        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += (inIndex_[i] >= tilingData_->inShape[i] ? tilingData_->inShape[i] - 1 : inIndex_[i]) *
                      tilingData_->inStride[i];
        }
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = copyLen;
        copyInParams.blockLen = tilingData_->inShape[dimW_] * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (padParam.padW - tilingData_->inShape[dimW_]) / BLK_ELEMS;
        uint32_t ubInOffset = 0;
        uint32_t hlOffsetIdx = dimH_ - ubAxis;
        uint32_t savedHlOffset = padParam.padHLOffset[hlOffsetIdx];
        if (outIndex_[dimH_] < tilingData_->leftPad[dimH_] && !is2DPadding) {
            padParam.padHLOffset[hlOffsetIdx] = tilingData_->leftPad[dimH_] - outIndex_[dimH_];
        }
        for (auto i = 0; i < MAX_H_DIMS; i++) {
            ubInOffset += padParam.padHLOffset[i] * padParam.padStride[i];
        }
        padParam.padHLOffset[hlOffsetIdx] = savedHlOffset;
        DataCopyPad(src[additionOffset_ + ubInOffset], input_[inAddr], copyInParams, padParams);
        // 将最上行或最下行复制到pad位
        uint64_t first = outIndex_[ubAxis];
        uint64_t last = (outIndex_[ubAxis] + ubFactor < tilingData_->outShape[ubAxis] ? outIndex_[ubAxis] + ubFactor :
                                                                                        tilingData_->outShape[ubAxis]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        ProcessPad2D(src, padParam, first, last);
    }

    __aicore__ inline void ProcessPad2D(LocalTensor<T> src, PadReplNormalParam& padParam, uint32_t first, uint32_t last)
    {
        const int8_t ubAxis = tilingData_->ubAxis;

        if (is2DPadding) {
            PadLeftSideFirst(src, padParam, 0);
            PadRightSide(src, padParam, 0, 1);
            PadLeftSideLast(src, padParam, 0);
            if (last > first + 1) {
                if constexpr (sizeof(T) == B64_BYTES) {
                    PadCopySame<AscendC::Reg::RegTraitNumTwo>(src, padParam, 0, padParam.padW, last - first - 1, dimH_);
                } else {
                    PadCopySame<AscendC::Reg::RegTraitNumOne>(src, padParam, 0, padParam.padW, last - first - 1, dimH_);
                }
            }
        } else {
            bool needPadUp = first < tilingData_->leftPad[dimH_];
            bool needPadDown = last > tilingData_->leftPad[dimH_] + tilingData_->inShape[dimH_];
            uint32_t firstOffset = needPadUp ? (tilingData_->leftPad[dimH_] - first) * padParam.padW : 0;
            uint32_t lastOffset = (UB_AXES == 2) ?
                                      firstOffset + (inCopyLen_[ubAxis] - 1) * padParam.padW :
                                      (tilingData_->leftPad[dimH_] + tilingData_->inShape[dimH_] - 1) * padParam.padW;
            PadLeftSideFirst(src, padParam, firstOffset);
            PadRightSide(src, padParam, firstOffset, inCopyLen_[dimH_]);
            if (needPadUp) {
                PadLeftSideLast(src, padParam, firstOffset);
                if constexpr (sizeof(T) == B64_BYTES) {
                    PadCopySame<AscendC::Reg::RegTraitNumTwo>(src, padParam, firstOffset, 0,
                                                              tilingData_->leftPad[dimH_] - first, dimH_);
                } else {
                    PadCopySame<AscendC::Reg::RegTraitNumOne>(src, padParam, firstOffset, 0,
                                                              tilingData_->leftPad[dimH_] - first, dimH_);
                }
            }
            if (inCopyLen_[dimH_] > 1) {
                PadLeftSide(src, padParam, firstOffset + padParam.padW, false, inCopyLen_[dimH_] - 1);
            }
            if (needPadDown) {
                uint32_t padLen = last - tilingData_->leftPad[dimH_] - tilingData_->inShape[dimH_];
                PadLeftSideLast(src, padParam, lastOffset);
                if constexpr (sizeof(T) == B64_BYTES) {
                    PadCopySame<AscendC::Reg::RegTraitNumTwo>(src, padParam, lastOffset, lastOffset + padParam.padW,
                                                              padLen, dimH_);
                } else {
                    PadCopySame<AscendC::Reg::RegTraitNumOne>(src, padParam, lastOffset, lastOffset + padParam.padW,
                                                              padLen, dimH_);
                }
            }
        }
    }

    __aicore__ inline void ProcessPad3D(LocalTensor<T> src, PadReplNormalParam& padParam, uint32_t first, uint32_t last)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        uint32_t padHW = tilingData_->outShape[dimH_] * padParam.padW;
        if (is3DPadding) {
            ProcessPad2D(src, padParam, 0, tilingData_->outShape[dimH_]);
            if (last > first + 1) {
                if constexpr (sizeof(T) == B64_BYTES) {
                    PadCopySame<AscendC::Reg::RegTraitNumTwo>(src, padParam, 0, padHW, last - first - 1, dimC_);
                } else {
                    PadCopySame<AscendC::Reg::RegTraitNumOne>(src, padParam, 0, padHW, last - first - 1, dimC_);
                }
            }
            // 每面第一行需要添加左pad
            PadLeftSide(src, padParam, padHW, true, 1);
        } else {
            bool needPadUp = first < tilingData_->leftPad[dimC_];
            bool needPadDown = last > tilingData_->leftPad[dimC_] + tilingData_->inShape[dimC_];
            uint32_t firstOffset = needPadUp ? (tilingData_->leftPad[dimC_] - first) * padHW : 0;
            uint32_t lastOffset = firstOffset + (inCopyLen_[ubAxis] - 1) * padHW;
            uint32_t copyLen = tilingData_->leftPad[dimC_] - first;
            PadLeftSideFirst(src, padParam, firstOffset + tilingData_->leftPad[dimH_] * padParam.padW);
            PadRightSide(src, padParam, firstOffset + tilingData_->leftPad[dimH_] * padParam.padW, inCopyLen_[dimH_]);
            PadLeftSideLast(src, padParam, firstOffset + tilingData_->leftPad[dimH_] * padParam.padW);
            PadLeftSideLast(
                src, padParam,
                firstOffset + (tilingData_->leftPad[dimH_] + tilingData_->inShape[dimH_] - 1) * padParam.padW);

            if constexpr (sizeof(T) == B64_BYTES) {
                PadCopyDiff<AscendC::Reg::RegTraitNumTwo>(src, padParam, firstOffset, ubAxis + 1);
            } else {
                PadCopyDiff<AscendC::Reg::RegTraitNumOne>(src, padParam, firstOffset, ubAxis + 1);
            }
            // 补每面除第一行以外的左pad
            PadLeftSide(src, padParam, firstOffset + (tilingData_->leftPad[dimH_] + 1) * padParam.padW, false,
                        inCopyLen_[dimH_] - 1);

            if (needPadUp) {
                if constexpr (sizeof(T) == B64_BYTES) {
                    PadCopySame<AscendC::Reg::RegTraitNumTwo>(src, padParam, firstOffset, 0, copyLen, dimC_);
                } else {
                    PadCopySame<AscendC::Reg::RegTraitNumOne>(src, padParam, firstOffset, 0, copyLen, dimC_);
                }
            }
            if (needPadDown) {
                uint32_t padLen = last - tilingData_->leftPad[dimC_] - tilingData_->inShape[dimC_];
                if constexpr (sizeof(T) == B64_BYTES) {
                    PadCopySame<AscendC::Reg::RegTraitNumTwo>(src, padParam, lastOffset, lastOffset + padHW, padLen,
                                                              dimC_);
                } else {
                    PadCopySame<AscendC::Reg::RegTraitNumOne>(src, padParam, lastOffset, lastOffset + padHW, padLen,
                                                              dimC_);
                }
            }
            // 补每面第一行左pad
            PadLeftSide(src, padParam, padHW, true, 1);
        }
    }

    __aicore__ inline void PadRightSide(const LocalTensor<T>& dst, PadReplNormalParam& padParam, uint32_t ubOffset,
                                        uint32_t copylen)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ RT*>(dst.GetPhyAddr());
        const uint16_t sizeNum = (sizeof(T) > CONST4) ? 2 : 1;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimN_];
        const uint16_t dimCNum = (UB_AXES < CONST3 || is3DPadding) ? 1 :
                                                                     (inCopyLen_[dimC_] > 0 ? inCopyLen_[dimC_] : 1);
        const uint32_t padW = padParam.padStride[UB_AXES - CONST2] * sizeNum;
        const uint32_t padHW = (UB_AXES >= CONST3) ? padParam.padStride[UB_AXES - CONST3] * sizeNum : 0;
        const uint32_t padCHW = (UB_AXES >= CONST4) ? padParam.padStride[UB_AXES - CONST4] * sizeNum : 0;
        const uint32_t padRightFloorAlign = CeilAlign(padParam.padWROffset * sizeNum, BLK_ELEMS * sizeNum);
        const uint32_t padLeftlen = (padParam.padWLOffset + padParam.padW - tilingData_->outShape[dimW_]) * sizeNum;
        const uint16_t padLeftVLNum = padLeftlen / (VL_ELEMS * sizeNum);
        const uint16_t padLeftBLNum = (padLeftlen % (VL_ELEMS * sizeNum)) / (BLK_ELEMS * sizeNum);
        const uint32_t PadRightSize = padRightFloorAlign - padParam.padWROffset * sizeNum;
        const uint32_t padRightLen = padParam.padW * sizeNum - padRightFloorAlign - padLeftVLNum * VL_ELEMS * sizeNum -
                                     padLeftBLNum * BLK_ELEMS * sizeNum;
        const uint16_t padRightVLNum = padRightLen / (VL_ELEMS * sizeNum);
        const uint16_t padRightBLNum = (padRightLen % (VL_ELEMS * sizeNum) + BLK_ELEMS * sizeNum - 1) /
                                       (BLK_ELEMS * sizeNum);
        const uint32_t lastOffset = (padParam.padWROffset - 1) * sizeNum;
        const uint16_t firstOffset = ubOffset * sizeNum;
        const uint16_t BLNum = padRightBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg rMask;
            AscendC::Reg::MaskReg outMask;
            AscendC::Reg::MaskReg outNMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();

            uint32_t norPadLen = BLK_ELEMS * sizeNum - PadRightSize;
            uint32_t outLen = BLK_ELEMS * sizeNum;
            uint32_t outLen2 = padRightBLNum * BLK_ELEMS * sizeNum;

            rMask = AscendC::Reg::UpdateMask<RT>(norPadLen);
            AscendC::Reg::Not(rMask, rMask, maskAll);
            outMask = AscendC::Reg::UpdateMask<RT>(outLen);
            AscendC::Reg::And(outNMask, outMask, rMask, maskAll);
            outMask = AscendC::Reg::UpdateMask<RT>(outLen2);

            if constexpr (UB_AXES == CONST2) {
                PadRightSideOne(dstAddr + firstOffset + additionOffset_ * sizeNum, copylen, padW, lastOffset,
                                padRightFloorAlign, padRightVLNum, BLNum, outMask, outNMask);
            } else if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    PadRightSideOne(dstAddr + firstOffset + additionOffset_ * sizeNum + c * padHW, copylen, padW,
                                    lastOffset, padRightFloorAlign, padRightVLNum, BLNum, outMask, outNMask);
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        PadRightSideOne(dstAddr + firstOffset + additionOffset_ * sizeNum + n * padCHW + c * padHW,
                                        copylen, padW, lastOffset, padRightFloorAlign, padRightVLNum, BLNum, outMask,
                                        outNMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void PadRightSideOne(__ubuf__ RT* dstAddr, uint32_t copylen, uint32_t step, uint32_t lastOffset,
                                           uint32_t padRightFloorAlign, uint16_t padRightVLNum, uint16_t padRightBLNum,
                                           Reg::MaskReg outMask, Reg::MaskReg outNMask)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();

            if constexpr (sizeof(T) > CONST4) {
                AscendC::Reg::RegTensor<RT> tmpIn;
                AscendC::Reg::RegTensor<RT> tmpIn1;
                AscendC::Reg::RegTensor<RT> tmpOut;
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn, dstAddr + h * step + lastOffset);
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn1,
                                                                             dstAddr + h * step + lastOffset + 1);
                    Reg::Interleave(vRegTmp, tmpOut, tmpIn, tmpIn1);
                    AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign - BLK_ELEMS * 2, vRegTmp,
                                             outNMask);
                    for (uint16_t i = 0; i < padRightVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + i * VL_ELEMS * 2, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < padRightBLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + padRightVLNum * VL_ELEMS * 2,
                                                 vRegTmp, outMask);
                    }
                }
            } else if constexpr (sizeof(T) == CONST4) {
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(vRegTmp, dstAddr + h * step + lastOffset);
                    AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign - BLK_ELEMS, vRegTmp, outNMask);
                    for (uint16_t i = 0; i < padRightVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + i * VL_ELEMS, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < padRightBLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + padRightVLNum * VL_ELEMS,
                                                 vRegTmp, outMask);
                    }
                }
            } else if constexpr (sizeof(T) == CONST2) {
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B16>(vRegTmp, dstAddr + h * step + lastOffset);
                    AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign - BLK_ELEMS, vRegTmp, outNMask);
                    for (uint16_t i = 0; i < padRightVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + i * VL_ELEMS, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < padRightBLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + padRightVLNum * VL_ELEMS,
                                                 vRegTmp, outMask);
                    }
                }
            } else if constexpr (sizeof(T) == 1) {
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B8>(vRegTmp, dstAddr + h * step + lastOffset);
                    AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign - BLK_ELEMS, vRegTmp, outNMask);
                    for (uint16_t i = 0; i < padRightVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + i * VL_ELEMS, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < padRightBLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + h * step + padRightFloorAlign + padRightVLNum * VL_ELEMS,
                                                 vRegTmp, outMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void PadLeftSideLast(const LocalTensor<T>& dst, PadReplNormalParam& padParam, uint32_t ubOffset)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ RT*>(dst.GetPhyAddr());
        const uint16_t sizeNum = (sizeof(T) > CONST4) ? 2 : 1;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimN_];
        const uint16_t dimCNum = (UB_AXES < CONST3 || is3DPadding) ? 1 :
                                                                     (inCopyLen_[dimC_] > 0 ? inCopyLen_[dimC_] : 1);
        const uint32_t padHW = (UB_AXES >= CONST3) ? padParam.padStride[UB_AXES - CONST3] * sizeNum : 0;
        const uint32_t padCHW = (UB_AXES >= CONST4) ? padParam.padStride[UB_AXES - CONST4] * sizeNum : 0;
        const uint32_t padRight = (tilingData_->outShape[dimW_] - tilingData_->leftPad[dimW_]) * sizeNum;
        const uint32_t padRightFloorAlign = CeilAlign(padRight, BLK_ELEMS * sizeNum);
        const uint32_t padLeftSize = padRightFloorAlign - padRight;
        const uint16_t padLeftVLNum = (padParam.padW * sizeNum - padRightFloorAlign) / (VL_ELEMS * sizeNum);
        const uint16_t padLeftBLNum = ((padParam.padW * sizeNum - padRightFloorAlign) % (VL_ELEMS * sizeNum)) /
                                      (BLK_ELEMS * sizeNum);
        const uint16_t firstOffset = ubOffset * sizeNum;
        const uint16_t BLNum = padLeftBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg rMask;
            AscendC::Reg::MaskReg outMask;
            AscendC::Reg::MaskReg outNMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::RegTensor<RT> tmpIn;
            AscendC::Reg::RegTensor<RT> tmpIn1;
            AscendC::Reg::RegTensor<RT> tmpOut;

            uint32_t norPadLen = BLK_ELEMS * sizeNum - padLeftSize;
            uint32_t outLen = BLK_ELEMS * sizeNum;
            uint32_t outnLen = padLeftBLNum * BLK_ELEMS * sizeNum;
            uint32_t ubInOffset = 0;
            rMask = AscendC::Reg::UpdateMask<RT>(norPadLen);
            AscendC::Reg::Not(rMask, rMask, maskAll);
            outNMask = AscendC::Reg::UpdateMask<RT>(outLen);
            AscendC::Reg::And(outMask, outNMask, rMask, maskAll);
            outNMask = AscendC::Reg::UpdateMask<RT>(outnLen);

            if constexpr (sizeof(T) == 1) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        ubInOffset = firstOffset + n * padCHW + c * padHW + additionOffset_;
                        AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B8>(vRegTmp, dstAddr + ubInOffset);
                        ubInOffset = ubInOffset + padRightFloorAlign;
                        AscendC::Reg::StoreAlign(dstAddr + ubInOffset - BLK_ELEMS, vRegTmp, outMask);
                        for (uint16_t i = 0; i < padLeftVLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + i * VL_ELEMS, vRegTmp, maskAll);
                        }
                        for (uint16_t i = 0; i < BLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + padLeftVLNum * VL_ELEMS, vRegTmp, outNMask);
                        }
                    }
                }
            } else if constexpr (sizeof(T) == CONST2) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        ubInOffset = firstOffset + n * padCHW + c * padHW + additionOffset_;
                        AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B16>(vRegTmp, dstAddr + ubInOffset);
                        ubInOffset = ubInOffset + padRightFloorAlign;
                        AscendC::Reg::StoreAlign(dstAddr + ubInOffset - BLK_ELEMS, vRegTmp, outMask);
                        for (uint16_t i = 0; i < padLeftVLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + i * VL_ELEMS, vRegTmp, maskAll);
                        }
                        for (uint16_t i = 0; i < BLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + padLeftVLNum * VL_ELEMS, vRegTmp, outNMask);
                        }
                    }
                }
            } else if constexpr (sizeof(T) == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        ubInOffset = firstOffset + n * padCHW + c * padHW + additionOffset_;
                        AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vRegTmp, dstAddr + ubInOffset);
                        ubInOffset = ubInOffset + padRightFloorAlign;
                        AscendC::Reg::StoreAlign(dstAddr + ubInOffset - BLK_ELEMS, vRegTmp, outMask);
                        for (uint16_t i = 0; i < padLeftVLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + i * VL_ELEMS, vRegTmp, maskAll);
                        }
                        for (uint16_t i = 0; i < BLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + padLeftVLNum * VL_ELEMS, vRegTmp, outNMask);
                        }
                    }
                }
            } else if constexpr (sizeof(T) > CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        ubInOffset = firstOffset + n * padCHW + c * padHW + additionOffset_ * 2;
                        AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn, dstAddr + ubInOffset);
                        AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn1, dstAddr + ubInOffset + 1);
                        Reg::Interleave(vRegTmp, tmpOut, tmpIn, tmpIn1);
                        ubInOffset = ubInOffset + padRightFloorAlign;
                        AscendC::Reg::StoreAlign(dstAddr + ubInOffset - BLK_ELEMS * 2, vRegTmp, outMask);
                        for (uint16_t i = 0; i < padLeftVLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + i * VL_ELEMS * 2, vRegTmp, maskAll);
                        }
                        for (uint16_t i = 0; i < BLNum; i++) {
                            AscendC::Reg::StoreAlign(dstAddr + ubInOffset + padLeftVLNum * VL_ELEMS * 2, vRegTmp,
                                                     outNMask);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void PadLeftSide(const LocalTensor<T>& dst, PadReplNormalParam& padParam, uint32_t ubOffset,
                                       bool isFirst, uint32_t copylen)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ RT*>(dst.GetPhyAddr());
        const uint16_t sizeNum = (sizeof(T) > CONST4) ? 2 : 1;
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint64_t ubFactor = tilingData_->ubFactor;
        const uint16_t dim3D = dimC_;
        const uint16_t totalLen = (UB_AXES < CONST3) ? 1 :
                                                       ((outIndex_[dim3D] + ubFactor < tilingData_->outShape[dim3D] &&
                                                         UB_AXES == CONST3) ?
                                                            ubFactor :
                                                            tilingData_->outShape[dim3D] - outIndex_[dim3D]);
        const uint16_t tempNNum = UB_AXES < CONST4 ? 1 : inCopyLen_[dimN_];
        const uint16_t dimNNum = (UB_AXES < CONST4 || isFirst) ? 1 : inCopyLen_[dimN_];
        const uint16_t dimCNum = (UB_AXES < CONST3 || (is3DPadding && !isFirst)) ?
                                     1 :
                                     (isFirst ? totalLen * tempNNum - 1 : inCopyLen_[dimC_]);
        const uint32_t padW = padParam.padStride[UB_AXES - CONST2] * sizeNum;
        const uint32_t padHW = (UB_AXES >= CONST3) ? padParam.padStride[UB_AXES - CONST3] * sizeNum : 0;
        const uint32_t padCHW = (UB_AXES >= CONST4) ? padParam.padStride[UB_AXES - CONST4] * sizeNum : 0;
        const uint32_t padLeftlen = (padParam.padWLOffset + padParam.padW - tilingData_->outShape[dimW_]) * sizeNum;
        const uint16_t padLeftVLNum = padLeftlen / (VL_ELEMS * sizeNum);
        const uint32_t PadLeftSize = padLeftlen - padLeftVLNum * VL_ELEMS * sizeNum;
        const uint16_t BLNum = PadLeftSize > 0 ? 1 : 0;
        const uint16_t firstOffset = ubOffset * sizeNum;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg lMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::RegTensor<RT> tmpIn;
            AscendC::Reg::RegTensor<RT> tmpIn1;
            AscendC::Reg::RegTensor<RT> tmpOut;

            uint32_t nolPadLen = VL_ELEMS * sizeNum - PadLeftSize;
            lMask = AscendC::Reg::UpdateMask<RT>(nolPadLen);
            AscendC::Reg::Not(lMask, lMask, maskAll);
            if constexpr (UB_AXES == CONST2) {
                PadLeftSideOne(dstAddr, firstOffset + additionOffset_ * sizeNum, copylen, padW, padLeftVLNum, BLNum,
                               lMask);
            } else if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    PadLeftSideOne(dstAddr, firstOffset + additionOffset_ * sizeNum + c * padHW, copylen, padW,
                                   padLeftVLNum, BLNum, lMask);
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        PadLeftSideOne(dstAddr, firstOffset + additionOffset_ * sizeNum + n * padCHW + c * padHW,
                                       copylen, padW, padLeftVLNum, BLNum, lMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void PadLeftSideOne(__ubuf__ RT* dstAddr, uint32_t firstOffset, uint32_t copylen, uint32_t step,
                                          uint16_t padLeftVLNum, uint16_t BLNum, Reg::MaskReg lMask)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();

            if constexpr (sizeof(T) > CONST4) {
                AscendC::Reg::RegTensor<RT> tmpIn;
                AscendC::Reg::RegTensor<RT> tmpIn1;
                AscendC::Reg::RegTensor<RT> tmpOut;
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn, dstAddr + firstOffset + h * step);
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn1,
                                                                             dstAddr + firstOffset + h * step + 1);
                    Reg::Interleave(vRegTmp, tmpOut, tmpIn, tmpIn1);
                    // 先逐个vreg，再逐个32B
                    for (uint16_t i = 0; i < padLeftVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (i + 1) * VL_ELEMS * 2, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < BLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (padLeftVLNum + 1) * VL_ELEMS * 2,
                                                 vRegTmp, lMask);
                    }
                }
            } else if constexpr (sizeof(T) == CONST4) {
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(vRegTmp, dstAddr + firstOffset + h * step);
                    // 先逐个vreg，再逐个32B
                    for (uint16_t i = 0; i < padLeftVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (i + 1) * VL_ELEMS, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < BLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (padLeftVLNum + 1) * VL_ELEMS,
                                                 vRegTmp, lMask);
                    }
                }
            } else if constexpr (sizeof(T) == CONST2) {
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B16>(vRegTmp, dstAddr + firstOffset + h * step);
                    // 先逐个vreg，再逐个32B
                    for (uint16_t i = 0; i < padLeftVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (i + 1) * VL_ELEMS, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < BLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (padLeftVLNum + 1) * VL_ELEMS,
                                                 vRegTmp, lMask);
                    }
                }
            } else if constexpr (sizeof(T) == 1) {
                for (uint16_t h = 0; h < copylen; h++) {
                    AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B8>(vRegTmp, dstAddr + firstOffset + h * step);
                    // 先逐个vreg，再逐个32B
                    for (uint16_t i = 0; i < padLeftVLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (i + 1) * VL_ELEMS, vRegTmp,
                                                 maskAll);
                    }
                    for (uint16_t i = 0; i < BLNum; i++) {
                        AscendC::Reg::StoreAlign(dstAddr + firstOffset + h * step - (padLeftVLNum + 1) * VL_ELEMS,
                                                 vRegTmp, lMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void PadLeftSideFirst(const LocalTensor<T>& dst, PadReplNormalParam& padParam, uint32_t ubOffset)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ RT*>(dst.GetPhyAddr());
        const uint16_t sizeNum = (sizeof(T) > CONST4) ? 2 : 1;
        const uint32_t padRightFloorAlign = CeilAlign(padParam.padWROffset * sizeNum, BLK_ELEMS * sizeNum);
        const uint32_t padLeftlen = (padParam.padWLOffset + padParam.padW - tilingData_->outShape[dimW_]) * sizeNum;
        const uint16_t padLeftVLNum = padLeftlen / (VL_ELEMS * sizeNum);
        const uint16_t padLeftBLNum = (padLeftlen % (VL_ELEMS * sizeNum)) / (BLK_ELEMS * sizeNum);
        const uint16_t firstOffset = ubOffset * sizeNum;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg rMask;
            AscendC::Reg::MaskReg outMask;

            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();

            uint32_t outLen = BLK_ELEMS * sizeNum;

            outMask = AscendC::Reg::UpdateMask<RT>(outLen);
            AscendC::Reg::RegTensor<RT> tmpIn;
            AscendC::Reg::RegTensor<RT> tmpIn1;
            AscendC::Reg::RegTensor<RT> tmpOut;

            if constexpr (sizeof(T) == 1) {
                AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B8>(vRegTmp,
                                                                       dstAddr + firstOffset + additionOffset_);
                // 第一行左pad，写到临时空间
                AscendC::Reg::StoreAlign(dstAddr, vRegTmp, outMask);
            } else if constexpr (sizeof(T) == CONST2) {
                AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B16>(vRegTmp,
                                                                        dstAddr + firstOffset + additionOffset_);
                // 第一行左pad，写到临时空间
                AscendC::Reg::StoreAlign(dstAddr, vRegTmp, outMask);
            } else if constexpr (sizeof(T) == CONST4) {
                AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vRegTmp,
                                                                        dstAddr + firstOffset + additionOffset_);
                // 第一行左pad，写到临时空间
                AscendC::Reg::StoreAlign(dstAddr, vRegTmp, outMask);
            } else if constexpr (sizeof(T) > CONST4) {
                AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn,
                                                                         dstAddr + firstOffset + 2 * additionOffset_);
                AscendC::Reg::LoadAlign<RT, Reg::LoadDist::DIST_BRC_B32>(
                    tmpIn1, dstAddr + firstOffset + 2 * additionOffset_ + 1);
                Reg::Interleave(vRegTmp, tmpOut, tmpIn, tmpIn1);
                AscendC::Reg::StoreAlign(dstAddr, vRegTmp, outMask);
            }
        }
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadCopySame(const LocalTensor<T>& dst, PadReplNormalParam& padParam, uint32_t inOffset,
                                       uint32_t outOffset, uint32_t copyLen, int8_t curAxis)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ RT*>(dst.GetPhyAddr());
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint16_t sizeNum = (sizeof(T) > CONST4) ? 2 : 1;
        const uint16_t dimHNum = copyLen;
        const uint16_t totalNum = padParam.padStride[curAxis - ubAxis] * sizeNum;
        const uint16_t padVLNum = totalNum / (VL_ELEMS * sizeNum);
        const uint16_t padBLNum = (totalNum % (VL_ELEMS * sizeNum) + (BLK_ELEMS * sizeNum) - 1) / (BLK_ELEMS * sizeNum);
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg outMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();

            uint32_t outLen = BLK_ELEMS * sizeNum * padBLNum;
            outMask = AscendC::Reg::UpdateMask<T, Trait>(outLen);

            for (uint16_t i = 0; i < padVLNum; i++) {
                AscendC::Reg::LoadAlign(vRegTmp, dstAddr + (inOffset + i * VL_ELEMS + additionOffset_) * sizeNum);
                for (uint16_t n = 0; n < dimHNum; n++) {
                    AscendC::Reg::StoreAlign(
                        dstAddr + (outOffset + i * VL_ELEMS + additionOffset_) * sizeNum + n * totalNum, vRegTmp,
                        maskAll);
                }
            }
            for (uint16_t i = 0; i < BLNum; i++) {
                AscendC::Reg::LoadAlign(vRegTmp,
                                        dstAddr + (inOffset + additionOffset_ + padVLNum * VL_ELEMS) * sizeNum);
                for (uint16_t n = 0; n < dimHNum; n++) {
                    AscendC::Reg::StoreAlign(
                        dstAddr + (outOffset + additionOffset_ + padVLNum * VL_ELEMS) * sizeNum + n * totalNum, vRegTmp,
                        outMask);
                }
            }
        }
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadCopyDiff(const LocalTensor<T>& dst, PadReplNormalParam& padParam, uint32_t ubOffset,
                                       int8_t curAxis)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ RT*>(dst.GetPhyAddr());
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint16_t sizeNum = (sizeof(T) > CONST4) ? 2 : 1;
        const uint16_t dimNNum = (curAxis - ubAxis <= 1) ? 1 : inCopyLen_[curAxis - CONST2];
        const uint16_t dimCNum = inCopyLen_[curAxis - 1];
        const uint32_t padHW = padParam.padStride[curAxis - ubAxis - 1] * sizeNum;
        const uint32_t padCHW = (curAxis - ubAxis <= 1) ? 0 : padParam.padStride[curAxis - ubAxis - 2] * sizeNum;
        const uint16_t oneLen = padParam.padStride[curAxis - ubAxis] * sizeNum;
        const uint16_t dimUp = tilingData_->leftPad[curAxis];
        const uint16_t dimDown = tilingData_->outShape[curAxis] - padParam.padHROffset[curAxis - ubAxis];
        if (dimUp == 0 && dimDown == 0) {
            return;
        }
        const uint16_t firstOffset = dimUp * oneLen;
        const uint16_t lastOffset = firstOffset + (inCopyLen_[curAxis] - 1) * oneLen;
        const uint16_t padVLNum = oneLen / (VL_ELEMS * sizeNum);
        const uint16_t padBLNum = (oneLen % (VL_ELEMS * sizeNum) + (BLK_ELEMS * sizeNum) - 1) / (BLK_ELEMS * sizeNum);
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> vRegTmp;
            AscendC::Reg::MaskReg outMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();

            uint32_t outLen = BLK_ELEMS * sizeNum * padBLNum;
            uint32_t inOffset = 0;
            outMask = AscendC::Reg::UpdateMask<T, Trait>(outLen);

            for (uint16_t n = 0; n < dimNNum; n++) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    inOffset = (ubOffset + additionOffset_) * sizeNum + c * padHW + n * padCHW;
                    for (uint16_t i = 0; i < padVLNum; i++) {
                        AscendC::Reg::LoadAlign(vRegTmp, dstAddr + inOffset + i * VL_ELEMS * sizeNum + firstOffset);
                        for (uint16_t h = 0; h < dimUp; h++) {
                            AscendC::Reg::StoreAlign(dstAddr + inOffset + i * VL_ELEMS * sizeNum + h * oneLen, vRegTmp,
                                                     maskAll);
                        }
                        AscendC::Reg::LoadAlign(vRegTmp, dstAddr + inOffset + i * VL_ELEMS * sizeNum + lastOffset);
                        for (uint16_t h = 0; h < dimDown; h++) {
                            AscendC::Reg::StoreAlign(
                                dstAddr + inOffset + i * VL_ELEMS * sizeNum + lastOffset + (h + 1) * oneLen, vRegTmp,
                                maskAll);
                        }
                    }
                    for (uint16_t i = 0; i < BLNum; i++) {
                        AscendC::Reg::LoadAlign(vRegTmp,
                                                dstAddr + inOffset + padVLNum * sizeNum * VL_ELEMS + firstOffset);
                        for (uint16_t h = 0; h < dimUp; h++) {
                            AscendC::Reg::StoreAlign(dstAddr + inOffset + padVLNum * VL_ELEMS * sizeNum + h * oneLen,
                                                     vRegTmp, outMask);
                        }
                        AscendC::Reg::LoadAlign(vRegTmp,
                                                dstAddr + inOffset + padVLNum * sizeNum * VL_ELEMS + lastOffset);
                        for (uint16_t h = 0; h < dimDown; h++) {
                            AscendC::Reg::StoreAlign(
                                dstAddr + inOffset + padVLNum * VL_ELEMS * sizeNum + lastOffset + (h + 1) * oneLen,
                                vRegTmp, outMask);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, PadReplNormalParam& padParam, int32_t idx)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t outAddr = 0;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
        for (uint32_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex_[i] * tilingData_->outStride[i];
        }

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = tilingData_->leftPad[dimW_] * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        if (tilingData_->leftPad[dimW_] != 0) {
            DataCopyExtParams copyOutParams0;
            copyOutParams0.blockCount = 1;
            copyOutParams0.blockLen = BLK_ELEMS * sizeof(T);
            uint32_t addNum = tilingData_->leftPad[dimW_] / BLK_ELEMS;
            for (uint32_t i = 0; i < addNum; i++) {
                DataCopyPad(output_[outAddr + i * BLK_ELEMS], src, copyOutParams0);
            }
            if (tilingData_->leftPad[dimW_] % BLK_ELEMS > 0) {
                copyOutParams0.blockLen = (tilingData_->leftPad[dimW_] % BLK_ELEMS) * sizeof(T);
                DataCopyPad(output_[outAddr + addNum * BLK_ELEMS], src, copyOutParams0);
            }
        }
        uint32_t blockCount = (outIndex_[ubAxis] + ubFactor < tilingData_->outShape[ubAxis] ?
                                   ubFactor :
                                   tilingData_->outShape[ubAxis] - outIndex_[ubAxis]);
        for (auto i = ubAxis + 1; i < dimW_; i++) {
            // blockCount 计算到倒数第二维
            blockCount = blockCount * tilingData_->outShape[i];
        }
        // 如果每行32B对齐，一次性搬出
        if (tilingData_->outShape[dimW_] % BLK_ELEMS == 0) {
            copyOutParams.blockLen = (blockCount * tilingData_->outShape[dimW_] - tilingData_->leftPad[dimW_]) *
                                     sizeof(T);
            DataCopyPad(output_[outAddr + tilingData_->leftPad[dimW_]], src[additionOffset_], copyOutParams);
        } else {
            // 先copy前blockCount - 1行，最后一行单独拷贝
            copyOutParams.blockCount = blockCount - 1;
            copyOutParams.blockLen = tilingData_->outShape[dimW_] * sizeof(T);
            if (copyOutParams.blockCount > 0) {
                DataCopyPad(output_[outAddr + tilingData_->leftPad[dimW_]], src[additionOffset_], copyOutParams);
            }

            DataCopyExtParams copyOutParams1;
            copyOutParams1.blockCount = 1;
            copyOutParams1.blockLen = (tilingData_->outShape[dimW_] - tilingData_->leftPad[dimW_]) * sizeof(T);
            DataCopyPad(output_[outAddr + tilingData_->leftPad[dimW_] +
                                copyOutParams.blockCount * tilingData_->outShape[dimW_]],
                        src[additionOffset_ + copyOutParams.blockCount * padParam.padW], copyOutParams1);
        }
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
    }
};
} // namespace PadV3
#endif
