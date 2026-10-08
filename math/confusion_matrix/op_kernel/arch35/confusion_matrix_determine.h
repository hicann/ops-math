/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confusion_matrix_determine.h
 * \brief confusion_matrix determine realization
 */

#ifndef CONFUSION_MATRIX_DETERMINE_H
#define CONFUSION_MATRIX_DETERMINE_H

#include "confusion_matrix_tiling_data.h"

namespace ConfusionMatrixSimt {
using namespace AscendC;

template <typename LABEL_TYPE, typename Y_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void DetermineSimtCompute(
    __gm__ LABEL_TYPE* labelsGmAddr, __gm__ LABEL_TYPE* predictionsGmAddr, __gm__ Y_TYPE* yGmAddr,
    const int64_t binsHeadIndex, const int64_t binsDataLength, const int64_t inputSize, const int64_t numClasses)
{
    for (int64_t index = static_cast<int64_t>(Simt::GetThreadIdx()); index < binsDataLength;
         index += static_cast<int64_t>(Simt::GetThreadNum<0>())) {
        yGmAddr[binsHeadIndex + index] = static_cast<Y_TYPE>(0);
        for (int64_t inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            int64_t label = static_cast<int64_t>(labelsGmAddr[inputIndex]);
            int64_t prediction = static_cast<int64_t>(predictionsGmAddr[inputIndex]);
            int64_t offset = label * numClasses + prediction;
            if (offset == binsHeadIndex + index) {
                yGmAddr[binsHeadIndex + index] += static_cast<Y_TYPE>(WEIGHT_ONE);
            }
        }
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void DetermineSimtComputeWithWeight(
    __gm__ LABEL_TYPE* labelsGmAddr, __gm__ LABEL_TYPE* predictionsGmAddr, __gm__ Y_TYPE* weightsGmAddr,
    __gm__ Y_TYPE* yGmAddr, const int64_t binsHeadIndex, const int64_t binsDataLength, const int64_t inputSize,
    const int64_t numClasses)
{
    for (int64_t index = static_cast<int64_t>(Simt::GetThreadIdx()); index < binsDataLength;
         index += static_cast<int64_t>(Simt::GetThreadNum<0>())) {
        yGmAddr[binsHeadIndex + index] = static_cast<Y_TYPE>(0);
        for (int64_t inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            int64_t label = static_cast<int64_t>(labelsGmAddr[inputIndex]);
            int64_t prediction = static_cast<int64_t>(predictionsGmAddr[inputIndex]);
            int64_t offset = label * numClasses + prediction;
            if (offset == binsHeadIndex + index) {
                yGmAddr[binsHeadIndex + index] += weightsGmAddr[inputIndex];
            }
        }
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
class ConfusionMatrixDetermine : public ConfusionMatrixSimtBase<LABEL_TYPE, Y_TYPE> {
public:
    __aicore__ inline ConfusionMatrixDetermine(){};
    __aicore__ inline void Init(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
                                const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe,
                                bool isWeightEmpty);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute();

private:
    int64_t binsFormerLength_ = 0;
    int64_t needBinsCoreNum_ = 0;
    int64_t binsTailLength_ = 0;
};

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixDetermine<LABEL_TYPE, Y_TYPE>::Init(
    GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
    const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(labels, predictions, weights, y, tilingData, tPipe, isWeightEmpty);
    binsFormerLength_ = tilingData->binsFormerLength;
    needBinsCoreNum_ = tilingData->needBinsCoreNum;
    binsTailLength_ = tilingData->binsTailLength;
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixDetermine<LABEL_TYPE, Y_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    if (this->blockIdx_ < needBinsCoreNum_) {
        Compute();
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixDetermine<LABEL_TYPE, Y_TYPE>::Compute()
{
    __gm__ LABEL_TYPE* labelsGmAddr = (__gm__ LABEL_TYPE*)this->labelsGm_.GetPhyAddr();
    __gm__ LABEL_TYPE* predictionsGmAddr = (__gm__ LABEL_TYPE*)this->predictionsGm_.GetPhyAddr();
    __gm__ Y_TYPE* yGmAddr = (__gm__ Y_TYPE*)this->yGm_.GetPhyAddr();
    int64_t binsHeadIndex = this->blockIdx_ * binsFormerLength_;
    int64_t binsDataLength = (this->blockIdx_ == needBinsCoreNum_ - 1) ? binsTailLength_ : binsFormerLength_;

    if (this->isWeightEmpty_) {
        Simt::VF_CALL<DetermineSimtCompute<LABEL_TYPE, Y_TYPE>>(Simt::Dim3{THREAD_NUM}, labelsGmAddr, predictionsGmAddr,
                                                                yGmAddr, binsHeadIndex, binsDataLength,
                                                                this->inputSize_, this->numClasses_);
    } else {
        __gm__ Y_TYPE* weightsGmAddr = (__gm__ Y_TYPE*)this->weightsGm_.GetPhyAddr();
        Simt::VF_CALL<DetermineSimtComputeWithWeight<LABEL_TYPE, Y_TYPE>>(
            Simt::Dim3{THREAD_NUM}, labelsGmAddr, predictionsGmAddr, weightsGmAddr, yGmAddr, binsHeadIndex,
            binsDataLength, this->inputSize_, this->numClasses_);
    }
}

} // namespace ConfusionMatrixSimt
#endif // CONFUSION_MATRIX_DETERMINE_H
