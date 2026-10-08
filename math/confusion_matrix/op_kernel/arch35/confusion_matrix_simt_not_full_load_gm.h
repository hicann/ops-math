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
 * \file confusion_matrix_simt_not_full_load_gm.h
 * \brief The implementation of confusion_matrix by not full load in gm.
 */

#ifndef CONFUSION_MATRIX_SIMT_NOT_FULL_LOAD_GM_H
#define CONFUSION_MATRIX_SIMT_NOT_FULL_LOAD_GM_H

#include "confusion_matrix_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ConfusionMatrixSimt {
using namespace AscendC;

template <typename LABEL_TYPE, typename Y_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmSimtCompute(
    __gm__ LABEL_TYPE* labelsGmAddr, __gm__ LABEL_TYPE* predictionsGmAddr, __gm__ Y_TYPE* yGmAddr,
    const int64_t inputOffset, const int64_t inputDataLength, const int64_t numClasses)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < inputDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t label = static_cast<int64_t>(labelsGmAddr[inputOffset + index]);
        int64_t prediction = static_cast<int64_t>(predictionsGmAddr[inputOffset + index]);
        int64_t offset = label * numClasses + prediction;
        asc_atomic_add(yGmAddr + offset, static_cast<Y_TYPE>(WEIGHT_ONE));
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmSimtComputeWithWeight(
    __gm__ LABEL_TYPE* labelsGmAddr, __gm__ LABEL_TYPE* predictionsGmAddr, __gm__ Y_TYPE* weightsGmAddr,
    __gm__ Y_TYPE* yGmAddr, const int64_t inputOffset, const int64_t inputDataLength, const int64_t numClasses)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < inputDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t label = static_cast<int64_t>(labelsGmAddr[inputOffset + index]);
        int64_t prediction = static_cast<int64_t>(predictionsGmAddr[inputOffset + index]);
        int64_t offset = label * numClasses + prediction;
        asc_atomic_add(yGmAddr + offset, weightsGmAddr[inputOffset + index]);
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
class ConfusionMatrixSimtNotFullLoadGm : public ConfusionMatrixSimtBase<LABEL_TYPE, Y_TYPE> {
public:
    __aicore__ inline ConfusionMatrixSimtNotFullLoadGm(){};
    __aicore__ inline void Init(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
                                const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe,
                                bool isWeightEmpty);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ResetBins();
    __aicore__ inline void Compute();
};

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, Y_TYPE>::Init(
    GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
    const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(labels, predictions, weights, y, tilingData, tPipe, isWeightEmpty);
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, Y_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    ResetBins();
    SyncAll();
    Compute();
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, Y_TYPE>::ResetBins()
{
    if (this->blockIdx_ >= this->resetBinsCoreNum_) {
        return;
    }

    int64_t resetBinsHeadIndex = this->blockIdx_ * this->resetBinsLength_;
    int64_t resetBinsDataLength = (this->blockIdx_ == this->resetBinsCoreNum_ - 1) ? this->resetBinsTailLength_ :
                                                                                     this->resetBinsLength_;

    InitOutput<Y_TYPE>(this->yGm_[resetBinsHeadIndex], resetBinsDataLength, static_cast<Y_TYPE>(0));
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, Y_TYPE>::Compute()
{
    if (this->blockIdx_ >= this->needCoreNum_) {
        return;
    }

    int64_t inputOffset = this->blockIdx_ * this->formerLength_;
    int64_t inputDataLength = (this->blockIdx_ == this->needCoreNum_ - 1) ? this->tailLength_ : this->formerLength_;
    __gm__ LABEL_TYPE* labelsGmAddr = (__gm__ LABEL_TYPE*)this->labelsGm_.GetPhyAddr();
    __gm__ LABEL_TYPE* predictionsGmAddr = (__gm__ LABEL_TYPE*)this->predictionsGm_.GetPhyAddr();
    __gm__ Y_TYPE* yGmAddr = (__gm__ Y_TYPE*)this->yGm_.GetPhyAddr();

    if (this->isWeightEmpty_) {
        asc_vf_call<GmSimtCompute<LABEL_TYPE, Y_TYPE>>(dim3{THREAD_NUM}, labelsGmAddr, predictionsGmAddr, yGmAddr,
                                                       inputOffset, inputDataLength, this->numClasses_);
    } else {
        __gm__ Y_TYPE* weightsGmAddr = (__gm__ Y_TYPE*)this->weightsGm_.GetPhyAddr();
        asc_vf_call<GmSimtComputeWithWeight<LABEL_TYPE, Y_TYPE>>(dim3{THREAD_NUM}, labelsGmAddr, predictionsGmAddr,
                                                                 weightsGmAddr, yGmAddr, inputOffset, inputDataLength,
                                                                 this->numClasses_);
    }
}

} // namespace ConfusionMatrixSimt
#endif // CONFUSION_MATRIX_SIMT_NOT_FULL_LOAD_GM_H
