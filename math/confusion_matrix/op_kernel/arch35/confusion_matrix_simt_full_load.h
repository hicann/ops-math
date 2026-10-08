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
 * \file confusion_matrix_simt_full_load.h
 * \brief The implementation of confusion_matrix by full load in ub.
 */

#ifndef CONFUSION_MATRIX_SIMT_FULL_LOAD_H
#define CONFUSION_MATRIX_SIMT_FULL_LOAD_H

#include "kernel_operator.h"
#include "confusion_matrix_simt_base.h"
#include "confusion_matrix_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ConfusionMatrixSimt {
using namespace AscendC;

template <typename LABEL_TYPE, typename Y_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtCompute(
    __gm__ LABEL_TYPE* labelsGmAddr, __gm__ LABEL_TYPE* predictionsGmAddr, __ubuf__ Y_TYPE* yLocalAddr,
    const int64_t inputOffset, const int64_t inputDataLength, const int64_t numClasses)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < inputDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t label = static_cast<int64_t>(labelsGmAddr[inputOffset + index]);
        int64_t prediction = static_cast<int64_t>(predictionsGmAddr[inputOffset + index]);
        int64_t offset = label * numClasses + prediction;
        asc_atomic_add(yLocalAddr + offset, static_cast<Y_TYPE>(WEIGHT_ONE));
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeWithWeight(
    __gm__ LABEL_TYPE* labelsGmAddr, __gm__ LABEL_TYPE* predictionsGmAddr, __gm__ Y_TYPE* weightsGmAddr,
    __ubuf__ Y_TYPE* yLocalAddr, const int64_t inputOffset, const int64_t inputDataLength, const int64_t numClasses)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < inputDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t label = static_cast<int64_t>(labelsGmAddr[inputOffset + index]);
        int64_t prediction = static_cast<int64_t>(predictionsGmAddr[inputOffset + index]);
        int64_t offset = label * numClasses + prediction;
        asc_atomic_add(yLocalAddr + offset, weightsGmAddr[inputOffset + index]);
    }
}

template <typename LABEL_TYPE, typename Y_TYPE>
class ConfusionMatrixSimtFullLoad : public ConfusionMatrixSimtBase<LABEL_TYPE, Y_TYPE> {
public:
    __aicore__ inline ConfusionMatrixSimtFullLoad(){};
    __aicore__ inline void Init(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
                                const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe,
                                bool isWeightEmpty);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ResetBins();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t offsetInGm, uint32_t stride);

private:
    LocalTensor<Y_TYPE> yLocal_;
    TPipe* tPipe_;
    TQue<TPosition::VECOUT, OUT_QUE_DEPTH> yQue_;
};

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtFullLoad<LABEL_TYPE, Y_TYPE>::Init(
    GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
    const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(labels, predictions, weights, y, tilingData, tPipe, isWeightEmpty);
    tPipe_ = tPipe;
    tPipe_->InitBuffer(yQue_, NUM_DOUBLE_BUFFER, this->fullLoadBufSize_);
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtFullLoad<LABEL_TYPE, Y_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    ResetBins();
    SyncAll();
    Compute();
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtFullLoad<LABEL_TYPE, Y_TYPE>::ResetBins()
{
    if (this->blockIdx_ >= this->resetBinsCoreNum_) {
        return;
    }

    yLocal_ = yQue_.template AllocTensor<Y_TYPE>();
    Duplicate<Y_TYPE>(yLocal_, 0, this->outputSize_);
    yQue_.EnQue(yLocal_);
    yLocal_ = yQue_.template DeQue<Y_TYPE>();

    int64_t resetBinsDataLength = this->blockIdx_ == this->resetBinsCoreNum_ - 1 ? this->resetBinsTailLength_ :
                                                                                   this->resetBinsLength_;
    uint32_t resetBinsDataByteSize = static_cast<uint32_t>(resetBinsDataLength * sizeof(Y_TYPE));
    DataCopyExtParams dataCopyExtParams{DATA_COPY_PAD_BLOCK_COUNT, resetBinsDataByteSize, 0, 0, 0};
    DataCopyPad(this->yGm_[this->blockIdx_ * this->resetBinsLength_], yLocal_, dataCopyExtParams);
    yQue_.template FreeTensor<Y_TYPE>(yLocal_);
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtFullLoad<LABEL_TYPE, Y_TYPE>::Compute()
{
    if (this->blockIdx_ >= this->needCoreNum_) {
        return;
    }

    yLocal_ = yQue_.template AllocTensor<Y_TYPE>();
    Duplicate<Y_TYPE>(yLocal_, 0, this->outputSize_);
    int64_t inputOffset = this->blockIdx_ * this->formerLength_;
    int64_t inputDataLength = this->blockIdx_ == this->needCoreNum_ - 1 ? this->tailLength_ : this->formerLength_;
    __gm__ LABEL_TYPE* labelsGmAddr = (__gm__ LABEL_TYPE*)this->labelsGm_.GetPhyAddr();
    __gm__ LABEL_TYPE* predictionsGmAddr = (__gm__ LABEL_TYPE*)this->predictionsGm_.GetPhyAddr();
    __ubuf__ Y_TYPE* yLocalAddr = (__ubuf__ Y_TYPE*)yLocal_.GetPhyAddr();

    if (this->isWeightEmpty_) {
        asc_vf_call<UbSimtCompute<LABEL_TYPE, Y_TYPE>>(dim3{THREAD_NUM}, labelsGmAddr, predictionsGmAddr, yLocalAddr,
                                                       inputOffset, inputDataLength, this->numClasses_);
    } else {
        __gm__ Y_TYPE* weightsGmAddr = (__gm__ Y_TYPE*)this->weightsGm_.GetPhyAddr();
        asc_vf_call<UbSimtComputeWithWeight<LABEL_TYPE, Y_TYPE>>(dim3{THREAD_NUM}, labelsGmAddr, predictionsGmAddr,
                                                                 weightsGmAddr, yLocalAddr, inputOffset,
                                                                 inputDataLength, this->numClasses_);
    }

    CopyOut(0, static_cast<uint32_t>(this->outputSize_ * sizeof(Y_TYPE)));
    yQue_.template FreeTensor<Y_TYPE>(yLocal_);
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtFullLoad<LABEL_TYPE, Y_TYPE>::CopyOut(int64_t offsetInGm, uint32_t stride)
{
    yQue_.EnQue(yLocal_);
    yLocal_ = yQue_.template DeQue<Y_TYPE>();
    SetAtomicAdd<Y_TYPE>();
    DataCopyExtParams dataCopyExtParams{DATA_COPY_PAD_BLOCK_COUNT, stride, 0, 0, 0};
    DataCopyPad(this->yGm_[offsetInGm], yLocal_, dataCopyExtParams);
    SetAtomicNone();
}

} // namespace ConfusionMatrixSimt
#endif // CONFUSION_MATRIX_SIMT_FULL_LOAD_H
