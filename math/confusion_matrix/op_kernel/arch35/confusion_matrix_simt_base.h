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
 * \file confusion_matrix_simt_base.h
 * \brief The base class of confusion_matrix. The main function of this file is:
 */

#ifndef CONFUSION_MATRIX_SIMT_BASE_H
#define CONFUSION_MATRIX_SIMT_BASE_H

#include "confusion_matrix_tiling_data.h"

namespace ConfusionMatrixSimt {
using namespace AscendC;

constexpr uint32_t OUT_QUE_DEPTH = 1;

constexpr uint32_t NUM_DOUBLE_BUFFER = 1;

constexpr uint16_t DATA_COPY_PAD_BLOCK_COUNT = 1;

constexpr int32_t WEIGHT_ONE = 1;

constexpr uint32_t THREAD_NUM = 1024;

template <typename LABEL_TYPE, typename Y_TYPE>
class ConfusionMatrixSimtBase {
public:
    __aicore__ inline ConfusionMatrixSimtBase(){};
    __aicore__ inline void BaseInit(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
                                    const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe,
                                    bool isWeightEmpty);

protected:
    __aicore__ inline void ParseTilingData(const ConfusionMatrixTilingData* __restrict tilingData);

protected:
    GlobalTensor<LABEL_TYPE> labelsGm_;
    GlobalTensor<LABEL_TYPE> predictionsGm_;
    GlobalTensor<Y_TYPE> weightsGm_;
    GlobalTensor<Y_TYPE> yGm_;

    int32_t blockIdx_ = 0;

    bool isWeightEmpty_ = false;
    int64_t numClasses_ = 0;
    int64_t outputSize_ = 0;
    int64_t inputSize_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;
    int64_t resetBinsCoreNum_ = 0;
    int64_t resetBinsLength_ = 0;
    int64_t resetBinsTailLength_ = 0;
    int64_t fullLoadBufSize_ = 0;
    int64_t batchUbBufSize_ = 0;
};

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtBase<LABEL_TYPE, Y_TYPE>::BaseInit(
    GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
    const ConfusionMatrixTilingData* __restrict tilingData, TPipe* tPipe, bool isWeightEmpty)
{
    ParseTilingData(tilingData);
    isWeightEmpty_ = isWeightEmpty;
    labelsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ LABEL_TYPE*>(labels));
    predictionsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ LABEL_TYPE*>(predictions));
    if (!isWeightEmpty_) {
        weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_TYPE*>(weights));
    }
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_TYPE*>(y));

    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
}

template <typename LABEL_TYPE, typename Y_TYPE>
__aicore__ inline void ConfusionMatrixSimtBase<LABEL_TYPE, Y_TYPE>::ParseTilingData(
    const ConfusionMatrixTilingData* __restrict tilingData)
{
    numClasses_ = tilingData->numClasses;
    outputSize_ = numClasses_ * numClasses_;
    inputSize_ = tilingData->inputSize;
    needCoreNum_ = tilingData->needXCoreNum;
    formerLength_ = tilingData->formerLength;
    tailLength_ = tilingData->tailLength;
    resetBinsCoreNum_ = tilingData->clearYCoreNum;
    resetBinsLength_ = tilingData->clearYFactor;
    resetBinsTailLength_ = tilingData->clearYTail;
    fullLoadBufSize_ = tilingData->fullLoadBufSize;
    batchUbBufSize_ = tilingData->batchUbBufSize;
}

} // namespace ConfusionMatrixSimt

#endif // CONFUSION_MATRIX_SIMT_BASE_H
