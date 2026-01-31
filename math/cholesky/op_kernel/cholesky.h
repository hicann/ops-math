/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "kernel_operator.h"

using namespace AscendC;

namespace Cholesky {
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t BASIC_BLOCK = 32;

template <typename T>
class Cholesky {
public:
    __aicore__ inline Cholesky(){};
    __aicore__ inline void InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void ProcessTril();
    __aicore__ inline void ProcessTriu();

private:
    __aicore__ inline void GetTilingData(const CholeskyTilingData* tilingData);
    __aicore__ inline void FirstColumn(uint32_t offsetPrefix, uint32_t matrixoffset);
    __aicore__ inline void FirstRow(uint32_t offsetPrefix, uint32_t matrixoffset);
    __aicore__ inline void SecondToNColumn(uint32_t index, uint32_t offsetPrefix, uint32_t matrixoffset);
    __aicore__ inline void SecondToNRow(uint32_t index, uint32_t offsetPrefix, uint32_t matrixoffset);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        return b == 0 ? a : (a + b -1) / b;
    }

private:
    uint32_t matSizeN = 0;
    uint32_t matrixNumCount = 0;
    uint32_t maxDataCount = 0;
    int32_t blockIdx = 0;
    int32_t numBlocks = 0;

    TQue<QuePosition::VECIN, BUFFER_NUM> matAQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> matLeftQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> matRightQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> matResultQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> matLQueue;

    GlobalTensor<T> matAGM;
    GlobalTensor<T> outGM;
};

template <typename T>
__aicore__ inline void Cholesky<T>::InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData, TPipe* pipe) {
    blockIdx = GetBlockIdx();
    numBlocks = GetBlockNum();
    GetTilingData(tilingData);
    
    matAGM.SetGlobalBuffer((__gm__ T*)self, matSizeN * matSizeN);
    outGM.SetGlobalBuffer((__gm__ T*)out, matSizeN * matSizeN);

    uint32_t columnBufferSize = matSizeN * BASIC_BLOCK;
    uint32_t rowBufferSize = CeilA2B(matSizeN * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK;

    pipe->InitBuffer(matAQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLeftQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matRightQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matResultQueue, BUFFER_NUM, rowBufferSize);

    maxDataCount = CeilA2B(matSizeN, BASIC_BLOCK) * BASIC_BLOCK;
}

template <typename T>
__aicore__ inline void Cholesky<T>::InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData, TPipe* pipe) {
    blockIdx = GetBlockIdx();
    numBlocks = GetBlockNum();
    GetTilingData(tilingData);
    
    matAGM.SetGlobalBuffer((__gm__ T*)self, matSizeN * matSizeN);
    outGM.SetGlobalBuffer((__gm__ T*)out, matSizeN * matSizeN);

    uint32_t columnBufferSize = matSizeN * BASIC_BLOCK;
    uint32_t rowBufferSize = CeilA2B(matSizeN * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK;

    pipe->InitBuffer(matAQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLeftQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matRightQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matResultQueue, BUFFER_NUM, columnBufferSize);

    maxDataCount = CeilA2B(matSizeN, BASIC_BLOCK) * BASIC_BLOCK;
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTril() {
    if (blockIdx < numBlocks) {
        auto loopTimes = matrixNumCount / numBlocks;
        for (uint32_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint32_t offsetPrefix = blockIdx + numBlocks * loopIndex;
            if (offsetPrefix < matrixNumCount) {
                uint32_t offset = offsetPrefix * matSizeN * matSizeN;
                FirstColumn(offsetPrefix, offset);
                for (uint32_t index = 1; index < matSizeN; index++) {
                    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();
                    matAQueue.EnQue(matALocal);

                    LocalTensor<T> matLLocal = matLQueue.AllocTensor<T>();
                    matLQueue.EnQue(matLLocal);

                    LocalTensor<T> matLeftLocal = matLeftQueue.AllocTensor<T>();
                    matLeftQueue.EnQue(matLeftLocal);

                    LocalTensor<T> matRightLocal = matRightQueue.AllocTensor<T>();
                    matRightQueue.EnQue(matRightLocal);

                    LocalTensor<T> matResultLocal = matResultQueue.AllocTensor<T>();
                    matResultQueue.EnQue(matResultLocal);

                    SecondToNColumn(index, offsetPrefix, offset);

                    matResultQueue.FreeTensor(matResultLocal);
                    matRightQueue.FreeTensor(matRightLocal);
                    matLeftQueue.FreeTensor(matLeftLocal);
                    matLQueue.FreeTensor(matLLocal);
                    matAQueue.FreeTensor(matALocal);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTriu() {
    if (blockIdx < numBlocks) {
        auto loopTimes = matrixNumCount / numBlocks;
        for (uint32_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint32_t offsetPrefix = blockIdx + numBlocks * loopIndex;
            if (offsetPrefix < matrixNumCount) {
                uint32_t offset = offsetPrefix * matSizeN * matSizeN;
                FirstRow(offsetPrefix, offset);
                for (uint32_t index = 1; index < matSizeN; index++) {
                    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();
                    matAQueue.EnQue(matALocal);

                    LocalTensor<T> matLLocal = matLQueue.AllocTensor<T>();
                    matLQueue.EnQue(matLLocal);

                    LocalTensor<T> matLeftLocal = matLeftQueue.AllocTensor<T>();
                    matLeftQueue.EnQue(matLeftLocal);

                    LocalTensor<T> matRightLocal = matRightQueue.AllocTensor<T>();
                    matRightQueue.EnQue(matRightLocal);

                    LocalTensor<T> matResultLocal = matResultQueue.AllocTensor<T>();
                    matResultQueue.EnQue(matResultLocal);

                    SecondToNRow(index, offsetPrefix, offset);

                    matResultQueue.FreeTensor(matResultLocal);
                    matRightQueue.FreeTensor(matRightLocal);
                    matLeftQueue.FreeTensor(matLeftLocal);
                    matLQueue.FreeTensor(matLLocal);
                    matAQueue.FreeTensor(matALocal);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::GetTilingData(const CholeskyTilingData* tilingData) {
    matSizeN = tilingData->matSizeN;
    matrixNumCount = tilingData->matrixNumCount;
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstColumn(uint32_t offsetPrefix, uint32_t matrixoffset) {
    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();
    DataCopyParams copyParamsMatALocal {static_cast<uint16_t>(matSizeN), sizeof(T), static_cast<uint16_t>((matSizeN - 1) * sizeof(T)), 0};
    DataCopyPadParams padParamsMatALocal {true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
    DataCopyPad(matALocal, matAGM[matrixoffset], copyParamsMatALocal, padParamsMatALocal);
    PipeBarrier<PIPE_ALL>();
    matAQueue.EnQue(matALocal);

    matALocal = matAQueue.DeQue<T>();
    T A11_sqrt = matALocal.GetValue(0);
    if (matrixNumCount > 1) {
        ascendc_assert(A11_sqrt > 0.0f, "(Batch element %d): The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).\n", offsetPrefix);
    } else {
        ascendc_assert(A11_sqrt > 0.0f, "The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).\n");
    }
    A11_sqrt = sqrt(A11_sqrt);
    PipeBarrier<PIPE_ALL>();
    Muls(matALocal, matALocal, T(1/A11_sqrt), matSizeN * BASIC_BLOCK / sizeof(T));
    PipeBarrier<PIPE_ALL>();
    DataCopyParams dataCopyOutParams {static_cast<uint16_t>(matSizeN), sizeof(T), 0, static_cast<uint16_t>((matSizeN - 1) * sizeof(T))};
    DataCopyPad(outGM[matrixoffset], matALocal, dataCopyOutParams);
    PipeBarrier<PIPE_ALL>();
    matAQueue.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstRow(uint32_t offsetPrefix, uint32_t matrixoffset) {
    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();
    DataCopy(matALocal, matAGM[matrixoffset], maxDataCount);
    PipeBarrier<PIPE_ALL>();
    matAQueue.EnQue(matALocal);

    matALocal = matAQueue.DeQue<T>();
    T A11_sqrt = matALocal.GetValue(0);
    PipeBarrier<PIPE_ALL>();
    if (matrixNumCount > 1) {
        ascendc_assert(A11_sqrt > 0.0f, "(Batch element %d): The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).\n", offsetPrefix);
    } else {
        ascendc_assert(A11_sqrt > 0.0f, "The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).\n");
    }
    PipeBarrier<PIPE_ALL>();
    A11_sqrt = sqrt(A11_sqrt);
    PipeBarrier<PIPE_ALL>();
    Muls(matALocal, matALocal, T(1/A11_sqrt), matSizeN);
    PipeBarrier<PIPE_ALL>();
    DataCopyParams dataCopyOutParams {1, static_cast<uint16_t>(sizeof(T) * matSizeN), 0, 0};
    DataCopyPad(outGM[matrixoffset], matALocal, dataCopyOutParams);
    PipeBarrier<PIPE_ALL>();
    matAQueue.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNColumn(uint32_t index, uint32_t offsetPrefix, uint32_t matrixoffset) {
    LocalTensor<T> matALocal = matAQueue.DeQue<T>();
    DataCopyParams copyParamsMatALocal {static_cast<uint16_t>(matSizeN - index), sizeof(T), static_cast<uint16_t>((matSizeN - 1) * sizeof(T)), 0};
    DataCopyPadParams padParamsMatALocal {true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
    DataCopyPad(matALocal, matAGM[matrixoffset + index * matSizeN + index], copyParamsMatALocal, padParamsMatALocal);
    PipeBarrier<PIPE_ALL>();
    
    LocalTensor<T> matLeftLocal = matLeftQueue.DeQue<T>();
    DataCopyParams copyParamsMatLocal {1, static_cast<uint16_t>(sizeof(T) * index), 0, 0};
    DataCopyPadParams padParamsMatLocal {false, 0, 0, 0};
    DataCopyPad(matLeftLocal, outGM[matrixoffset + index * matSizeN], copyParamsMatLocal, padParamsMatLocal);
    PipeBarrier<PIPE_ALL>();
    
    LocalTensor<T> matRightLocal = matRightQueue.DeQue<T>();
    LocalTensor<T> matResultLocal = matResultQueue.DeQue<T>();

    LocalTensor<T> matLLocal = matLQueue.DeQue<T>();
    for (uint32_t i = 0; i < matSizeN - index; i++) {
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(matRightLocal, outGM[matrixoffset + (index + i) * matSizeN], copyParamsMatLocal, padParamsMatLocal);
        PipeBarrier<PIPE_ALL>();
        Mul(matResultLocal, matLeftLocal, matRightLocal, index);
        PipeBarrier<PIPE_ALL>();
        ReduceSum<T>(matResultLocal, matResultLocal, matResultLocal, index);
        PipeBarrier<PIPE_ALL>();
        matLLocal.SetValue(i * BASIC_BLOCK / sizeof(T), matResultLocal.GetValue(0));
    }

    PipeBarrier<PIPE_ALL>();
    Sub(matLLocal, matALocal, matLLocal, (matSizeN - index) * BASIC_BLOCK / sizeof(T));
    PipeBarrier<PIPE_ALL>();
    T b1 = matLLocal.GetValue(0);
    if (matrixNumCount > 1) {
        ascendc_assert(b1 > 0.0f, "(Batch element %d): The factorization could not be completed because the input is not positive-definite (the leading minor of order %d is not positive-definite).\n", offsetPrefix, index + 1);
    } else {
        ascendc_assert(b1 > 0.0f, "The factorization could not be completed because the input is not positive-definite (the leading minor of order %d is not positive-definite).\n", index + 1);
    }
    b1 = sqrt(b1);
    PipeBarrier<PIPE_ALL>();
    Muls(matLLocal, matLLocal, T(1/b1), (matSizeN - index) * BASIC_BLOCK / sizeof(T));
    PipeBarrier<PIPE_ALL>();
    DataCopyParams dataCopyOutParams {static_cast<uint16_t>(matSizeN - index), sizeof(T), 0, static_cast<uint16_t>((matSizeN - 1) * sizeof(T))};
    DataCopyPad(outGM[matrixoffset + index * matSizeN + index], matLLocal, dataCopyOutParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNRow(uint32_t index, uint32_t offsetPrefix, uint32_t matrixoffset) {
    LocalTensor<T> matALocal = matAQueue.DeQue<T>();
    PipeBarrier<PIPE_ALL>();
    DataCopyParams copyParamsMatALocal {1, static_cast<uint16_t>(sizeof(T) * (matSizeN - index)), 0, 0};
    DataCopyPadParams padParamsMatALocal {false, 0, 0, 0};
    DataCopyPad(matALocal, matAGM[matrixoffset + index * matSizeN + index], copyParamsMatALocal, padParamsMatALocal);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<T> matLeftLocal = matLeftQueue.DeQue<T>();
    DataCopyParams copyParamsMatLocal {static_cast<uint16_t>(index), sizeof(T), static_cast<uint16_t>((matSizeN - 1) * sizeof(T)), 0};
    DataCopyPadParams padParamsMatLocal {true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
    DataCopyPad(matLeftLocal, outGM[matrixoffset + index], copyParamsMatLocal, padParamsMatLocal);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<T> matRightLocal = matRightQueue.DeQue<T>();
    LocalTensor<T> matResultLocal = matResultQueue.DeQue<T>();
    LocalTensor<T> matLLocal = matLQueue.DeQue<T>();
    for (uint32_t i = 0; i < matSizeN - index; i++) {
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(matRightLocal, outGM[matrixoffset + index + i], copyParamsMatLocal, padParamsMatLocal);
        PipeBarrier<PIPE_ALL>();
        Mul(matResultLocal, matLeftLocal, matRightLocal, index * BASIC_BLOCK / sizeof(T));
        PipeBarrier<PIPE_ALL>();
        ReduceSum<T>(matResultLocal, matResultLocal, matResultLocal, index * BASIC_BLOCK / sizeof(T));
        PipeBarrier<PIPE_ALL>();
        matLLocal.SetValue(i, matResultLocal.GetValue(0));
        PipeBarrier<PIPE_ALL>();
    }

    PipeBarrier<PIPE_ALL>();
    Sub(matLLocal, matALocal, matLLocal, matSizeN - index);
    
    PipeBarrier<PIPE_ALL>();
    T b1 = matLLocal.GetValue(0);
    PipeBarrier<PIPE_ALL>();
    if (matrixNumCount > 1) {
        ascendc_assert(b1 > 0.0f, "(Batch element %d): The factorization could not be completed because the input is not positive-definite (the leading minor of order %d is not positive-definite).\n", offsetPrefix, index + 1);
    } else {
        ascendc_assert(b1 > 0.0f, "The factorization could not be completed because the input is not positive-definite (the leading minor of order %d is not positive-definite).\n", index + 1);
    }
    PipeBarrier<PIPE_ALL>();
    b1 = sqrt(b1);
    PipeBarrier<PIPE_ALL>();
    Muls(matLLocal, matLLocal, T(1/b1), matSizeN - index);
    PipeBarrier<PIPE_ALL>();
    DataCopyParams dataCopyOutParams {1, static_cast<uint16_t>(sizeof(T) * (matSizeN - index)), 0, 0};
    DataCopyPad(outGM[matrixoffset + index * matSizeN + index], matLLocal, dataCopyOutParams);
    PipeBarrier<PIPE_ALL>();
}

}
#endif