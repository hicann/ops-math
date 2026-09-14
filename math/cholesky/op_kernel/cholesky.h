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
#if __NPU_ARCH__ == 3510
#include "simt_api/common_functions.h"
#endif

using namespace AscendC;

namespace Cholesky {
// 单缓冲：当前流水为同步串行（搬运/计算逐级等待），双缓冲无重叠收益，仅保留 1 份 UB 分配
constexpr uint32_t BUFFER_NUM = 1;
// [general] 32B 对齐为 Ascend 通用约束
constexpr uint32_t BASIC_BLOCK = 32;
// 正定性检查上界：拦截 +inf（inf > 0 为真，需显式排除）；NaN 与任何比较均为假，天然被拦截
constexpr float MAX_FINITE_FLOAT = 3.402823466e+38f;
// 单矩阵面板宽度（列/行面板）：右看面板算法的串行段宽度。8 的倍数保证面板行 32B 对齐；
// tiling 侧以相同阈值决定单矩阵是否启用多核
constexpr uint32_t PANEL_WIDTH = 64;
// 批量协同阈值：矩阵维度超过该值时批量模式改为全核逐矩阵协同面板（否则保持每核一矩阵）。
// 96 使 m127 批量进协同路径（旧每核一矩阵路径 upper 2438us/11.5x，协同约 480us），
// m63/m64 批量旧路径健康（ratio 0.6~1.1x）不受影响
constexpr uint32_t BATCH_COOP_MIN_M = 96;
// 批量模式点积 2D 搬运分块缓冲上限（与 InitBuffer 的 matRightQueue_ 容量一致）
constexpr uint64_t CHUNK_BUFFER_BYTES = 32UL * 1024UL;
#if __NPU_ARCH__ == 3510
// [RegBase-native] RegTensor 向量长度 256B，fp32 为 64 元素（DAV_3510）
constexpr uint32_t FP32_VECTOR_LENGTH = 64;
// [RegBase-native] SIMT __launch_bounds__ 线程数约束，与 asc_vf_call(dim3(...)) 一致（DAV_3510）
constexpr uint32_t DIAGONAL_SCAN_THREADS = 512;

// [RegBase-native] SIMT 路径（__simt_vf__ + asc_vf_call(dim3)），仅 DAV_3510 生效
__simt_vf__ __aicore__ __launch_bounds__(DIAGONAL_SCAN_THREADS) inline void CholeskyDiagonalVf(uint64_t elementCount,
                                                                                               uint32_t matrixSize,
                                                                                               __gm__ float* input,
                                                                                               __gm__ float* output)
{
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elementCount;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        output[index] = index / matrixSize == index % matrixSize ? sqrt(input[index]) : 0.0f;
    }
}

// [RegBase-native] 原地转置输出：上三角值搬移到下三角（L = U^T），上三角清零（对齐
// 旧 tril 路径只写下三角、golden 上三角为零的输出契约）。单调用者执行；
// 线程循环采用 histogram_v2 GmCast 同构模式（threadIdx 起、blockDim 跨步）
__simt_vf__ __aicore__ __launch_bounds__(DIAGONAL_SCAN_THREADS) inline void CholeskyTransposeSwapVf(
    uint64_t elementCount, uint32_t matrixSize, __gm__ float* mat)
{
    for (uint64_t index = threadIdx.x; index < elementCount; index += blockDim.x) {
        uint64_t row = index / matrixSize;
        uint64_t col = index % matrixSize;
        if (row < col) {
            mat[col * matrixSize + row] = mat[index];
            mat[index] = 0.0f;
        }
    }
}

// [RegBase-native] SIMD-RegBase 路径（__simd_vf__ + Reg::*），仅 DAV_3510 生效
__simd_vf__ inline void ScaleCholeskyVf(__ubuf__ float* dst, float scale, uint32_t count)
{
    uint16_t repeat = static_cast<uint16_t>((count + FP32_VECTOR_LENGTH - 1) / FP32_VECTOR_LENGTH);
    for (uint16_t i = 0; i < repeat; ++i) {
        uint32_t remain = count - static_cast<uint32_t>(i) * FP32_VECTOR_LENGTH;
        auto mask = Reg::UpdateMask<float>(remain);
        auto address = Reg::CreateAddrReg<float>(i, FP32_VECTOR_LENGTH);
        Reg::RegTensor<float> value;
        Reg::LoadAlign(value, dst, address);
        Reg::Muls(value, value, scale, mask);
        Reg::StoreAlign(dst, value, address, mask);
    }
}
#endif

__aicore__ inline void ScaleCholesky(LocalTensor<float>& dst, float scale, uint32_t count)
{
#if __NPU_ARCH__ == 3510
    if (count == 1) {
        asc_vf_call<ScaleCholeskyVf>(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), scale, count);
    } else {
        Muls(dst, dst, scale, count);
    }
#else
    Muls(dst, dst, scale, count);
#endif
}

template <typename T>
class Cholesky {
public:
    __aicore__ inline Cholesky(){};
    __aicore__ inline void InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData,
                                    TPipe* pipe);
    __aicore__ inline void InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData,
                                    TPipe* pipe);
    __aicore__ inline void ProcessTril();
    __aicore__ inline void ProcessTriu();

private:
    __aicore__ inline void SyncPipeVToS();
    __aicore__ inline void SyncPipeMte2ToV();
    __aicore__ inline void SyncPipeMte2ToS();
    __aicore__ inline void SyncPipeMte3ToS();
    __aicore__ inline void SyncPipeVToMte3();
    __aicore__ inline void GetTilingData(const CholeskyTilingData* tilingData);
    __aicore__ inline void FirstColumn(uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void SecondToNColumn(uint32_t index, uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void FirstRow(uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void SecondToNRow(uint32_t index, uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void FactorPanelTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel);
    __aicore__ inline void UpdateTrailingTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel);
#if __NPU_ARCH__ == 3510
    __aicore__ inline void RunUpperPanels();
    __aicore__ inline void TransposeOutputInPlace();
#endif

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return b == 0 ? a : (a + b - 1) / b;
    }

    __aicore__ inline uint32_t AlignToSlot(uint32_t elements)
    {
        constexpr uint32_t SLOT = BASIC_BLOCK / sizeof(float);
        return (elements + SLOT - 1) / SLOT * SLOT;
    }

private:
    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t matSizeN_ = 0;
    uint64_t matrixNumCount_ = 0;
    uint32_t blockSize_ = 0;
    // 协同模式：单矩阵，或维度超过 BATCH_COOP_MIN_M 的批量（全核逐矩阵面板算法）
    bool cooperative_ = false;
    __gm__ T* selfBase_ = nullptr;
    __gm__ T* outBase_ = nullptr;
    T invSqrtA11_ = 0.0f; // 存储首主元缩放因子，避免重复计算和直接访问GM内存
    T zero_ = 0.0f;
    T one_ = 1.0f;

    TQue<QuePosition::VECIN, BUFFER_NUM> matAQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matLeftQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matRightQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matResultQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> matLQueue_;

    GlobalTensor<T> matAGM_;
    GlobalTensor<T> outGM_;
    GlobalTensor<T> workspaceFlagGM_;

    // 辅助函数声明
    __aicore__ inline void ProcessColumnDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                   LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                   uint32_t index, uint64_t offset, uint32_t blockStart,
                                                   uint32_t count);

    __aicore__ inline void ProcessRowDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                uint32_t index, uint64_t offset, uint32_t blockStart, uint32_t count);

    __aicore__ inline T ComputeScaleFactor(const LocalTensor<T>& matLLocal, uint64_t offsetPrefix, uint32_t index);
    __aicore__ inline void SyncSingleMatrix();
    __aicore__ inline bool ProcessDiagonalMatrix();
};

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeVToS()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeMte2ToV()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeMte2ToS()
{
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeMte3ToS()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeVToMte3()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void Cholesky<T>::GetTilingData(const CholeskyTilingData* tilingData)
{
    matSizeN_ = tilingData->matSizeN;
    matrixNumCount_ = tilingData->matrixNumCount;
    blockSize_ = tilingData->blockSize;
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncSingleMatrix()
{
    // 原版语义：仅单矩阵（逐列路径）跨核同步；批量每核独立禁止 SyncAll（到达数不匹配会死锁）
    if (matrixNumCount_ == 1) {
        AscendC::SyncAll();
    }
}

template <typename T>
__aicore__ inline bool Cholesky<T>::ProcessDiagonalMatrix()
{
#if __NPU_ARCH__ == 3510
    if (matrixNumCount_ != 1) {
        return false;
    }
    bool localDiagonal = true;
    LocalTensor<T> scanLocal = matAQueue_.AllocTensor<T>();
    for (uint32_t row = blockIdx_; row < matSizeN_ && localDiagonal; row += blockDim_) {
        for (uint32_t column = 0; column < matSizeN_ && localDiagonal; column += blockSize_) {
            uint32_t count = matSizeN_ - column > blockSize_ ? blockSize_ : matSizeN_ - column;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(scanLocal, matAGM_[static_cast<uint64_t>(row) * matSizeN_ + column], copyParams, padParams);
            SyncPipeMte2ToS();
            for (uint32_t offset = 0; offset < count; ++offset) {
                if (column + offset != row && scanLocal.GetValue(offset) != zero_) {
                    localDiagonal = false;
                    break;
                }
            }
        }
    }
    matAQueue_.FreeTensor(scanLocal);

    // 跨核投票：本核判定结果经 Duplicate 写 UB 后单元素 DataCopyPad 写出到 GM 自身槽位，
    // SyncAll 前等待 MTE3 写出完成，保证其余核 SyncAll 后可见
    LocalTensor<T> flagLocal = matAQueue_.AllocTensor<T>();
    Duplicate(flagLocal, localDiagonal ? zero_ : one_, 1);
    SyncPipeVToMte3();
    DataCopyExtParams flagCopyParams{1, sizeof(T), 0, 0, 0};
    DataCopyPad(workspaceFlagGM_[blockIdx_], flagLocal, flagCopyParams);
    SyncPipeMte3ToS();
    SyncSingleMatrix();

    // 读票：每票单块搬入后标量读取（与首主元搬运同模式）。不能按 blockDim_ 个
    // sizeof(T) 块做一次 2D 搬入：blockLen < 32B 时 UB 侧各块按 32B 对齐间距落位，
    // 连续 GetValue(core) 会读到槽间残留数据，导致跨核投票误判
    DataCopyExtParams oneVoteParams{1, sizeof(T), 0, 0, 0};
    DataCopyPadExtParams<T> oneVotePadParams{false, 0, 0, 0};
    bool allDiagonal = true;
    for (uint32_t core = 0; core < blockDim_; ++core) {
        DataCopyPad(flagLocal, workspaceFlagGM_[core], oneVoteParams, oneVotePadParams);
        SyncPipeMte2ToS();
        if (flagLocal.GetValue(0) != zero_) {
            allDiagonal = false;
            break;
        }
    }
    matAQueue_.FreeTensor(flagLocal);
    if (!allDiagonal) {
        return false;
    }
    const uint64_t elementCount = static_cast<uint64_t>(matSizeN_) * matSizeN_;
    asc_vf_call<CholeskyDiagonalVf>(dim3(DIAGONAL_SCAN_THREADS), elementCount, matSizeN_,
                                    (__gm__ float*)matAGM_.GetPhyAddr(), (__gm__ float*)outGM_.GetPhyAddr());
    return true;
#else
    return false;
#endif
}

template <typename T>
__aicore__ inline void Cholesky<T>::InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                             const CholeskyTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    GetTilingData(tilingData);
    cooperative_ = matrixNumCount_ == 1 || matSizeN_ > BATCH_COOP_MIN_M;
    selfBase_ = (__gm__ T*)self;
    outBase_ = (__gm__ T*)out;

    matAGM_.SetGlobalBuffer(selfBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    outGM_.SetGlobalBuffer(outBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    workspaceFlagGM_.SetGlobalBuffer((__gm__ T*)workspace, blockDim_);

#if __NPU_ARCH__ == 3510
    if (cooperative_) {
        // 协同面板路径（仅 3510）：Schur 补驻留 outGM 尾部区域，2D 面板块需要
        // 更大搬运缓冲。matLeft 需容纳面板 slotted 主元拷贝（PANEL_WIDTH × slot 元素）。
        // 非 3510 走原版逐列路径，保持原版批量缓冲布局（matLeft/matResult = columnBufferSize）
        pipe->InitBuffer(matAQueue_, BUFFER_NUM, static_cast<uint64_t>(blockSize_) * BASIC_BLOCK);
        pipe->InitBuffer(matLQueue_, BUFFER_NUM, static_cast<uint64_t>(PANEL_WIDTH) * BASIC_BLOCK);
        pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, static_cast<uint64_t>(PANEL_WIDTH) * BASIC_BLOCK);
        pipe->InitBuffer(matRightQueue_, BUFFER_NUM, static_cast<uint64_t>(blockSize_) * PANEL_WIDTH * sizeof(T));
        pipe->InitBuffer(matResultQueue_, BUFFER_NUM, static_cast<uint64_t>(blockSize_) * sizeof(T));
        return;
    }
#endif

    // 使用分块大小计算buffer，减少UB内存使用
    uint64_t columnBufferSize = blockSize_ * BASIC_BLOCK;
    uint64_t rowBufferSize = CeilDiv(blockSize_ * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK;

    pipe->InitBuffer(matAQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matRightQueue_, BUFFER_NUM, CHUNK_BUFFER_BYTES);
    pipe->InitBuffer(matResultQueue_, BUFFER_NUM, rowBufferSize);
}

// 协同模式：对 base 偏移处的单个矩阵执行上三角右看面板算法
// 协同模式：对 base 偏移处的单个矩阵执行上三角右看面板算法（仅 3510）。
// 面板间同步无条件 SyncAll：调用方保证所有核执行同一矩阵序列（单矩阵或协同批量）
#if __NPU_ARCH__ == 3510
template <typename T>
__aicore__ inline void Cholesky<T>::RunUpperPanels()
{
    for (uint32_t panelStart = 0; panelStart < matSizeN_; panelStart += PANEL_WIDTH) {
        const uint32_t panelEnd = panelStart + PANEL_WIDTH < matSizeN_ ? panelStart + PANEL_WIDTH : matSizeN_;
        const bool firstPanel = panelStart == 0;
        if (blockIdx_ == (panelStart / PANEL_WIDTH) % blockDim_) {
            FactorPanelTriu(panelStart, panelEnd, firstPanel);
        }
        AscendC::SyncAll();
        if (panelEnd < matSizeN_) {
            UpdateTrailingTriu(panelStart, panelEnd, firstPanel);
            AscendC::SyncAll();
        }
    }
}
#endif

// 上三角结果原位转置（搬移 L=U^T + 上三角清零）：单调用者执行，作用于 outGM_ 当前矩阵区域（仅 3510）
#if __NPU_ARCH__ == 3510
template <typename T>
__aicore__ inline void Cholesky<T>::TransposeOutputInPlace()
{
    if (blockIdx_ == 0) {
        const uint64_t elementCount = static_cast<uint64_t>(matSizeN_) * matSizeN_;
        asc_vf_call<CholeskyTransposeSwapVf>(dim3(DIAGONAL_SCAN_THREADS), elementCount, matSizeN_,
                                             (__gm__ float*)outGM_.GetPhyAddr());
        // asc_vf_call 为异步发射：显式等待 SIMT VF 计算完成再进入全核同步（histogram_v2 先例）
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }
    AscendC::SyncAll();
}
#endif

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTril()
{
    if (ProcessDiagonalMatrix()) {
        return;
    }
#if __NPU_ARCH__ == 3510
    if (cooperative_) {
        // 协同模式（单矩阵或大维度批量）：下三角 = 上三角分解结果的转置（A 对称，SPD 契约
        // 保证）。上三角面板算法已验证 6.7x，转置经 SIMT 成对交换完成，免去逐行点积慢路径
        for (uint64_t m = 0; m < matrixNumCount_; ++m) {
            const uint64_t offset = m * matSizeN_ * matSizeN_;
            matAGM_.SetGlobalBuffer(selfBase_ + offset, matSizeN_ * matSizeN_);
            outGM_.SetGlobalBuffer(outBase_ + offset, matSizeN_ * matSizeN_);
            RunUpperPanels();
            TransposeOutputInPlace();
        }
        return;
    }
#endif
    // 非 3510 / 非协同批量：原版每核一矩阵（单矩阵 offsetPrefix=0），单矩阵逐列 SyncAll
    if (blockIdx_ < blockDim_) {
        auto loopTimes = matrixNumCount_ == 1 ? 0 : matrixNumCount_ / blockDim_;
        for (uint64_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint64_t offsetPrefix = matrixNumCount_ == 1 ? 0 : blockIdx_ + blockDim_ * loopIndex;
            if (offsetPrefix < matrixNumCount_) {
                uint64_t offset = offsetPrefix * matSizeN_ * matSizeN_;
                FirstColumn(offsetPrefix, offset);
                SyncSingleMatrix();
                for (uint32_t index = 1; index < matSizeN_; index++) {
                    SecondToNColumn(index, offsetPrefix, offset);
                    SyncSingleMatrix();
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstColumn(uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    if (matrixNumCount_ == 1) {
        // 单元素 DataCopyPad 搬入首主元后经 MTE2_S 同步再标量读取，避免逐元素 GM 访问
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToS();
        T a11 = matALocal.GetValue(0);
        ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                       "The factorization could not be completed because the input is not positive-definite "
                       "(the leading minor of order 1 is not positive-definite).\n");
        invSqrtA11_ = T(1 / sqrt(a11));
    }

    // 核内分块处理，每次处理blockSize大小的数据
    for (uint32_t blockStart = 0; blockStart < matSizeN_; blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        uint32_t count = (matSizeN_ - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - blockStart);

        DataCopyExtParams copyParamsMatALocal{static_cast<uint16_t>(count), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matALocal, matAGM_[offset + blockStart * matSizeN_], copyParamsMatALocal, padParamsMatALocal);
        SyncPipeMte2ToV();

        // 只在处理第一个元素时计算平方根并存储缩放因子
        if (blockStart == 0 && matrixNumCount_ > 1) {
            SyncPipeMte2ToS();
            T a11 = matALocal.GetValue(0);
            ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                           "(Batch element %llu): The factorization could not be completed because the input is not "
                           "positive-definite (the leading minor of order 1 is not positive-definite).\n",
                           offsetPrefix);
            invSqrtA11_ = T(1 / sqrt(a11));
        }
        ScaleCholesky(matALocal, static_cast<float>(invSqrtA11_), count * BASIC_BLOCK / sizeof(T));

        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{static_cast<uint16_t>(count), sizeof(T), 0,
                                            static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0};
        DataCopyPad(outGM_[offset + blockStart * matSizeN_], matALocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    matAQueue_.FreeTensor(matALocal);
}

// 辅助函数：批量模式列点积（行块 2D 分块搬运 + 批量同步，消除逐行 MTE2→V/V→S 同步）
template <typename T>
__aicore__ inline void Cholesky<T>::ProcessColumnDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                            LocalTensor<T>& matRightLocal,
                                                            LocalTensor<T>& matResultLocal, uint32_t index,
                                                            uint64_t offset, uint32_t blockStart, uint32_t count)
{
    const uint32_t slot = BASIC_BLOCK / sizeof(T);
    const uint32_t leftBlockNum = (index + blockSize_ - 1) / blockSize_;

    for (uint32_t leftBlockIdx = 0; leftBlockIdx < leftBlockNum; leftBlockIdx++) {
        const uint32_t leftBlockStart = leftBlockIdx * blockSize_;
        const uint32_t leftBlockSize = (index - leftBlockStart) > blockSize_ ? blockSize_ : (index - leftBlockStart);
        const uint32_t leftPad = AlignToSlot(leftBlockSize);

        // 主元行段 L[index, leftBlockStart..+leftBlockSize)
        DataCopyExtParams copyLeft{1, static_cast<uint32_t>(leftBlockSize * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padLeft{false, 0, 0, 0};
        DataCopyPad(matLeftLocal, outGM_[offset + index * matSizeN_ + leftBlockStart], copyLeft, padLeft);

        uint32_t chunkRows = static_cast<uint32_t>(CHUNK_BUFFER_BYTES / (leftPad * sizeof(T)));
        if (chunkRows == 0) {
            chunkRows = 1;
        }
        for (uint32_t rowBase = 0; rowBase < count; rowBase += chunkRows) {
            const uint32_t rows = count - rowBase > chunkRows ? chunkRows : count - rowBase;
            // 行块 [blockStart+rowBase, +rows) × [leftBlockStart, +leftBlockSize)：一次 2D 搬入
            // （padTail 扩展行距，dstStride=0，禁止叠加）
            DataCopyExtParams copyRight{static_cast<uint16_t>(rows), static_cast<uint32_t>(leftBlockSize * sizeof(T)),
                                        static_cast<uint32_t>((matSizeN_ - leftBlockSize) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(leftPad - leftBlockSize), 0};
            DataCopyPad(matRightLocal, outGM_[offset + (index + blockStart + rowBase) * matSizeN_ + leftBlockStart],
                        copyRight, padRight);
            SyncPipeMte2ToV();
            for (uint32_t r = 0; r < rows; ++r) {
                Mul(matResultLocal, matLeftLocal, matRightLocal[r * leftPad], leftBlockSize);
                ReduceSum<T>(matResultLocal, matResultLocal, matRightLocal[r * leftPad], leftBlockSize);
                Add(matLLocal[(rowBase + r) * slot], matLLocal[(rowBase + r) * slot], matResultLocal, slot);
            }
        }
    }
}

// 辅助函数：计算缩放因子并进行正定性检查
template <typename T>
__aicore__ inline T Cholesky<T>::ComputeScaleFactor(const LocalTensor<T>& matLLocal, uint64_t offsetPrefix,
                                                    uint32_t index)
{
    // 调用方 Sub（V 写）与本次标量读取之间同步
    SyncPipeVToS();
    T b1 = matLLocal.GetValue(0);
    if (matrixNumCount_ > 1) {
        ascendc_assert(b1 > 0.0f && b1 <= MAX_FINITE_FLOAT,
                       "(Batch element %llu): The factorization could not be completed because the input is not "
                       "positive-definite (the leading minor of order %u is not positive-definite).\n",
                       offsetPrefix, index + 1);
    } else {
        ascendc_assert(b1 > 0.0f && b1 <= MAX_FINITE_FLOAT,
                       "The factorization could not be completed because the input is not positive-definite (the "
                       "leading minor of order %u is not positive-definite).\n",
                       index + 1);
    }

    // 计算缩放因子
    return T(1 / sqrt(b1));
}

// 辅助函数：批量模式行点积。保持逐列 Mul+ReduceSum 树归约的原始数值序（Add 链在病态
// 输入下累积舍入会使主元变负触发正定性断言），仅将同步从 PipeBarrier 收敛为事件同步
template <typename T>
__aicore__ inline void Cholesky<T>::ProcessRowDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                         LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                         uint32_t index, uint64_t offset, uint32_t blockStart,
                                                         uint32_t count)
{
    const uint32_t leftBlockNum = (index + blockSize_ - 1) / blockSize_;

    for (uint32_t leftBlockIdx = 0; leftBlockIdx < leftBlockNum; leftBlockIdx++) {
        const uint32_t leftBlockStart = leftBlockIdx * blockSize_;
        const uint32_t leftBlockSize = (index - leftBlockStart) > blockSize_ ? blockSize_ : (index - leftBlockStart);

        // 搬运当前块的matLeftLocal数据（槽位布局，padding×padding=0 保证槽位乘积可整体归约）
        DataCopyExtParams copyParamsLeftLocal{static_cast<uint16_t>(leftBlockSize), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsLeftLocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matLeftLocal, outGM_[offset + index + leftBlockStart * matSizeN_], copyParamsLeftLocal,
                    padParamsLeftLocal);

        // 对当前块中的每一列，分块搬运matRightLocal并计算
        for (uint32_t colInBlock = 0; colInBlock < count; colInBlock++) {
            uint32_t columnRightPivot = blockStart + colInBlock;

            // 搬运当前列的matRightLocal数据
            DataCopyPad(matRightLocal, outGM_[offset + (index + columnRightPivot) + leftBlockStart * matSizeN_],
                        copyParamsLeftLocal, padParamsLeftLocal);
            SyncPipeMte2ToV();

            Mul(matResultLocal, matLeftLocal, matRightLocal, leftBlockSize * BASIC_BLOCK / sizeof(T));
            ReduceSum<T>(matResultLocal, matResultLocal, matRightLocal, leftBlockSize * BASIC_BLOCK / sizeof(T));
            // V 写与 S 读之间同步：先等向量归约完成，再标量读取累加
            SyncPipeVToS();
            T currentSum = matResultLocal.GetValue(0);
            T existingSum = matLLocal.GetValue(colInBlock);
            matLLocal.SetValue(colInBlock, existingSum + currentSum);
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNColumn(uint32_t index, uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matLLocal = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matLeftLocal = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matRightLocal = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matResultLocal = matResultQueue_.AllocTensor<T>();

    // 存储当前列的缩放因子，所有分块共享同一个缩放因子
    T columnScaleFactor = 0.0f;
    if (matrixNumCount_ == 1) {
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToV();
        Duplicate(matLLocal, zero_, BASIC_BLOCK / sizeof(T));
        ProcessColumnDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, 0, 1);
        Sub(matLLocal, matALocal, matLLocal, BASIC_BLOCK / sizeof(T));
        columnScaleFactor = ComputeScaleFactor(matLLocal, offsetPrefix, index);
    }

    // 对当前列的所有元素进行分块处理
    for (uint32_t blockStart = 0; blockStart < (matSizeN_ - index); blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        // 计算当前块的大小
        uint32_t count = (matSizeN_ - index - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - index - blockStart);

        // 搬运当前块的count个A元素
        DataCopyExtParams copyParamsMatALocal{static_cast<uint16_t>(count), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index + blockStart * matSizeN_],
                    copyParamsMatALocal, padParamsMatALocal);
        SyncPipeMte2ToV();

        // 初始化当前块的L结果为0
        Duplicate(matLLocal, zero_, count * BASIC_BLOCK / sizeof(T));

        // 调用辅助函数处理点积计算
        ProcessColumnDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, blockStart,
                                count);

        // 执行计算操作
        Sub(matLLocal, matALocal, matLLocal, count * BASIC_BLOCK / sizeof(T));

        // 只在第一次分块时计算缩放因子和进行正定性检查
        if (blockStart == 0 && matrixNumCount_ > 1) {
            columnScaleFactor = ComputeScaleFactor(matLLocal, offsetPrefix, index);
        }

        // 对当前块的所有元素应用同一个缩放因子
        ScaleCholesky(matLLocal, static_cast<float>(columnScaleFactor), count * BASIC_BLOCK / sizeof(T));

        // 得到count个L元素并搬出
        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{static_cast<uint16_t>(count), sizeof(T), 0,
                                            static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0};
        DataCopyPad(outGM_[offset + index * matSizeN_ + index + blockStart * matSizeN_], matLLocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    // 释放张量资源
    matResultQueue_.FreeTensor(matResultLocal);
    matRightQueue_.FreeTensor(matRightLocal);
    matLeftQueue_.FreeTensor(matLeftLocal);
    matLQueue_.FreeTensor(matLLocal);
    matAQueue_.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                             const CholeskyTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    GetTilingData(tilingData);
    cooperative_ = matrixNumCount_ == 1 || matSizeN_ > BATCH_COOP_MIN_M;
    selfBase_ = (__gm__ T*)self;
    outBase_ = (__gm__ T*)out;

    matAGM_.SetGlobalBuffer(selfBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    outGM_.SetGlobalBuffer(outBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    workspaceFlagGM_.SetGlobalBuffer((__gm__ T*)workspace, blockDim_);

#if __NPU_ARCH__ == 3510
    if (cooperative_) {
        // 协同面板路径（仅 3510）：与 InitTril 相同的面板缓冲布局（tril/triu 仅
        // GM 索引不同）。matLeft 需容纳面板 slotted 主元拷贝（PANEL_WIDTH × slot 元素）。
        // 非 3510 走原版逐列路径，保持原版批量缓冲布局
        pipe->InitBuffer(matAQueue_, BUFFER_NUM, static_cast<uint64_t>(blockSize_) * BASIC_BLOCK);
        pipe->InitBuffer(matLQueue_, BUFFER_NUM, static_cast<uint64_t>(PANEL_WIDTH) * BASIC_BLOCK);
        pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, static_cast<uint64_t>(PANEL_WIDTH) * BASIC_BLOCK);
        pipe->InitBuffer(matRightQueue_, BUFFER_NUM, static_cast<uint64_t>(blockSize_) * PANEL_WIDTH * sizeof(T));
        pipe->InitBuffer(matResultQueue_, BUFFER_NUM, static_cast<uint64_t>(blockSize_) * sizeof(T));
        return;
    }
#endif

    // 使用分块大小计算buffer，减少UB内存使用。
    // matLeft/matResult 需容纳逐列槽位乘加（leftBlockSize × slot 元素，原版行点积数值序）
    uint32_t columnBufferSize = blockSize_ * BASIC_BLOCK;
    uint32_t rowBufferSize = CeilDiv(blockSize_ * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK;

    pipe->InitBuffer(matAQueue_, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLQueue_, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matRightQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matResultQueue_, BUFFER_NUM, columnBufferSize);
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTriu()
{
    if (ProcessDiagonalMatrix()) {
        return;
    }
#if __NPU_ARCH__ == 3510
    if (cooperative_) {
        // 协同模式（单矩阵或大维度批量）：上三角右看面板算法，面板间全核并行更新 Schur 补，
        // SyncAll 次数从逐列 M 次降为逐面板 2*ceil(M/PANEL_WIDTH) 次；批量逐矩阵串行推进
        for (uint64_t m = 0; m < matrixNumCount_; ++m) {
            const uint64_t offset = m * matSizeN_ * matSizeN_;
            matAGM_.SetGlobalBuffer(selfBase_ + offset, matSizeN_ * matSizeN_);
            outGM_.SetGlobalBuffer(outBase_ + offset, matSizeN_ * matSizeN_);
            RunUpperPanels();
        }
        return;
    }
#endif
    // 非 3510 / 非协同批量：原版每核一矩阵（单矩阵 offsetPrefix=0），单矩阵逐列 SyncAll
    if (blockIdx_ < blockDim_) {
        auto loopTimes = matrixNumCount_ == 1 ? 0 : matrixNumCount_ / blockDim_;
        for (uint64_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint64_t offsetPrefix = matrixNumCount_ == 1 ? 0 : blockIdx_ + blockDim_ * loopIndex;
            if (offsetPrefix < matrixNumCount_) {
                uint64_t offset = offsetPrefix * matSizeN_ * matSizeN_;
                FirstRow(offsetPrefix, offset);
                SyncSingleMatrix();
                for (uint32_t index = 1; index < matSizeN_; index++) {
                    SecondToNRow(index, offsetPrefix, offset);
                    SyncSingleMatrix();
                }
            }
        }
    }
}

// 单矩阵面板分解（上三角）：面板行 [panelStart, panelEnd) 由属主核串行分解，
// 行 j 的点积仅覆盖面板内列 [panelStart, j)，按 k 广播乘加（连续向量运算）
template <typename T>
__aicore__ inline void Cholesky<T>::FactorPanelTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel)
{
    const GlobalTensor<T>& srcGM = firstPanel ? matAGM_ : outGM_;
    LocalTensor<T> matPivot = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matA = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matRight = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matTmp = matResultQueue_.AllocTensor<T>();
    const uint32_t slot = BASIC_BLOCK / sizeof(T);

    for (uint32_t j = panelStart; j < panelEnd; ++j) {
        const uint32_t width = j - panelStart;
        if (width > 0) {
            // 主元列 out[panelStart..j, j) → 槽位（S 侧逐 k 标量读取）
            DataCopyExtParams pivotCopy{static_cast<uint16_t>(width), sizeof(T),
                                        static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> pivotPad{true, 0, static_cast<uint8_t>(slot - 1), 0};
            DataCopyPad(matPivot, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j], pivotCopy, pivotPad);
            SyncPipeMte2ToS();
        }
        T scaleFactor = one_;
        for (uint32_t colStart = j; colStart < matSizeN_; colStart += blockSize_) {
            const uint32_t cols = matSizeN_ - colStart > blockSize_ ? blockSize_ : matSizeN_ - colStart;
            const uint32_t colsPad = AlignToSlot(cols);
            // A 行段（连续）→ matA
            DataCopyExtParams copyA{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padA{false, 0, 0, 0};
            DataCopyPad(matA, srcGM[static_cast<uint64_t>(j) * matSizeN_ + colStart], copyA, padA);
            if (width > 0) {
                // out[panelStart..j, colStart..+cols)：k 行 × cols 列 2D 搬入（padTail 扩展行距，dstStride=0）
                DataCopyExtParams copyRight{static_cast<uint16_t>(width), static_cast<uint32_t>(cols * sizeof(T)),
                                            static_cast<uint32_t>((matSizeN_ - cols) * sizeof(T)), 0, 0};
                DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(colsPad - cols), 0};
                DataCopyPad(matRight, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + colStart], copyRight,
                            padRight);
                SyncPipeMte2ToV();
                Duplicate(matTmp, zero_, cols);
                for (uint32_t k = 0; k < width; ++k) {
                    // 主元列拷贝为 4B 块 × padTail 的 2D 布局，每块落位 32B 槽距，需按 slot 距读取
                    T pivotValue = matPivot.GetValue(k * slot);
                    Muls(matRight[k * colsPad], matRight[k * colsPad], static_cast<float>(pivotValue), cols);
                    Add(matTmp, matTmp, matRight[k * colsPad], cols);
                }
                Sub(matA, matA, matTmp, cols);
            } else {
                SyncPipeMte2ToV();
            }
            if (colStart == j) {
                if (width > 0) {
                    SyncPipeVToS();
                } else {
                    SyncPipeMte2ToS();
                }
                T pivot = matA.GetValue(0);
                ascendc_assert(pivot > 0.0f && pivot <= MAX_FINITE_FLOAT,
                               "The factorization could not be completed because the input is not positive-definite "
                               "(the leading minor of order %u is not positive-definite).\n",
                               j + 1);
                scaleFactor = T(1 / sqrt(pivot));
            }
            Muls(matA, matA, static_cast<float>(scaleFactor), cols);
            SyncPipeVToMte3();
            DataCopyExtParams copyOut{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGM_[static_cast<uint64_t>(j) * matSizeN_ + colStart], matA, copyOut);
            SyncPipeMte3ToS();
        }
    }

    matResultQueue_.FreeTensor(matTmp);
    matRightQueue_.FreeTensor(matRight);
    matAQueue_.FreeTensor(matA);
    matLQueue_.FreeTensor(matPivot);
}

// 单矩阵面板尾随更新（上三角镜像）：全核按列轮转并行
// A[i, j] -= Σ_{k∈面板} out[i, k]·out[j, k]（i <= j，i, j >= panelEnd）
template <typename T>
__aicore__ inline void Cholesky<T>::UpdateTrailingTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel)
{
    if (panelEnd >= matSizeN_) {
        return;
    }
    const GlobalTensor<T>& srcGM = firstPanel ? matAGM_ : outGM_;
    LocalTensor<T> matPivot = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matA = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matAcc = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matRight = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matTmp = matResultQueue_.AllocTensor<T>();
    const uint32_t slot = BASIC_BLOCK / sizeof(T);
    const uint32_t width = panelEnd - panelStart;

    // 上三角尾随更新：A[i, j] -= Σ_{k∈面板} U[k, i]·U[k, j]（i <= j，i, j >= panelEnd）。
    // 按行条带并行（行 i 跨核轮转）：A 行段连续读写；U 面板向量取列段（面板行 k × 列 i/j，上三角有效区），
    // 主元列段 slotted 搬入后 S 侧逐 k 标量读取，面板行块 2D 搬入后逐 k 乘减（连续向量运算）
    for (uint64_t i = panelEnd; i < matSizeN_; ++i) {
        if ((i - panelEnd) % blockDim_ != blockIdx_) {
            continue;
        }
        // U 面板列段 U[panelStart..panelEnd, i] → matPivot 槽位
        DataCopyExtParams pivotCopy{static_cast<uint16_t>(width), sizeof(T),
                                    static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pivotPad{true, 0, static_cast<uint8_t>(slot - 1), 0};
        DataCopyPad(matPivot, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + i], pivotCopy, pivotPad);
        SyncPipeMte2ToS();
        for (uint32_t j0 = i; j0 < matSizeN_; j0 += blockSize_) {
            const uint32_t jc = matSizeN_ - j0 > blockSize_ ? blockSize_ : matSizeN_ - j0;
            const uint32_t colsPad = AlignToSlot(jc);
            // Schur A[i, j0..j0+jc)：连续行段 → matA
            DataCopyExtParams copyA{1, static_cast<uint32_t>(jc * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padA{false, 0, 0, 0};
            DataCopyPad(matA, srcGM[i * matSizeN_ + j0], copyA, padA);
            // U 面板行块 U[panelStart..panelEnd, j0..j0+jc) → matRight（k 行 × jc 列，行距 colsPad）
            DataCopyExtParams copyRight{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)),
                                        static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(colsPad - jc), 0};
            DataCopyPad(matRight, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], copyRight, padRight);
            SyncPipeMte2ToV();
            // 先累加后单次减：逐 k 乘积累加（Add 链），最终一次 Sub，降低病态输入的累积舍入
            Duplicate(matAcc, zero_, jc);
            for (uint32_t k = 0; k < width; ++k) {
                T pivotValue = matPivot.GetValue(k * slot);
                Muls(matTmp, matRight[k * colsPad], static_cast<float>(pivotValue), jc);
                Add(matAcc, matAcc, matTmp, jc);
            }
            Sub(matA, matA, matAcc, jc);
            SyncPipeVToMte3();
            DataCopyExtParams copyOut{1, static_cast<uint32_t>(jc * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGM_[i * matSizeN_ + j0], matA, copyOut);
            SyncPipeMte3ToS();
        }
    }

    matResultQueue_.FreeTensor(matTmp);
    matRightQueue_.FreeTensor(matRight);
    matLeftQueue_.FreeTensor(matAcc);
    matAQueue_.FreeTensor(matA);
    matLQueue_.FreeTensor(matPivot);
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstRow(uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    if (matrixNumCount_ == 1) {
        // 单元素 DataCopyPad 搬入首主元后经 MTE2_S 同步再标量读取，避免逐元素 GM 访问
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToS();
        T a11 = matALocal.GetValue(0);
        ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                       "The factorization could not be completed because the input is not positive-definite "
                       "(the leading minor of order 1 is not positive-definite).\n");
        invSqrtA11_ = T(1 / sqrt(a11));
    }

    // 核内分块处理，每次处理blockSize大小的数据
    for (uint32_t blockStart = 0; blockStart < matSizeN_; blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        uint32_t count = (matSizeN_ - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - blockStart);

        DataCopyExtParams copyParamsMatALocal{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + blockStart], copyParamsMatALocal, padParamsMatALocal);
        SyncPipeMte2ToV();

        // 只在处理第一个元素时计算平方根并存储缩放因子
        if (blockStart == 0 && matrixNumCount_ > 1) {
            SyncPipeMte2ToS();
            T a11 = matALocal.GetValue(0);
            ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                           "(Batch element %llu): The factorization could not be completed because the input is not "
                           "positive-definite (the leading minor of order 1 is not positive-definite).\n",
                           offsetPrefix);
            invSqrtA11_ = T(1 / sqrt(a11));
        }
        ScaleCholesky(matALocal, static_cast<float>(invSqrtA11_), count);

        // 搬出当前块的结果
        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPad(outGM_[offset + blockStart], matALocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    matAQueue_.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNRow(uint32_t index, uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matLLocal = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matLeftLocal = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matRightLocal = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matResultLocal = matResultQueue_.AllocTensor<T>();

    // 存储当前行的缩放因子，所有分块共享同一个缩放因子
    T rowScaleFactor = 0.0f;
    if (matrixNumCount_ == 1) {
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToV();
        Duplicate(matLLocal, zero_, BASIC_BLOCK / sizeof(T));
        ProcessRowDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, 0, 1);
        Sub(matLLocal, matALocal, matLLocal, 1);
        rowScaleFactor = ComputeScaleFactor(matLLocal, offsetPrefix, index);
    }

    // 对当前行的所有元素进行分块处理
    for (uint32_t blockStart = 0; blockStart < (matSizeN_ - index); blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        // 计算当前块的大小
        uint32_t count = (matSizeN_ - index - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - index - blockStart);

        // 搬运当前块的count个A元素
        DataCopyExtParams copyParamsMatALocal{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index + blockStart], copyParamsMatALocal,
                    padParamsMatALocal);
        SyncPipeMte2ToV();

        // 初始化当前块的L结果为0
        Duplicate(matLLocal, zero_, count);

        // 调用辅助函数处理点积计算
        ProcessRowDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, blockStart, count);

        // 执行计算操作
        Sub(matLLocal, matALocal, matLLocal, count);

        // 只在第一次分块时计算缩放因子和进行正定性检查
        if (blockStart == 0 && matrixNumCount_ > 1) {
            rowScaleFactor = ComputeScaleFactor(matLLocal, offsetPrefix, index);
        }

        // 对当前块的所有元素应用同一个缩放因子
        ScaleCholesky(matLLocal, static_cast<float>(rowScaleFactor), count);

        // 得到count个L元素并搬出
        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPad(outGM_[offset + index * matSizeN_ + index + blockStart], matLLocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    // 释放张量资源
    matResultQueue_.FreeTensor(matResultLocal);
    matRightQueue_.FreeTensor(matRightLocal);
    matLeftQueue_.FreeTensor(matLeftLocal);
    matLQueue_.FreeTensor(matLLocal);
    matAQueue_.FreeTensor(matALocal);
}

} // namespace Cholesky
#endif
