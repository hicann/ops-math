/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_with_single_axis.h
 * \brief 单轴特化模板：单输入 broadcast kernel (静态Tensor编程 + Mutex同步)
 *
 * 静态Tensor编程:
 *   - 使用LocalTensor构造函数直接指定UB地址, 不通过TPipe/TBuf/TQue
 *   - 使用 Mutex (Lock/Unlock) 同步流水线
 *   - A轴: Double Buffer, Ping/Pong交替, 双MutexID
 *   - B轴: DataCopyPad读标量→Duplicate填充UB→MTE3循环搬出
 */

#ifndef BROADCAST_TO_WITH_SINGLE_AXIS_H_
#define BROADCAST_TO_WITH_SINGLE_AXIS_H_

#include "kernel_operator.h"
#include "broadcast_to_with_single_axis_tiling_data.h"

namespace BrcSA {
using namespace AscendC;

constexpr MutexID MUTEX_ID_0 = 0;
constexpr MutexID MUTEX_ID_1 = 1;

template <typename T, bool IsBrc>
class BroadcastSingleAxis {
public:
    __aicore__ inline BroadcastSingleAxis(){};
    __aicore__ inline ~BroadcastSingleAxis() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ SingleAxisBrcTilingData* tilingDataPtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyScalarAndDup();
    __aicore__ inline void CopyDataIn(int64_t gmOffset, uint32_t curTileSize, LocalTensor<T>& ubBuf);
    __aicore__ inline void CopyDataOut(int64_t gmOffset, uint32_t curTileSize, LocalTensor<T>& ubBuf);
    __aicore__ inline void CalcMCTiling();

private:
    GlobalTensor<T> inGM_;
    GlobalTensor<T> outGM_;

    // 静态Tensor: LocalTensor构造函数直接指定UB地址, 不通过TPipe
    LocalTensor<T> ubPing_;
    LocalTensor<T> ubPong_;
    LocalTensor<T> scalarBuf_; // BRC: 存放从GM读取的标量, 供Duplicate(tensor版)广播

    DataCopyPadExtParams<T> copyPadParams_{false, 0, 0, 0};
    DataCopyExtParams scParams_{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

    int64_t blockIdx_ = 0;
    uint64_t shapeSize_ = 0;
    uint32_t loopNum_ = 0;
    uint32_t tileSize_ = 0;
    uint32_t tileOffset_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t blockFactor_ = 0;
};

// ============================================================
// Implementation
// ============================================================

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::Init(GM_ADDR x, GM_ADDR y,
                                                           __tiling_data_ptr__ SingleAxisBrcTilingData* tilingDataPtr)
{
    AscendC::InitSocState();
    tileSize_ = tilingDataPtr->tileSize;
    shapeSize_ = tilingDataPtr->shapeSize;

    inGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    blockNum_ = tilingDataPtr->blockNum;
    blockFactor_ = tilingDataPtr->blockFactor;

    // tileSize已由tiling保证128B对齐, 故 tileSize * sizeof(T) 天然32B对齐
    if constexpr (IsBrc) {
        // B轴: scalarBuf在DataCopyPad前需要; 其余延迟到CopyScalarAndDup
        scalarBuf_ = LocalTensor<T>(TPosition::VECCALC, tileSize_ * sizeof(T), 32 / sizeof(T));
    } else {
        // A轴: 不走CopyScalarAndDup, LocalTensor在Init分配, 多核切分移到Process
        ubPing_ = LocalTensor<T>(TPosition::VECCALC, 0, tileSize_);
        ubPong_ = LocalTensor<T>(TPosition::VECCALC, tileSize_ * sizeof(T), tileSize_);
    }
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CopyScalarAndDup()
{
    // Mutex同步MTE2→V: DataCopyPad(标量)在MTE2流水线, Duplicate在V流水线
    // Duplicate(tensor版): ubPing_[i] = scalarBuf_[0], 全程V流水线, Mutex可正确同步
    Mutex::Lock<PIPE_MTE2>(MUTEX_ID_0);
    DataCopyPad(scalarBuf_, inGM_, scParams_, copyPadParams_);
    Mutex::Unlock<PIPE_MTE2>(MUTEX_ID_0);

    ubPing_ = LocalTensor<T>(TPosition::VECCALC, 0, tileSize_);
    Mutex::Lock<PIPE_V>(MUTEX_ID_0);
    Duplicate(ubPing_, scalarBuf_, static_cast<int32_t>(tileSize_));
    Mutex::Unlock<PIPE_V>(MUTEX_ID_0);
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CopyDataIn(int64_t gmOffset, uint32_t curTileSize,
                                                                 LocalTensor<T>& ubBuf)
{
    DataCopyExtParams params{1, curTileSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(ubBuf, inGM_[gmOffset], params, copyPadParams_);
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CopyDataOut(int64_t gmOffset, uint32_t curTileSize,
                                                                  LocalTensor<T>& ubBuf)
{
    DataCopyExtParams params{1, curTileSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(outGM_[gmOffset], ubBuf, params);
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CalcMCTiling()
{
    blockIdx_ = GetBlockIdx();
    uint32_t totalTiles = (shapeSize_ + tileSize_ - 1) / tileSize_;
    uint32_t mainCoreBlock = blockFactor_;
    uint32_t mainCoreNum = totalTiles - (blockFactor_ - 1) * blockNum_;
    if (blockIdx_ < mainCoreNum) {
        loopNum_ = mainCoreBlock;
        tileOffset_ = blockIdx_ * mainCoreBlock;
    } else {
        loopNum_ = mainCoreBlock - 1;
        tileOffset_ = mainCoreNum * mainCoreBlock + (blockIdx_ - mainCoreNum) * (mainCoreBlock - 1);
    }
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::Process()
{
    if constexpr (IsBrc) {
        // B轴: DataCopyPad读标量→Duplicate填充UB→MTE3循环搬出
        CopyScalarAndDup();

        // 多核切分: 延迟到CopyScalarAndDup之后, 标量计算与MTE2/V流水重叠
        CalcMCTiling();

        Mutex::Lock<PIPE_MTE3>(MUTEX_ID_0);
        for (uint32_t i = 0; i < loopNum_; i++) {
            int64_t gmOffset = (static_cast<int64_t>(tileOffset_) + i) * tileSize_;
            int64_t remaining = static_cast<int64_t>(shapeSize_) - gmOffset;
            uint32_t curTileSize = (remaining < static_cast<int64_t>(tileSize_)) ? static_cast<uint32_t>(remaining) :
                                                                                   tileSize_;
            CopyDataOut(gmOffset, curTileSize, ubPing_);
        }
        Mutex::Unlock<PIPE_MTE3>(MUTEX_ID_0);
    } else {
        // A轴: Double Buffer, Ping/Pong交替, 双MutexID
        CalcMCTiling();

        for (uint32_t i = 0; i < loopNum_; i++) {
            int64_t gmOffset = (static_cast<int64_t>(tileOffset_) + i) * tileSize_;
            int64_t remaining = static_cast<int64_t>(shapeSize_) - gmOffset;
            uint32_t curTileSize = (remaining < static_cast<int64_t>(tileSize_)) ? static_cast<uint32_t>(remaining) :
                                                                                   tileSize_;

            MutexID curMutex = (i & 1) ? MUTEX_ID_1 : MUTEX_ID_0;
            LocalTensor<T>& curBuf = (i & 1) ? ubPong_ : ubPing_;

            Mutex::Lock<PIPE_MTE2>(curMutex);
            CopyDataIn(gmOffset, curTileSize, curBuf);
            Mutex::Unlock<PIPE_MTE2>(curMutex);

            Mutex::Lock<PIPE_MTE3>(curMutex);
            CopyDataOut(gmOffset, curTileSize, curBuf);
            Mutex::Unlock<PIPE_MTE3>(curMutex);
        }
    }
}

} // namespace BrcSA

#endif // BROADCAST_TO_WITH_SINGLE_AXIS_H_
