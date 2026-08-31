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
 * \file cdist_grad_common.h
 * \brief CdistGrad arch22 CRTP base — TQue double-buffered pipeline + two-phase deterministic reduce
 *
 * All inputs arrive as broadcast [B, P, Q, M] continuous tensors (aclnn UnsqueezeNd + BroadcastTo).
 * Kernel computes gradX1[b,i,:] = sum_j grad[b,i,j] * f(x1[b,i,:], x2[b,j,:], cdist[b,i,j])
 * fully vectorized: grad/cdist are M-wide vectors (scalar repeated along k by broadcast).
 *
 * Derived classes (CdistGradP0/P1/P2/PInf/PGeneral) implement only ComputeForJ(j) and
 * optionally PrepareChunk(currentRTile).
 *
 * Synchronization (AscendC pipeline model):
 *   - Intra-pipe (e.g. Sub -> Mul -> Div -> Add in ComputeForJ): same-pipe FIFO, no sync needed.
 *   - Inter-pipe MTE2->V (chunk CopyIn): TQue<VECIN> EnQue/DeQue.
 *   - Inter-pipe V->MTE3 (CopyOut): TQue<VECOUT> EnQue/DeQue.
 *   - Inter-pipe MTE2->V on a TBuf (x1Row load, Phase2 ws read): SetFlag/WaitFlag<MTE2_V>.
 *
 * Determinism: Q-split path uses two-phase workspace reduce (Phase1 partial sums to workspace
 * slot per sub-task, SyncAll, Phase2 merge in fixed qPart ascending order), NOT atomics.
 */

#ifndef CDIST_GRAD_COMMON_H
#define CDIST_GRAD_COMMON_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../cdist_grad_tiling_data_arch22.h"
#include "../cdist_grad_tiling_key_arch22.h"

namespace NsCdistGrad {

using namespace AscendC;

constexpr int64_t BLOCK_BYTES = 32;

template <typename T, typename Derived>
class CdistGradBase {
public:
    static constexpr bool IS_FP16 = std::is_same_v<T, half>;

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult, GM_ADDR gradX1,
                                GM_ADDR workspace, const CdistGradTilingData* tilingData);
    __aicore__ inline void Process();

protected:
    // 256B (64 fp32) alignment required by Compare — segment granularity.
    static __aicore__ inline int64_t AlignUpSeg(int64_t v) { return ((v + 63) / 64) * 64; }

    // Data views for derived ComputeForJ(j): fp32 rows of the current chunk.
    __aicore__ inline void ComputeChunk(int64_t currentRTile);
    __aicore__ inline void ProcessSubTask(int64_t subIdx);
    __aicore__ inline void CopyInX1Row(int64_t b, int64_t i, int64_t mStart, int64_t mTileReal);
    __aicore__ inline void CopyInChunk(int64_t b, int64_t i, int64_t rStart, int64_t currentRTile, int64_t mStart,
                                       int64_t mTileReal);
    __aicore__ inline void CopyOutPartial(int64_t mTileReal); // Phase1: accum -> ws slot segment
    __aicore__ inline void CopyOutAccum(int64_t mTileReal);   // direct write gradX1 segment
    __aicore__ inline void MergeToGradX1();                   // Phase2: fixed-order merge workspace -> gradX1
    __aicore__ inline void CopyOutRowToGradX1(const LocalTensor<float>& accum, int64_t row, int64_t mStart,
                                              int64_t mTileReal);

    TPipe pipe;
    // Chunk queues (MTE2 -> Vector), double buffered.
    TQue<QuePosition::VECIN, 1> x2Queue;
    TQue<QuePosition::VECIN, 1> gradQueue;
    TQue<QuePosition::VECIN, 1> distQueue;
    // Output queue (Vector -> MTE3) for gradX1 / workspace partial writes.
    TQue<QuePosition::VECOUT, 1> outQueue;
    // Row input queue (MTE2 -> V) for x1Row load (Phase1) and workspace row read (Phase2).
    // Replaces manual SetFlag/WaitFlag<MTE2_V>: manual events conflict with active VECIN
    // queues on the shared event-id pool (see sinkhorn lesson).
    TQue<QuePosition::VECIN, 1> rowInQueue;

    // Fixed compute buffers (fp32).
    TBuf<QuePosition::VECCALC> x1RowBuf;
    TBuf<QuePosition::VECCALC> accumBuf;
    TBuf<QuePosition::VECCALC> diffBuf;
    TBuf<QuePosition::VECCALC> signBuf;
    TBuf<QuePosition::VECCALC> powDstBuf; // Power destination (in-place Power not allowed)
    TBuf<QuePosition::VECCALC> maskBuf;   // chunk-level Compare bit map
    TBuf<QuePosition::VECCALC> maskBuf2;  // per-row Compare bit map
    TBuf<QuePosition::VECCALC> tmpBuf;    // Power temporary
    TBuf<QuePosition::VECCALC> castBuf;   // fp16 cast target
    TBuf<QuePosition::VECCALC> wsReadBuf; // Phase2 workspace row read (type T)

    // Private chunk copies: queue slots are freed right after the copy — the MTE2 refill
    // of a freed slot can never race in-flight Vector reads (which only touch these TBufs).
    TBuf<QuePosition::VECCALC> x2CalcBuf;
    TBuf<QuePosition::VECCALC> gradCalcBuf;
    TBuf<QuePosition::VECCALC> distCalcBuf;

    // Constants.
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> oneBuf;
    TBuf<QuePosition::VECCALC> negOneBuf;

    GlobalTensor<T> gradOutputGM;
    GlobalTensor<T> x1GM;
    GlobalTensor<T> x2GM;
    GlobalTensor<T> cdistResultGM;
    GlobalTensor<T> gradX1GM;
    GlobalTensor<T> wsGM; // two-phase workspace (element type T)

    // Tiling parameters.
    int64_t batchSize_ = 0;
    int64_t pSize_ = 0;
    int64_t rSize_ = 0;
    int64_t mSize_ = 0;
    int64_t mAligned_ = 0;     // CURRENT segment aligned width (all row buffers & vector counts)
    int64_t mAlignedFull_ = 0; // full-row aligned width (workspace slot stride only)
    int64_t mTileSize_ = 0;
    int64_t numMTiles_ = 1;
    int64_t lastMTileSize_ = 0;
    int64_t mStart_ = 0; // current segment start offset within the row
    int64_t rTile_ = 0;
    int64_t numRChunks_ = 0;
    int64_t lastRChunkSize_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t tailCoreTasks_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t qSplit_ = 1;
    int64_t qPartSize_ = 0;
    int64_t tmpBufSize_ = 0;
    float pValueF_ = 2.0f;

    // Runtime state.
    int64_t startSubTask_ = 0;
    int64_t subTaskCount_ = 0;
    bool useTwoPhase_ = false;
    int64_t currentRow_ = 0;    // (b*P+i) for current sub-task
    int64_t currentRTile_ = 0;  // rows in the chunk being computed (debug/derived use)
    int64_t currentSubIdx_ = 0; // global sub-task index
    // fp32 views of the current chunk (set in ComputeChunk).
    LocalTensor<float> x2Chunk_;
    LocalTensor<float> gradChunk_;
    LocalTensor<float> distChunk_;
    LocalTensor<float> x1Row_;
    LocalTensor<float> accum_;
    LocalTensor<float> zero_;
    LocalTensor<float> one_;
    LocalTensor<float> negOne_;
};

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::Init(GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult,
                                                       GM_ADDR gradX1, GM_ADDR workspace,
                                                       const CdistGradTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    pSize_ = tilingData->pSize;
    rSize_ = tilingData->rSize;
    mSize_ = tilingData->mSize;
    mAlignedFull_ = tilingData->mAligned;
    mTileSize_ = tilingData->mTileSize;
    numMTiles_ = tilingData->numMTiles;
    lastMTileSize_ = tilingData->lastMTileSize;
    mAligned_ = AlignUpSeg(mTileSize_); // widest segment (runtime re-set per segment)
    rTile_ = tilingData->rTile;
    numRChunks_ = tilingData->numRChunks;
    lastRChunkSize_ = tilingData->lastRChunkSize;
    tasksPerCore_ = tilingData->tasksPerCore;
    tailCoreTasks_ = tilingData->tailCoreTasks;
    usedCoreNum_ = tilingData->usedCoreNum;
    qSplit_ = tilingData->qSplit;
    qPartSize_ = tilingData->qPartSize;
    tmpBufSize_ = tilingData->tmpBufSize;
    pValueF_ = tilingData->pValueF;

    int64_t blockIdx = AscendC::GetBlockIdx();
    startSubTask_ = blockIdx * tasksPerCore_;
    if (blockIdx < usedCoreNum_ - 1) {
        subTaskCount_ = tasksPerCore_;
    } else {
        subTaskCount_ = tailCoreTasks_;
    }
    if (blockIdx >= usedCoreNum_) {
        subTaskCount_ = 0;
    }
    useTwoPhase_ = (qSplit_ > 1);

    // GM tensors. All inputs are broadcast [B,P,Q,M]; output gradX1 is [B,P,M].
    int64_t pqm = batchSize_ * pSize_ * rSize_ * mSize_;
    gradOutputGM.SetGlobalBuffer((__gm__ T*)gradOutput, pqm);
    x1GM.SetGlobalBuffer((__gm__ T*)x1, pqm);
    x2GM.SetGlobalBuffer((__gm__ T*)x2, pqm);
    cdistResultGM.SetGlobalBuffer((__gm__ T*)cdistResult, pqm);
    gradX1GM.SetGlobalBuffer((__gm__ T*)gradX1, batchSize_ * pSize_ * mSize_);
    if (useTwoPhase_) {
        AscendC::SetSysWorkspace(workspace); // required before GetUserWorkspace
        GM_ADDR ws = AscendC::GetUserWorkspace(workspace);
        int64_t totalSubTasks = batchSize_ * pSize_ * qSplit_;
        wsGM.SetGlobalBuffer((__gm__ T*)ws, totalSubTasks * mAlignedFull_);
    }

    // Chunk queues (double buffered), sized for the WIDEST M segment.
    int64_t chunkBytes = rTile_ * mAligned_ * static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(x2Queue, 2, chunkBytes);
    pipe.InitBuffer(gradQueue, 2, chunkBytes);
    pipe.InitBuffer(distQueue, 2, chunkBytes);
    // Output queue (single buffered): one row of T at a time.
    pipe.InitBuffer(outQueue, 1, mAligned_ * static_cast<int64_t>(sizeof(T)));

    // Fixed compute buffers.
    int64_t mBytes = mAligned_ * static_cast<int64_t>(sizeof(float));
    pipe.InitBuffer(x1RowBuf, mBytes);
    pipe.InitBuffer(accumBuf, mBytes);
    pipe.InitBuffer(diffBuf, mBytes);
    pipe.InitBuffer(signBuf, mBytes);
    pipe.InitBuffer(powDstBuf, mBytes);
    int64_t maskBytes = mAligned_ / 8;
    if (maskBytes < 32)
        maskBytes = 32;
    int64_t chunkMaskBytes = rTile_ * mAligned_ / 8;
    if (chunkMaskBytes < maskBytes)
        chunkMaskBytes = maskBytes;
    pipe.InitBuffer(maskBuf, chunkMaskBytes);
    pipe.InitBuffer(maskBuf2, maskBytes);
    if (tmpBufSize_ > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufSize_);
    } else {
        pipe.InitBuffer(tmpBuf, 32);
    }
    pipe.InitBuffer(wsReadBuf, mAligned_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(zeroBuf, mBytes);
    pipe.InitBuffer(oneBuf, mBytes);
    pipe.InitBuffer(negOneBuf, mBytes);

    {
        int64_t calcBytes = rTile_ * mAligned_ * static_cast<int64_t>(sizeof(float));
        pipe.InitBuffer(x2CalcBuf, calcBytes);
        pipe.InitBuffer(gradCalcBuf, calcBytes);
        pipe.InitBuffer(distCalcBuf, calcBytes);
    }
    if constexpr (IS_FP16) {
        int64_t castBytes = mAligned_ * static_cast<int64_t>(sizeof(half));
        if (castBytes < 32)
            castBytes = 32;
        pipe.InitBuffer(castBuf, castBytes);
    }

    // Constants (fp32).
    zero_ = zeroBuf.Get<float>();
    one_ = oneBuf.Get<float>();
    negOne_ = negOneBuf.Get<float>();
    Duplicate(zero_, 0.0f, static_cast<uint32_t>(mAligned_));
    Duplicate(one_, 1.0f, static_cast<uint32_t>(mAligned_));
    Duplicate(negOne_, -1.0f, static_cast<uint32_t>(mAligned_));
    accum_ = accumBuf.Get<float>();

    // MTE2->V event (shared by x1Row load and Phase2 workspace read).
    pipe.InitBuffer(rowInQueue, 1, mAligned_ * static_cast<int64_t>(sizeof(T)));
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyInX1Row(int64_t b, int64_t i, int64_t mStart, int64_t mTileReal)
{
    // x1[b,i,0,mStart:mEnd] — j-direction broadcast row, this M segment.
    int64_t gmOffset = (b * pSize_ + i) * rSize_ * mSize_ + mStart;
    LocalTensor<T> x1In = rowInQueue.AllocTensor<T>();
    DataCopyPad(x1In, x1GM[gmOffset], {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(T))), 0, 0},
                {false, 0, 0, 0});
    rowInQueue.EnQue(x1In);
    LocalTensor<T> x1Ready = rowInQueue.DeQue<T>();
    // Copy out of the queue slot into the dedicated TBuf (both dtypes): aliasing the
    // queue tensor past FreeTensor races the next MTE2 fill of the reused slot.
    LocalTensor<float> x1Row = x1RowBuf.Get<float>();
    if constexpr (IS_FP16) {
        Cast(x1Row, x1Ready, RoundMode::CAST_NONE, static_cast<uint32_t>(mAligned_));
    } else {
        Adds(x1Row, x1Ready, 0.0f, static_cast<uint32_t>(mAligned_));
    }
    rowInQueue.FreeTensor(x1Ready);
    x1Row_ = x1Row;
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyInChunk(int64_t b, int64_t i, int64_t rStart,
                                                              int64_t currentRTile, int64_t mStart, int64_t mTileReal)
{
    // NOTE: DataCopyPad srcStride/dstStride are INTER-block gaps (not pitch).
    // Per-row single-block copies: GM rows are contiguous (gap 0); each UB row lives at
    // j*mAligned offset. Rows not 32B-multiples are auto-padded right with dummy; padding
    // lanes never reach the output (CopyOut blockLen = mTileReal).
    uint16_t rowLen = static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(T)));
    // Contiguous-chunk fast path: rows must be back-to-back in BOTH GM and UB.
    // GM row pitch is the FULL row (mSize_), so contiguity requires a single M segment
    // covering the whole row (numMTiles_ == 1) whose width also equals the UB pitch
    // (mTileReal == mAligned_). With M-tiling the GM rows are mSize_ apart while a
    // segment only spans mTileReal — a flat copy would interleave row k's segment
    // with row k+1's head.
    const bool wholeChunk = (numMTiles_ == 1 && mTileReal == mAligned_);
    const uint16_t chunkLen = static_cast<uint16_t>(currentRTile * static_cast<int64_t>(rowLen));

    // x2[b, i=0, rStart:rEnd, :] — x2 repeated along i.
    int64_t x2Base = b * pSize_ * rSize_ * mSize_ + rStart * mSize_ + mStart;
    LocalTensor<T> x2 = x2Queue.AllocTensor<T>();
    if (wholeChunk) {
        DataCopyPad(x2, x2GM[x2Base], {1, chunkLen, 0, 0}, {false, 0, 0, 0});
    } else {
        for (int64_t j = 0; j < currentRTile; j++) {
            DataCopyPad(x2[j * mAligned_], x2GM[x2Base + j * mSize_], {1, rowLen, 0, 0}, {false, 0, 0, 0});
        }
    }
    x2Queue.EnQue(x2);

    // grad[b, i, rStart:rEnd, :] — contiguous, scalar grad[b,i,j] repeated along k.
    int64_t gradBase = ((b * pSize_ + i) * rSize_ + rStart) * mSize_ + mStart;
    LocalTensor<T> grad = gradQueue.AllocTensor<T>();
    if (wholeChunk) {
        DataCopyPad(grad, gradOutputGM[gradBase], {1, chunkLen, 0, 0}, {false, 0, 0, 0});
    } else {
        for (int64_t j = 0; j < currentRTile; j++) {
            DataCopyPad(grad[j * mAligned_], gradOutputGM[gradBase + j * mSize_], {1, rowLen, 0, 0}, {false, 0, 0, 0});
        }
    }
    gradQueue.EnQue(grad);

    // cdist[b, i, rStart:rEnd, :] — contiguous, scalar dist[b,i,j] repeated along k.
    int64_t distBase = ((b * pSize_ + i) * rSize_ + rStart) * mSize_ + mStart;
    LocalTensor<T> dist = distQueue.AllocTensor<T>();
    if (wholeChunk) {
        DataCopyPad(dist, cdistResultGM[distBase], {1, chunkLen, 0, 0}, {false, 0, 0, 0});
    } else {
        for (int64_t j = 0; j < currentRTile; j++) {
            DataCopyPad(dist[j * mAligned_], cdistResultGM[distBase + j * mSize_], {1, rowLen, 0, 0}, {false, 0, 0, 0});
        }
    }
    distQueue.EnQue(dist);
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::ComputeChunk(int64_t currentRTile)
{
    LocalTensor<T> x2 = x2Queue.DeQue<T>();
    LocalTensor<T> grad = gradQueue.DeQue<T>();
    LocalTensor<T> dist = distQueue.DeQue<T>();

    {
        uint32_t chunkCount = static_cast<uint32_t>(currentRTile * mAligned_);
        LocalTensor<float> x2f = x2CalcBuf.Get<float>();
        LocalTensor<float> gradf = gradCalcBuf.Get<float>();
        LocalTensor<float> distf = distCalcBuf.Get<float>();
        if constexpr (IS_FP16) {
            Cast(x2f, x2, RoundMode::CAST_NONE, chunkCount);
            Cast(gradf, grad, RoundMode::CAST_NONE, chunkCount);
            Cast(distf, dist, RoundMode::CAST_NONE, chunkCount);
        } else {
            // Copy out of the queue slots immediately; compute reads only these TBufs.
            Adds(x2f, x2, 0.0f, chunkCount);
            Adds(gradf, grad, 0.0f, chunkCount);
            Adds(distf, dist, 0.0f, chunkCount);
        }
        x2Chunk_ = x2f;
        gradChunk_ = gradf;
        distChunk_ = distf;
        // Free the queue slots right away. The copies above are the only readers; drain
        // the Vector pipe first so the copy has RETIRED before the slot can be re-allocated
        // and refilled by MTE2 (Free is scalar-ordered, the copy executes asynchronously).
        PipeBarrier<PIPE_ALL>();
        x2Queue.FreeTensor(x2);
        gradQueue.FreeTensor(grad);
        distQueue.FreeTensor(dist);
    }

    static_cast<Derived*>(this)->PrepareChunk(currentRTile);

    for (int64_t j = 0; j < currentRTile; j++) {
        static_cast<Derived*>(this)->ComputeForJ(j);
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::ProcessSubTask(int64_t subIdx)
{
    int64_t taskIdx = subIdx / qSplit_;
    int64_t qPart = subIdx % qSplit_;
    int64_t b = taskIdx / pSize_;
    int64_t i = taskIdx % pSize_;
    currentRow_ = taskIdx;
    currentSubIdx_ = subIdx;

    int64_t qStart = qPart * qPartSize_;
    int64_t qEnd = qStart + qPartSize_;
    if (qEnd > rSize_)
        qEnd = rSize_;

    // ---- M-segment loop: each segment has its own x1 slice, accumulator, chunk pass ----
    for (int64_t mSeg = 0; mSeg < numMTiles_; mSeg++) {
        mStart_ = mSeg * mTileSize_;
        int64_t mTileReal = (mSeg == numMTiles_ - 1) ? lastMTileSize_ : mTileSize_;
        mAligned_ = AlignUpSeg(mTileReal);

        CopyInX1Row(b, i, mStart_, mTileReal);

        Duplicate(accum_, 0.0f, static_cast<uint32_t>(mAligned_));

        for (int64_t chunk = 0; chunk < numRChunks_; chunk++) {
            int64_t chunkStart = chunk * rTile_;
            int64_t chunkSize = (chunk == numRChunks_ - 1) ? lastRChunkSize_ : rTile_;
            int64_t rStart = (chunkStart > qStart) ? chunkStart : qStart;
            int64_t rEnd = (chunkStart + chunkSize < qEnd) ? chunkStart + chunkSize : qEnd;
            if (rStart >= rEnd)
                continue;
            int64_t currentRTile = rEnd - rStart;

            CopyInChunk(b, i, rStart, currentRTile, mStart_, mTileReal);
            ComputeChunk(currentRTile);
        }

        if (useTwoPhase_) {
            CopyOutPartial(mTileReal);
        } else {
            CopyOutRowToGradX1(accum_, currentRow_, mStart_, mTileReal);
        }
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutRowToGradX1(const LocalTensor<float>& accum, int64_t row,
                                                                     int64_t mStart, int64_t mTileReal)
{
    int64_t gmOffset = row * mSize_ + mStart;
    if constexpr (IS_FP16) {
        LocalTensor<half> outT = outQueue.AllocTensor<half>();
        Cast(outT, accum, RoundMode::CAST_ROUND, static_cast<uint32_t>(mAligned_));
        outQueue.EnQue(outT);
        LocalTensor<half> outY = outQueue.DeQue<half>();
        DataCopyPad(gradX1GM[gmOffset], outY,
                    {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(half))), 0, 0});
        outQueue.FreeTensor(outY);
    } else {
        LocalTensor<float> outT = outQueue.AllocTensor<float>();
        Adds(outT, accum, 0.0f, static_cast<uint32_t>(mAligned_));
        outQueue.EnQue(outT);
        LocalTensor<float> outY = outQueue.DeQue<float>();
        DataCopyPad(gradX1GM[gmOffset], outY,
                    {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(float))), 0, 0});
        outQueue.FreeTensor(outY);
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutAccum(int64_t mTileReal)
{
    CopyOutRowToGradX1(accum_, currentRow_, mStart_, mTileReal);
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutPartial(int64_t mTileReal)
{
    int64_t wsOffset = currentSubIdx_ * mAlignedFull_ + mStart_;
    if constexpr (IS_FP16) {
        LocalTensor<half> outT = outQueue.AllocTensor<half>();
        Cast(outT, accum_, RoundMode::CAST_NONE, static_cast<uint32_t>(mAligned_));
        outQueue.EnQue(outT);
        LocalTensor<half> outY = outQueue.DeQue<half>();
        DataCopyPad(wsGM[wsOffset], outY,
                    {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(half))), 0, 0});
        outQueue.FreeTensor(outY);
    } else {
        LocalTensor<float> outT = outQueue.AllocTensor<float>();
        Adds(outT, accum_, 0.0f, static_cast<uint32_t>(mAligned_));
        outQueue.EnQue(outT);
        LocalTensor<float> outY = outQueue.DeQue<float>();
        DataCopyPad(wsGM[wsOffset], outY,
                    {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(float))), 0, 0});
        outQueue.FreeTensor(outY);
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::MergeToGradX1()
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalRows = batchSize_ * pSize_;
    int64_t rowsPerCore = (totalRows + usedCoreNum_ - 1) / usedCoreNum_;
    int64_t rowStart = blockIdx * rowsPerCore;
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > totalRows)
        rowEnd = totalRows;
    if (rowStart >= rowEnd)
        return;

    LocalTensor<T> wsRow = wsReadBuf.Get<T>();
    LocalTensor<float> partial = diffBuf.Get<float>();
    for (int64_t row = rowStart; row < rowEnd; row++) {
        for (int64_t mSeg = 0; mSeg < numMTiles_; mSeg++) {
            mStart_ = mSeg * mTileSize_;
            int64_t mTileReal = (mSeg == numMTiles_ - 1) ? lastMTileSize_ : mTileSize_;
            mAligned_ = AlignUpSeg(mTileReal);
            Duplicate(accum_, 0.0f, static_cast<uint32_t>(mAligned_));
            // Fixed ascending qPart order -> bit-identical deterministic result.
            for (int64_t q = 0; q < qSplit_; q++) {
                int64_t wsOffset = (row * qSplit_ + q) * mAlignedFull_ + mStart_;
                LocalTensor<T> wsIn = rowInQueue.AllocTensor<T>();
                DataCopyPad(wsIn, wsGM[wsOffset],
                            {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(T))), 0, 0},
                            {false, 0, 0, 0});
                rowInQueue.EnQue(wsIn);
                LocalTensor<T> wsReady = rowInQueue.DeQue<T>();
                wsRow = wsReady;
                if constexpr (IS_FP16) {
                    Cast(partial, wsRow, RoundMode::CAST_NONE, static_cast<uint32_t>(mAligned_));
                } else {
                    Adds(partial, wsRow, 0.0f, static_cast<uint32_t>(mAligned_));
                }
                Add(accum_, accum_, partial, static_cast<uint32_t>(mAligned_));
                rowInQueue.FreeTensor(wsReady);
            }
            CopyOutRowToGradX1(accum_, row, mStart_, mTileReal);
        }
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::Process()
{
    if (useTwoPhase_) {
        for (int64_t t = 0; t < subTaskCount_; t++) {
            ProcessSubTask(startSubTask_ + t);
        }
        // Drain MTE3 before the cross-core barrier: SyncAll only fences instruction
        // flow arrival, workspace writes (MTE3) may still be in flight on other cores.
        PipeBarrier<PIPE_ALL>();
        SyncAll(); // ensure all partial sums written
        MergeToGradX1();
        PipeBarrier<PIPE_ALL>(); // drain MTE3 (gradX1 writes) before final barrier
        SyncAll();               // ensure all gradX1 rows written
    } else {
        for (int64_t t = 0; t < subTaskCount_; t++) {
            ProcessSubTask(startSubTask_ + t); // segment-wise CopyOut inside
        }
    }
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_COMMON_H
