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
 * \file radix_top_k_ws.h
 * \brief Radix TopK kernel — workspace variant (tileHist & tileTopK in GM workspace)
 */

#ifndef RADIX_TOPK_WS_H
#define RADIX_TOPK_WS_H

#include "radix_top_k_common.h"

namespace RadixTopK {
using namespace AscendC;

template <typename T, bool largest>
class RadixTopKWs : public RadixTopKBaseKernel<T, largest> {
    using Base = RadixTopKBaseKernel<T, largest>;

public:
    __aicore__ inline RadixTopKWs(TPipe& tpipe, const RadixTopKTilingData& tilingData) : Base(tpipe, tilingData) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace)
    {
        InitWsGm(x, k, values, indices, workspace);
        InitBuffers(sizeof(int32_t)); // 索引输出 int32；int64 大 shape 子类改传 sizeof(int64_t)
    }

    __aicore__ inline void SubProcess(uint64_t batchId);
    __aicore__ inline void Process();

    // int64 大 shape 子类 RadixTopKWsI64 复用以下成员与 WS 搬运工具（仅访问级别，无逻辑变化）
protected:
    /** @brief 初始化 GM/WS 布局（含 indexGm_ 的 int32 视图）；int64 子类复用后再补设 indexGm64_ */
    __aicore__ inline void InitWsGm(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace);
    __aicore__ inline void InitParams();
    /** @brief 形参为索引输出元素字节数：int32 变体 sizeof(int32_t)、int64 变体 sizeof(int64_t) */
    __aicore__ inline void InitBuffers(const uint32_t& indexElemSize);
    __aicore__ inline void ClearTileTopKInWs();
    __aicore__ inline void ClearHistInWs(const int32_t& roundId);
    __aicore__ inline void ClearTileHistInWs(const uint32_t& dstOffset, const uint32_t& dataNum);
    __aicore__ inline void CopyTileHistWs2Ub(const LocalTensor<int32_t>& dstTensor, const uint32_t& srcOffset,
                                             const uint32_t& dataNum);
    __aicore__ inline void CopyTileTopKWs2Ub(const LocalTensor<int32_t>& dstTensor, const uint32_t& srcOffset,
                                             const uint32_t& dataNum);
    __aicore__ inline void CopyTileTopKUb2Ws(const LocalTensor<int32_t>& srcTensor, const uint32_t& dstOffset,
                                             const uint32_t& dataNum);
    __aicore__ inline void CopyTileHistInWs(const uint32_t& dstOffset, const uint32_t& srcOffset);
    __aicore__ inline void SubTileHistWs2Ub(const LocalTensor<int32_t>& dstTensor,
                                            const LocalTensor<int32_t>& srcTensor, const uint32_t& srcOffset0,
                                            const uint32_t& srcOffset1, const uint32_t& dataNum);
    __aicore__ inline void SubTileHistInWs(const uint32_t& dstOffset, const uint32_t& srcOffset0,
                                           const uint32_t& srcOffset1, const uint32_t& dataNum);
    __aicore__ inline void SubTileHistInWsAll(const uint32_t& dstOffset, const uint32_t& srcOffset0,
                                              const uint32_t& srcOffset1);
    __aicore__ inline void AddTileHist2TileTopKInWs(const uint64_t& tileHistOffset);
    __aicore__ inline bool Update(const int32_t& roundId);
    __aicore__ inline uint32_t CreateVecIndex4TopK();
    __aicore__ inline uint64_t FindFirstTileWithTopK(const LocalTensor<int32_t>& tileTopK);
    __aicore__ inline void TileTopK(const uint64_t& batchId, const uint64_t& blockOffset);

    __aicore__ inline void AddTileHistToTileTopK(LocalTensor<int32_t>& globalHist);
    __aicore__ inline void HandleLastRoundBoundary(LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);
    __aicore__ inline void WriteCoreTopKFromWs(LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);

    GlobalTensor<int32_t> tileTopKGm_;            /**< WS 中 tileTopK 段的 Global Tensor */
    GlobalTensor<int32_t> tileHistGm_;            /**< WS 中 tileHist 段的 Global Tensor */
    uint32_t tileNumRepeatTimes_;                 /**< tile 写入 WS 的整块次数 */
    uint32_t tileNumRemain_;                      /**< tile 写入 WS 的剩余数量 */
    uint32_t tileNumBy2RepeatTimes_;              /**< tile 双 buffer 整块次数 */
    uint32_t tileNumBy2Remain_;                   /**< tile 双 buffer 剩余数量 */
    uint32_t tileNumBy3RepeatTimes_;              /**< tile 三 buffer 整块次数 */
    uint32_t tileNumBy3Remain_;                   /**< tile 三 buffer 剩余数量 */
    TBuf<TPosition::VECCALC> tileHistAndTopKBuf_; /**< tileHist 和 tileTopK 复用 UB buffer */
    TBuf<TPosition::VECCALC> tempBuf_;            /**< 临时 buffer（复用为 cmpMask / vecIndex） */
};

#include "radix_top_k_ws_utils.h"

/**
 * @brief 主流程入口：遍历所有 batch，对每个 batch 完成 WS 变体 Radix TopK 计算
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::Process()
{
    for (uint64_t batchId = 0; batchId < this->batch_; batchId++) {
        SubProcess(batchId);
        SyncAll();
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);
}

/**
 * @brief 初始化 GM/WS 布局（含 indexGm_ 的 int32 视图）
 *        int64 大 shape 子类 RadixTopKWsI64 复用本函数后再补设 indexGm64_，不重复整段布局代码
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::InitWsGm(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices,
                                                         GM_ADDR workspace)
{
    InitParams();
    this->xGm_.SetGlobalBuffer((__gm__ T*)x);
    this->valueGm_.SetGlobalBuffer((__gm__ T*)values);
    this->kGm_.SetGlobalBuffer((__gm__ int32_t*)k);
    this->indexGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
    this->globalHistGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);
    this->boundaryBinCumSumGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_);
    this->coreTopKGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_ + this->coreNum_);
    uint32_t tileTopKOffset = 0;
    if (this->blockIdx_ < this->formerCoreNum_) {
        tileTopKOffset = this->blockIdx_ * this->formerTileNum_;
    } else {
        tileTopKOffset = this->formerCoreNum_ * this->formerTileNum_ +
                         (this->blockIdx_ - this->formerCoreNum_) * this->tailTileNum_;
    }
    uint32_t tileHistOffset = this->numValue_ * tileTopKOffset;
    tileTopKGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_ + this->coreNum_ * 2 +
                                tileTopKOffset);
    tileHistGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_ + this->coreNum_ * 2 +
                                this->totalTileNum_ + tileHistOffset);
}

/**
 * @brief 初始化基本参数（调用基类）+ WS 变体特有参数（tileNum 分段次数）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::InitParams()
{
    this->InitBaseParams();
    tileNumRepeatTimes_ = this->tileNum_ / MAX_TILE_NUM_IN_UB;
    tileNumRemain_ = this->tileNum_ % MAX_TILE_NUM_IN_UB;
    tileNumBy2RepeatTimes_ = this->tileNum_ / MAX_TILE_NUM_IN_UB_BY2;
    tileNumBy2Remain_ = this->tileNum_ % MAX_TILE_NUM_IN_UB_BY2;
    tileNumBy3RepeatTimes_ = this->tileNum_ / MAX_TILE_NUM_IN_UB_BY3;
    tileNumBy3Remain_ = this->tileNum_ % MAX_TILE_NUM_IN_UB_BY3;
}

/**
 * @brief 初始化 UB 缓冲区：xBuf Ping-Pong、tileHistAndTopKBuf、tempBuf、outBuf、globalHistBuf
 *        WS 变体中 tileHist/tileTopK 存 GM，UB 只用作分段读写缓冲
 * @param indexElemSize 索引输出元素字节数（int32 变体 4、int64 变体 8）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::InitBuffers(const uint32_t& indexElemSize)
{
    uint32_t cmpMaskSize = this->tileLen_ * sizeof(uint8_t) / BYTE_SIZE;
    this->tpipe_.InitBuffer(this->xBufPing_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(this->xBufPong_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(tileHistAndTopKBuf_, MAX_TILE_NUM_IN_UB * sizeof(int32_t));
    this->tpipe_.InitBuffer(tempBuf_, cmpMaskSize);
    this->tpipe_.InitBuffer(this->outValueBuf_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(this->outIndexBuf_, this->tileLen_ * indexElemSize);
    this->tpipe_.InitBuffer(this->globalHistBuf_, this->numValue_ * sizeof(int32_t));
}

/**
 * @brief 对单个 batch 执行 WS 变体 Radix TopK：
 *        逐轮 Ping-Pong CopyIn → Twiddle → 直方图统计(分段写 WS) → 全局同步 → 边界查找 → 累加 tileTopK(WS)
 * @param batchId 当前 batch ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::SubProcess(uint64_t batchId)
{
    this->involvedMask16_ = 0;
    this->andMask16_ = 0xC000; // ((1 << BITS_PER_ROUND) - 1) << (16 - BITS_PER_ROUND);
    this->totalDefinitelyInTopK_ = 0;
    this->boundaryBin = 0;
    this->boundaryBinPrev = 1;
    this->globalHistBoundaryNum_ = this->tileNum_ * this->tileLen_;
    this->remainK_ = this->kValue_;
    uint64_t blockOffset = this->CalcBlockOffset(batchId);
    ClearTileTopKInWs();

    for (int32_t roundId = this->round_ - 1; roundId >= 0; roundId--) {
        uint64_t firstTileLen = this->tileNum_ > 1 ? this->tileLen_ : this->tailTileLen_;
        this->CopyIn(this->xBufPing_, firstTileLen, blockOffset);
        this->pingPongFlag = true;
        this->template CopyOut2Ws<true>(this->numValue_, this->blockIdx_ * this->numValue_);
        ClearHistInWs(roundId);

        SetFlag<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);

        for (int32_t repeatId = 0; repeatId <= tileNumBy3RepeatTimes_; repeatId++) {
            int32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB_BY3;
            int32_t endTileId = AscendC::Std::min(static_cast<int32_t>(startTileId + MAX_TILE_NUM_IN_UB_BY3),
                                                  static_cast<int32_t>(this->tileNum_));
            MTE3ToSSync();
            for (int32_t tileId = startTileId; tileId < endTileId; tileId++) {
                PINGPONG_TILE_BEGIN(tileId, blockOffset);

                this->CanonicalizeDataForLargest(curBuf, curTileLen);
                this->TwiddleInB16(curBuf, curTileLen);
                this->DoAndMask(curBuf, curTileLen);
                LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
                this->CalcCumsumHistogram16(curBuf, roundId, tileHist, tileId - startTileId, MAX_TILE_NUM_IN_UB_BY3,
                                            MAX_TILE_NUM_IN_UB_BY3, curTileLen);

                PINGPONG_TILE_END(tileId);
            }
            SToMTE3Sync();
            LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
            for (int32_t binMask = this->numValue_ - 1; binMask > 0; binMask--) {
                DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1),
                                                    static_cast<uint32_t>(sizeof(int32_t) * (endTileId - startTileId)),
                                                    0, 0, 0};
                uint32_t tileHistOffset = binMask * this->tileNum_ + repeatId * MAX_TILE_NUM_IN_UB_BY3;
                DataCopyPad(tileHistGm_[tileHistOffset], tileHist[(binMask - 1) * MAX_TILE_NUM_IN_UB_BY3],
                            tileHistOutParams);
            }
        }
        this->template CopyOut2Ws<false>(this->numValue_, this->blockIdx_ * this->numValue_);
        SyncAll();
        this->andMask16_ >>= BITS_PER_ROUND;
        if (Update(roundId))
            break;
        // 与 ub 模板对齐：确保所有核完成 Update 内 CopyInGlobalHist 对 globalHistGm_ 的读取，
        // 再允许快核进入下一轮 CopyOut2Ws<true> 覆盖
        SyncAll();
    }
    SyncAll();
    TileTopK(batchId, blockOffset);
}

/**
 * @brief 单轮 Update（WS 变体）：读入全局直方图 → 归约 → 找边界 bin → 累加 tileTopK(WS) → 处理末轮边界 → 写 coreTopK
 * @param roundId 当前轮 ID
 * @return true: 可提前退出
 */
template <typename T, bool largest>
__aicore__ inline bool RadixTopKWs<T, largest>::Update(const int32_t& roundId)
{
    LocalTensor<int32_t> resTensor = this->outIndexBuf_.template Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;

    this->CopyInGlobalHist(globalHist);
    MTE2ToVSync();
    this->ReduceGlobalHist(globalHist);
    this->FindBoundaryBin(globalHist, roundId);

    AddTileHistToTileTopK(globalHist);

    this->remainK_ = this->kValue_ - this->totalDefinitelyInTopK_;
    LocalTensor<float> tileHistFp32 = resTensor.template ReinterpretCast<float>();
    if (roundId == 0 && this->remainK_ > 0) {
        HandleLastRoundBoundary(resTensor, tileHistFp32);
    }

    bool isLastRound = (this->remainK_ == 0 || roundId == 0);
    if (isLastRound) {
        WriteCoreTopKFromWs(resTensor, tileHistFp32);
    }

    return isLastRound;
}

/**
 * @brief WS 变体：将边界 bin 之前的确定性元素通过 WS 累加到 tileTopK
 * @param globalHist 全局直方图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::AddTileHistToTileTopK(LocalTensor<int32_t>& globalHist)
{
    if (globalHist(this->boundaryBin) == this->remainK_) {
        AddTileHist2TileTopKInWs(this->boundaryBin * this->tileNum_);
        this->totalDefinitelyInTopK_ += globalHist(this->boundaryBin);
    } else if (this->boundaryBinPrev < this->numValue_) {
        AddTileHist2TileTopKInWs(this->boundaryBinPrev * this->tileNum_);
        this->totalDefinitelyInTopK_ += globalHist(this->boundaryBinPrev);
    }
}

/**
 * @brief 处理最后一轮边界 bin 的元素分配：从 WS 读取各 tile 的边界 bin / 前驱 bin 计数，
 *        按 (core, tile) 顺序把剩余配额 remainK_ 分配到各 tile 的 tileTopK
 *
 *        计数与配额统一用 int64 标量（跨核总量可达 sortLen > 2^31；int32 形状下同样安全），
 *        分配按 MAX_TILE_NUM_IN_UB_BY3 分段：段内先求差分总和 segSum，若剩余配额能吞下整段
 *        则按元素向量 Add 给段内各 tile 加各自差分、跳过逐 tile 标量扫描，只在配额耗尽的那个
 *        段内退化为逐 tile 标量扫描。两条变体共用本实现（int64 变体不再单独 override）。
 * @param resTensor 临时 Tensor
 * @param tileHistFp32 tileHist 的 fp32 视图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::HandleLastRoundBoundary(LocalTensor<int32_t>& resTensor,
                                                                        LocalTensor<float>& tileHistFp32)
{
    int64_t reduceSumValue = 0;
    // 首个迭代前，ReduceGlobalHist 末尾用 Scalar SetValue 写过 resTensor[0..3]，
    // 需保证 S 写对后续 MTE2 覆写 resTensor 可见（与 ub 模板 HandleLastRoundBoundary 的 SToVSync 对齐）
    SToMTE2Sync();
    for (int32_t i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = i == tileNumBy2RepeatTimes_ ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;
        if (this->boundaryBinPrev < this->numValue_) {
            SubTileHistWs2Ub(resTensor, resTensor, this->boundaryBin * this->tileNum_ + i * MAX_TILE_NUM_IN_UB_BY2,
                             this->boundaryBinPrev * this->tileNum_ + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
            PipeBarrier<PIPE_V>();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
        } else {
            CopyTileHistWs2Ub(resTensor, this->boundaryBin * this->tileNum_ + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
            MTE2ToVSync();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
        }
        int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
        int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
        int32_t segSize = dataNum < safeSumTileNum ? dataNum : safeSumTileNum;
        PipeBarrier<PIPE_V>();
        for (int32_t segOff = 0; segOff < dataNum; segOff += segSize) {
            int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(dataNum - segOff));
            SToVSync();
            ReduceSum(tileHistFp32, tileHistFp32[segOff], tileHistFp32[segOff], curSegLen);
            VToSSync();
            // 分段和 <= 2^24 精确；累加器 int64（跨段总量 = per-core 边界 bin 元素数）
            reduceSumValue += static_cast<int64_t>(tileHistFp32.GetValue(0));
        }
    }
    // per-core 边界 bin 计数 < 2^31（per-core 元素数上限），int32 写回安全
    resTensor.SetValue(0, static_cast<int32_t>(reduceSumValue));
    SToMTE3Sync();
    this->CopyOutBoundaryBinCumSum(resTensor);
    SyncAll();

    int64_t cumSumValuePrev = 0;
    // 复用基类的前缀和（出参已是 int64）：本变体与 UB 变体口径一致，避免两份只差累加器类型的实现
    this->ComputeCumSumPrev(resTensor, cumSumValuePrev);
    int64_t remainCoreBoundaryNum = static_cast<int64_t>(this->remainK_) - cumSumValuePrev;
    if (remainCoreBoundaryNum > 0) {
        LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
        LocalTensor<int32_t> tileHistBoundary = tileTopK[MAX_TILE_NUM_IN_UB_BY3];
        LocalTensor<int32_t> tileHistBoundaryPrev = tileTopK[2 * MAX_TILE_NUM_IN_UB_BY3];
        for (int32_t i = 0; i <= tileNumBy3RepeatTimes_; i++) {
            uint32_t dataNum = i == tileNumBy3RepeatTimes_ ? tileNumBy3Remain_ : MAX_TILE_NUM_IN_UB_BY3;
            if (dataNum == 0)
                continue;
            uint32_t startTileId = i * MAX_TILE_NUM_IN_UB_BY3;
            uint32_t endTileId = startTileId + dataNum;

            CopyTileHistWs2Ub(tileHistBoundary, this->boundaryBin * this->tileNum_ + startTileId, dataNum);
            CopyTileHistWs2Ub(tileHistBoundaryPrev, this->boundaryBinPrev * this->tileNum_ + startTileId, dataNum);
            CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
            MTE2ToVSync();
            // 段级定位：先算本段边界 bin 差分总和 segSum（向量），若剩余配额能吞下整段，
            // 则按元素向量 Add 给段内各 tile 加各自差分（tileTopK[i] += diff[i]），
            // 跳过逐 tile 标量扫描——标量扫描只在配额耗尽的那个段内进行（≤2048 次后必然 break）。
            // 注意：整段吞下不是加常量 segSum（那会给每个 tile 超发全段总和的配额，
            // 导致 curTileK 超过 tile 实际 EQ 元素数、拷出陈旧 lane——NaN/-1 输出的根源）
            if (this->boundaryBinPrev < this->numValue_) {
                Sub(tileHistBoundary, tileHistBoundary, tileHistBoundaryPrev, dataNum);
                PipeBarrier<PIPE_V>();
            }
            int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
            int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
            // fp32 视图借用已被 Sub 消费的 prev 区域，保持 tileHistBoundary 的 int32 差分
            // 不被 Cast 覆写（整段 Add 与段内标量扫描都还要复用它）
            LocalTensor<float> segFp32 = tileHistBoundaryPrev.template ReinterpretCast<float>();
            PipeBarrier<PIPE_V>();
            Cast(segFp32, tileHistBoundary, RoundMode::CAST_NONE, dataNum);
            PipeBarrier<PIPE_V>();
            int64_t segSum = 0;
            int32_t segSize = dataNum < static_cast<uint32_t>(safeSumTileNum) ? static_cast<int32_t>(dataNum) :
                                                                                safeSumTileNum;
            for (int32_t segOff = 0; segOff < static_cast<int32_t>(dataNum); segOff += segSize) {
                int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(dataNum - segOff));
                SToVSync();
                ReduceSum(segFp32, segFp32[segOff], segFp32[segOff], curSegLen);
                VToSSync();
                segSum += static_cast<int64_t>(segFp32.GetValue(0));
            }
            if (remainCoreBoundaryNum >= segSum) {
                // 整段吞下：tileTopK += diff 向量按元素加（int32 Add）
                SToVSync();
                PipeBarrier<PIPE_V>();
                Add(tileTopK, tileTopK, tileHistBoundary, dataNum);
                PipeBarrier<PIPE_V>();
                remainCoreBoundaryNum -= segSum;
                // Add 是 V 写 tileTopK，紧接着 MTE3 要读同一块 UB：PipeBarrier<PIPE_V> 只保证 V 流水线
                // 内部顺序，SToMTE3Sync 只覆盖 S→MTE3，两者都管不到 V→MTE3，MTE3 可能读到 Add 之前的
                // 旧值（该段 k 配额错乱）。同文件 AddTileHist2TileTopKInWs 对同样的 Add→CopyOut 序列
                // 用的就是 VToMTE3Sync，这里必须保持一致。
                VToMTE3Sync();
                CopyTileTopKUb2Ws(tileTopK, startTileId, dataNum);
                MTE3ToMTE2Sync();
                if (remainCoreBoundaryNum == 0) {
                    break;
                }
                continue;
            }
            // 段内部分配：逐 tile 标量扫描（本段必是配额耗尽段；差分已在 tileHistBoundary 内，
            // 标量侧无需再减 prev——prev 区域已被 segFp32 借用覆写）
            MTE2ToSSync();
            for (int32_t tileId = startTileId; tileId < static_cast<int32_t>(endTileId); tileId++) {
                int64_t curTileBoundaryNum = tileHistBoundary.GetValue(tileId - startTileId);
                if (curTileBoundaryNum < remainCoreBoundaryNum) {
                    remainCoreBoundaryNum -= curTileBoundaryNum;
                    tileTopK.SetValue(tileId - startTileId, tileTopK.GetValue(tileId - startTileId) +
                                                                static_cast<int32_t>(curTileBoundaryNum));
                } else {
                    tileTopK.SetValue(tileId - startTileId, tileTopK.GetValue(tileId - startTileId) +
                                                                static_cast<int32_t>(remainCoreBoundaryNum));
                    remainCoreBoundaryNum = 0;
                    break;
                }
            }
            SToMTE3Sync();
            CopyTileTopKUb2Ws(tileTopK, startTileId, dataNum);
            MTE3ToMTE2Sync();
        }
    }
}

/**
 * @brief 从 WS 分段读取 tileTopK 累加得到本 core 总 TopK 计数并写回 WS
 *        累加器 int64（int32 形状下同样安全），两条变体共用
 * @param resTensor 临时 Tensor
 * @param tileHistFp32 tileHist 的 fp32 视图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::WriteCoreTopKFromWs(LocalTensor<int32_t>& resTensor,
                                                                    LocalTensor<float>& tileHistFp32)
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    int64_t totalTileTopKInCore = 0;
    int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
    int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
    for (int32_t i = 0; i <= tileNumRepeatTimes_; i++) {
        int32_t dataNum = i == tileNumRepeatTimes_ ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToVSync();
        int32_t segSize = dataNum < safeSumTileNum ? dataNum : safeSumTileNum;
        for (int32_t segOff = 0; segOff < dataNum; segOff += segSize) {
            int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(dataNum - segOff));
            SToVSync();
            Cast(tileHistFp32, tileTopK[segOff], RoundMode::CAST_NONE, curSegLen);
            PipeBarrier<PIPE_V>();
            ReduceSum(tileHistFp32, tileHistFp32, tileHistFp32, curSegLen);
            VToSSync();
            totalTileTopKInCore += static_cast<int64_t>(tileHistFp32.GetValue(0));
        }
    }
    // per-core 计数 < 2^31（per-core 元素数上限），coreTopK 通道保持 int32
    this->CopyOutCoreTopK(resTensor, static_cast<int32_t>(totalTileTopKInCore));
}

/**
 * @brief 为 TileTopK 阶段创建预生成索引，WS模板从 tileHistAndTopKBuf_ 剩余空间分配
 * @return 实际可用的 maxIndexLen
 */
template <typename T, bool largest>
__aicore__ inline uint32_t RadixTopKWs<T, largest>::CreateVecIndex4TopK()
{
    // tileNumAlign_ 是 UB 内 tileHistAndTopKBuf_ 已占用的元素数（= 对齐后的 tileNum_）。i64 变体的
    // 多段 regime 会命中 tileNum_ > MAX_TILE_NUM_IN_UB：此时下面的无符号相减会下溢成巨大值，再被
    // tileLen_ 兜住，最终 GetWithOffset(maxIndexLen, tileNumAlign_ * 4) 的偏移会写出 6144 元素 buffer。
    // 这里把「不变量：仅当 tileNumAlign_ < MAX_TILE_NUM_IN_UB 才调用」显式落到代码，返回 0 时下游
    // 走「现算索引」分支，行为与 maxIndexLen==0 的既有兜底一致。
    if (this->tileNumAlign_ >= MAX_TILE_NUM_IN_UB) {
        return 0;
    }
    uint64_t int32Align = BLOCK_SIZE / sizeof(int32_t);
    uint32_t maxIndexLen = Ops::Base::FloorDiv(MAX_TILE_NUM_IN_UB - this->tileNumAlign_, int32Align) * int32Align;
    if (this->tileLen_ < maxIndexLen) {
        maxIndexLen = this->tileLen_;
    } else if (maxIndexLen < 512) {
        maxIndexLen = 0;
    }
    if (maxIndexLen > 0) {
        LocalTensor<int32_t> vecIndex = tileHistAndTopKBuf_.GetWithOffset<int32_t>(
            maxIndexLen, this->tileNumAlign_ * sizeof(int32_t));
        Base::CreateVecIndex4TopK(maxIndexLen, vecIndex);
    }
    return maxIndexLen;
}

/**
 * @brief WS 变体：从 WS 分段读取 tileTopK 查找第一个有 TopK 的 tile
 * @param tileTopK tileTopK UB buffer
 * @return 第一个有 TopK 的 tile ID
 */
template <typename T, bool largest>
__aicore__ inline uint64_t RadixTopKWs<T, largest>::FindFirstTileWithTopK(const LocalTensor<int32_t>& tileTopK)
{
    uint64_t firstId = 0;
    for (int i = 0; i <= tileNumRepeatTimes_; i++) {
        int32_t startTileId = i * MAX_TILE_NUM_IN_UB;
        int32_t id = 0;
        uint32_t dataNum = i == tileNumRepeatTimes_ ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
        MTE2ToSSync();
        while (id < dataNum && tileTopK.GetValue(id) <= 0)
            id++;
        if (id < dataNum) {
            firstId = startTileId + id;
            break;
        }
    }
    return firstId;
}

/**
 * @brief WS 变体 TileTopK：从 WS 分段读取 tileTopK，按 tileTopK 计数提取 TopK 元素并写回输出
 *        按 MAX_TILE_NUM_IN_UB 分段处理，Ping-Pong 双缓冲
 * @param batchId 当前 batch ID
 * @param blockOffset 当前 core 在输入数据中的起始偏移
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::TileTopK(const uint64_t& batchId, const uint64_t& blockOffset)
{
    LocalTensor<int32_t> tmpLocal = tempBuf_.Get<int32_t>();
    uint64_t coreTopKOffset = this->CopyInCoreTopK(tmpLocal);
    uint64_t batchIndexOffset = batchId * this->kValue_;
    coreTopKOffset += batchIndexOffset;

    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    uint64_t firstId = FindFirstTileWithTopK(tileTopK);

    uint64_t firstLen = (firstId == this->tileNum_ - 1) ? this->tailTileLen_ : this->tileLen_;
    this->CopyIn(this->xBufPing_, firstLen, blockOffset + firstId * this->tileLen_);
    uint32_t maxIndexLen = CreateVecIndex4TopK();
    // 同 UB 变体：maxIndexLen == 0 时 SubTopKAndCopyOut 走现算索引兜底分支、
    // 不使用 vecIndex；因此 0 时改用 buffer 首地址占位，不做零长度取句柄。
    LocalTensor<int32_t> vecIndexTopK = maxIndexLen > 0 ?
                                            tileHistAndTopKBuf_.GetWithOffset<int32_t>(
                                                maxIndexLen, this->tileNumAlign_ * sizeof(int32_t)) :
                                            tileHistAndTopKBuf_.Get<int32_t>(); // 占位句柄，0 时不会被使用

    this->InitTopKEvent();
    uint64_t startRepeatId = firstId / MAX_TILE_NUM_IN_UB;
    for (uint64_t repeatId = startRepeatId; repeatId <= tileNumRepeatTimes_; repeatId++) {
        uint64_t startTileId = repeatId * MAX_TILE_NUM_IN_UB;
        uint64_t endTileId = AscendC::Std::min(static_cast<uint64_t>(startTileId + MAX_TILE_NUM_IN_UB),
                                               static_cast<uint64_t>(this->tileNum_));
        if (repeatId > startRepeatId) {
            CopyTileTopKWs2Ub(tileTopK, repeatId * MAX_TILE_NUM_IN_UB, MAX_TILE_NUM_IN_UB);
            MTE2ToSSync();
        } else {
            startTileId = firstId;
        }
        for (uint64_t tileId = startTileId; tileId < endTileId; tileId++) {
            PINGPONG_TILE_BEGIN(tileId, blockOffset);

            uint64_t tileBaseOffset = blockOffset - batchId * this->sortLen_;
            uint64_t tileOffset = tileBaseOffset + tileId * this->tileLen_;

            int32_t curTileK = tileTopK.GetValue(tileId - repeatId * MAX_TILE_NUM_IN_UB);
            PipeBarrier<PIPE_V>();
            PROCESS_ONE_TILE_TOPK_OR_SKIP(curTileK, vecIndexTopK);

            PINGPONG_TILE_END(tileId);
        }
    }
    this->FinishTopKEvent();
}

} // namespace RadixTopK
#endif // RADIX_TOPK_WS_H
