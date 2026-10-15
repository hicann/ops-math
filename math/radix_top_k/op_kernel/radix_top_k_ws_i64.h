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
 * \file radix_top_k_ws_i64.h
 * \brief Radix TopK kernel — int64 大 shape 变体（sortLen > INT32_MAX）
 *
 * 计算逻辑与 RadixTopKWs 完全一致（继承复用其全部 WS 搬运工具与同步结构），
 * 仅以下位置 int64 化。910B 向量指令（Add/Adds/Duplicate/GatherMask）不支持
 * int64，唯一可用的 int64 向量能力是 Cast 的 s32<->s64 转换通道，且不能承担
 * 加法/合并，因此 int64 算术一律走标量：
 *   1) 跨核 globalHist 归约与边界查找（总量可达 sortLen > 2^31）：标量 int64 累加，
 *      每轮 coreNum*numValue_（<=200）次，代价可忽略；
 *   2) 边界计数跨核前缀和（ComputeCumSumPrevI64）与 remainCoreBoundaryNum：int64 标量；
 *   3) 索引输出：GatherMask 仍在 int32 域压缩 tile 内索引，逐元素标量加 int64
 *      tileOffset 后就地后向扩展写 int64，成本 O(kValue)。
 *
 * 与基类 RadixTopKWs 的复用边界（避免两处同步修改）：
 *   - 复用（本类不重写）：InitWsGm（GM/WS 布局）、InitBuffers（差一个索引元素字节数）、
 *     HandleLastRoundBoundary（int64 计数 + 段级快速路径）、WriteCoreTopKFromWs、
 *     ComputeCumSumPrevI64、全部 WS 搬运/清除/同步工具；
 *   - 必须重写：Process/SubProcess（非虚成员 + 静态派发，基类体内会静态绑定到基类 Update）、
 *     Update 及其 int64 归约/边界查找/neighbor 累加、TileTopK、索引输出相关函数。
 *   若要进一步合并 Process/SubProcess，需要引入 CRTP 或把基类模板化为 <T, largest, IndexT>，
 *   属结构性重构，未在本 PR 处理。
 *
 * int32 安全域（保持不变）：per-core globalHist/tileHist/tileTopK 计数与 coreTopK
 * （per-core 元素数 = sortLen/coreNum < 2^31，由 tiling 保证，物理内存内恒成立；
 * kValue <= INT32_MAX 由 tiling 校验）、vecIndex（< tileLen）、x/value 数据通路。
 */

#ifndef RADIX_TOP_K_WS_I64_H
#define RADIX_TOP_K_WS_I64_H

#include "radix_top_k_ws.h"

namespace RadixTopK {
using namespace AscendC;

/**
 * @brief i64 变体专用的 tile 循环前导：与 PROCESS_ONE_TILE_TOPK_OR_SKIP 的唯一区别是
 *        不传 vecIndex/maxIndexLen——i64 路径的索引一律在 SubTopKAndCopyOutI64 内现算
 *        （VectorMergeI64），预生成索引链路整条不适用。
 * 前置条件：curBuf, coreTopKOffset, tileOffset, curTileLen 在作用域内
 */
#define PROCESS_ONE_TILE_TOPK_OR_SKIP_I64(curTileKVal)                                                    \
    if (curTileKVal > 0) {                                                                                \
        auto xLocal = curBuf.template Get<T>();                                                           \
        auto cmpMaskTensor = this->tempBuf_.template Get<uint8_t>();                                      \
        if constexpr (IsSameType<T, bfloat16_t>::value) {                                                 \
            auto xLocalFp32 = xLocal.template ReinterpretCast<float>();                                   \
            Cast(xLocalFp32, xLocal[this->tileLen_], RoundMode::CAST_NONE, curTileLen);                   \
            this->template ProcessOneTileTopK<float>(xLocalFp32, curTileLen, curTileKVal, coreTopKOffset, \
                                                     cmpMaskTensor, tileOffset);                          \
        } else if constexpr (IsSameType<T, half>::value) {                                                \
            auto xLocalHalf = xLocal[this->tileLen_];                                                     \
            if constexpr (largest) {                                                                      \
                this->CanonicalizeDataForLargest(curBuf, curTileLen);                                     \
            }                                                                                             \
            this->template ProcessOneTileTopK<half>(xLocalHalf, curTileLen, curTileKVal, coreTopKOffset,  \
                                                    cmpMaskTensor, tileOffset);                           \
        }                                                                                                 \
    } else {                                                                                              \
        PipeBarrier<PIPE_MTE2>();                                                                         \
    }

template <typename T, bool largest>
class RadixTopKWsI64 : public RadixTopKWs<T, largest> {
    using Base = RadixTopKWs<T, largest>;

public:
    __aicore__ inline RadixTopKWsI64(TPipe& tpipe, const RadixTopKTilingData& tilingData) : Base(tpipe, tilingData) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace);
    __aicore__ inline void SubProcess(uint64_t batchId);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReduceGlobalHistI64(LocalTensor<int32_t>& globalHist, LocalTensor<int64_t>& reduced);
    __aicore__ inline void FindBoundaryBinI64(const LocalTensor<int64_t>& reduced, int32_t roundId);
    __aicore__ inline void AddTileHistToTileTopKI64(const LocalTensor<int64_t>& reduced);
    __aicore__ inline bool Update(const int32_t& roundId);
    __aicore__ inline void CopyOutResultI64(const int32_t& count, const uint64_t& coreTopKOffset);

    template <AscendC::CMPMODE cmpMode, typename CT>
    __aicore__ inline void SubTopKAndCopyOutI64(LocalTensor<CT>& xLocal, const CT& boundaryNumber,
                                                const uint64_t& curTileLen, int32_t& curTileK, uint64_t& coreTopKOffset,
                                                LocalTensor<uint8_t>& cmpMaskTensor, const int64_t& tileOffset);

    template <typename CT>
    __aicore__ inline void ProcessOneTileTopK(LocalTensor<CT>& xLocalTyped, const uint64_t& curTileLen,
                                              int32_t& curTileK, uint64_t& coreTopKOffset,
                                              LocalTensor<uint8_t>& cmpMaskTensor, const int64_t& tileOffset);

    __aicore__ inline void TileTopK(const uint64_t& batchId, const uint64_t& blockOffset);

    // 掩码分批上限：Level-0 接口的 repeat 计数为 uint8（255 次/批），超过需分批下发
    static constexpr uint32_t MAX_REP_PER_BATCH = 255;
    /** @brief 连续域整批 += value（含分批与掩码设置/复位），见定义处前置条件 */
    __aicore__ inline void AddsContiguousBias(LocalTensor<int32_t> tensor, uint32_t repNum, const int32_t& value);
    /** @brief 交错视图偶 lane += evenBias、奇 lane += oddBias（含分批与掩码设置/复位） */
    __aicore__ inline void AddsInterleavedLaneBias(LocalTensor<int32_t> tensor, uint32_t totalRep,
                                                   const int32_t& evenBias, const int32_t& oddBias);
    /** @brief 显式 repeat 的原地 Cast int32→int64（负值符号扩展） */
    __aicore__ inline void CastInt32ToInt64Inplace(const LocalTensor<int64_t>& dst64, const LocalTensor<int32_t>& src32,
                                                   uint32_t repNum);

    // 向量版索引合并：符号偏置 Adds(-threshold) + Cast 符号扩展检测进位 + 双掩码 Adds
    // 主体加法（统一无分支，详见函数处注释）
    __aicore__ inline void VectorMergeI64(const LocalTensor<uint8_t>& cmpMaskTensor, const uint32_t& count,
                                          const int64_t& tileOffset, const uint64_t& curTileLen);

    GlobalTensor<int64_t> indexGm64_; /**< indices 输出的 int64 视图 */
};

/**
 * @brief 主流程入口：与 ws 变体一致
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::Process()
{
    for (uint64_t batchId = 0; batchId < this->batch_; batchId++) {
        SubProcess(batchId);
        SyncAll();
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);
}

/**
 * @brief 初始化：复用基类 ws 的 GM/WS 布局，仅额外增加 indexGm64_（indices 的 int64 视图），
 *        并按 int64 索引申请 outIndexBuf_
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices,
                                                        GM_ADDR workspace)
{
    this->InitWsGm(x, k, values, indices, workspace);
    indexGm64_.SetGlobalBuffer((__gm__ int64_t*)indices);
    this->InitBuffers(sizeof(int64_t)); // 唯一差异：outIndexBuf_ 按 int64 索引输出加倍
}

/**
 * @brief 对单个 batch 执行 int64 变体 Radix TopK（流程与 ws 变体一致）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::SubProcess(uint64_t batchId)
{
    this->involvedMask16_ = 0;
    this->andMask16_ = 0xC000;
    this->totalDefinitelyInTopK_ = 0;
    this->boundaryBin = 0;
    this->boundaryBinPrev = 1;
    this->globalHistBoundaryNum_ = this->tileNum_ * this->tileLen_;
    this->remainK_ = this->kValue_;
    uint64_t blockOffset = this->CalcBlockOffset(batchId);
    this->ClearTileTopKInWs();

    for (int32_t roundId = this->round_ - 1; roundId >= 0; roundId--) {
        uint64_t firstTileLen = this->tileNum_ > 1 ? this->tileLen_ : this->tailTileLen_;
        this->CopyIn(this->xBufPing_, firstTileLen, blockOffset);
        this->pingPongFlag = true;
        this->template CopyOut2Ws<true>(this->numValue_, this->blockIdx_ * this->numValue_);
        this->ClearHistInWs(roundId);

        SetFlag<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);

        for (int32_t repeatId = 0; repeatId <= this->tileNumBy3RepeatTimes_; repeatId++) {
            int32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB_BY3;
            int32_t endTileId = AscendC::Std::min(static_cast<int32_t>(startTileId + MAX_TILE_NUM_IN_UB_BY3),
                                                  static_cast<int32_t>(this->tileNum_));
            MTE3ToSSync();
            for (int32_t tileId = startTileId; tileId < endTileId; tileId++) {
                PINGPONG_TILE_BEGIN(tileId, blockOffset);

                this->CanonicalizeDataForLargest(curBuf, curTileLen);
                this->TwiddleInB16(curBuf, curTileLen);
                this->DoAndMask(curBuf, curTileLen);
                LocalTensor<int32_t> tileHist = this->tileHistAndTopKBuf_.template Get<int32_t>();
                this->CalcCumsumHistogram16(curBuf, roundId, tileHist, tileId - startTileId, MAX_TILE_NUM_IN_UB_BY3,
                                            MAX_TILE_NUM_IN_UB_BY3, curTileLen);

                PINGPONG_TILE_END(tileId);
            }
            SToMTE3Sync();
            LocalTensor<int32_t> tileHist = this->tileHistAndTopKBuf_.template Get<int32_t>();
            for (int32_t binMask = this->numValue_ - 1; binMask > 0; binMask--) {
                DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1),
                                                    static_cast<uint32_t>(sizeof(int32_t) * (endTileId - startTileId)),
                                                    0, 0, 0};
                uint32_t tileHistOffset = binMask * this->tileNum_ + repeatId * MAX_TILE_NUM_IN_UB_BY3;
                DataCopyPad(this->tileHistGm_[tileHistOffset], tileHist[(binMask - 1) * MAX_TILE_NUM_IN_UB_BY3],
                            tileHistOutParams);
            }
        }
        this->template CopyOut2Ws<false>(this->numValue_, this->blockIdx_ * this->numValue_);
        SyncAll();
        this->andMask16_ >>= BITS_PER_ROUND;
        if (Update(roundId))
            break;
        // 与 ub/ws 模板对齐：确保所有核完成 Update 内对 globalHistGm_ 的读取后再进入下一轮覆盖
        SyncAll();
    }
    SyncAll();
    TileTopK(batchId, blockOffset);
}

/**
 * @brief 跨核 globalHist 归约（int64 标量累加）：归约总量可达 sortLen > 2^31，
 *        向量 Add 不支持 int64；per-core 分段值 < 2^31，int32 读取安全
 * @param globalHist 各 core 分段的 int32 直方图（coreNum * numValue_ 个元素）
 * @param reduced 归约结果（numValue_ 个 int64）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::ReduceGlobalHistI64(LocalTensor<int32_t>& globalHist,
                                                                       LocalTensor<int64_t>& reduced)
{
    // 先完成全部读取再写回：reduced 的 int64 视图覆盖 int32 槽 0..7（core0/core1 分段），
    // 若边算边写会污染尚未读取的分段数据
    // 数组长度跟随 BITS_PER_ROUND（numValue_ 由 InitBaseParams 以同一常量初始化），
    // 不再硬编码 4：改 BITS_PER_ROUND 时数组自动跟随，不会栈越界
    constexpr int32_t NUM_BINS = 1 << BITS_PER_ROUND;
    int64_t sums[NUM_BINS];
    for (int32_t binMask = 0; binMask < NUM_BINS; binMask++) {
        int64_t sum = 0;
        for (int32_t coreId = 0; coreId < this->coreNum_; coreId++) {
            sum += static_cast<int64_t>(globalHist.GetValue(coreId * this->numValue_ + binMask));
        }
        sums[binMask] = sum;
    }
    for (int32_t binMask = 0; binMask < NUM_BINS; binMask++) {
        reduced.SetValue(binMask, sums[binMask]);
    }
}

/**
 * @brief 在 int64 全局直方图中找累积量首次 >= remainK_ 的 bin（与 ws 逻辑一致，仅比较值 int64 化）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::FindBoundaryBinI64(const LocalTensor<int64_t>& reduced,
                                                                      int32_t roundId)
{
    this->boundaryBin = -1;
    this->boundaryBinPrev = -1;
    for (int32_t binMask = this->numValue_ - 1; binMask >= 0; binMask--) {
        if (reduced.GetValue(binMask) >= static_cast<int64_t>(this->remainK_)) {
            this->boundaryBin = binMask;
            this->boundaryBinPrev = binMask + 1;
            break;
        }
    }
    this->involvedMask16_ |= static_cast<int16_t>(this->boundaryBin) << (roundId * BITS_PER_ROUND);
}

/**
 * @brief 将边界 bin 之前的确定性元素累加到 tileTopK（与 ws 逻辑一致，比较值 int64 化）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::AddTileHistToTileTopKI64(const LocalTensor<int64_t>& reduced)
{
    if (reduced.GetValue(this->boundaryBin) == static_cast<int64_t>(this->remainK_)) {
        this->AddTileHist2TileTopKInWs(this->boundaryBin * this->tileNum_);
        this->totalDefinitelyInTopK_ += static_cast<uint64_t>(reduced.GetValue(this->boundaryBin));
    } else if (this->boundaryBinPrev < this->numValue_) {
        this->AddTileHist2TileTopKInWs(this->boundaryBinPrev * this->tileNum_);
        this->totalDefinitelyInTopK_ += static_cast<uint64_t>(reduced.GetValue(this->boundaryBinPrev));
    }
}

/**
 * @brief 单轮 Update（与 ws 变体一致；归约/边界查找 int64 标量化）
 */
template <typename T, bool largest>
__aicore__ inline bool RadixTopKWsI64<T, largest>::Update(const int32_t& roundId)
{
    LocalTensor<int32_t> resTensor = this->outIndexBuf_.template Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;
    // 归约结果写 resTensor 前 4 个 int64（覆盖前 8 个 int32，即 core0/core1 分段，此后不再使用）
    LocalTensor<int64_t> reduced = this->outIndexBuf_.template Get<int64_t>();

    this->CopyInGlobalHist(globalHist);
    MTE2ToSSync();
    ReduceGlobalHistI64(globalHist, reduced);
    FindBoundaryBinI64(reduced, roundId);

    AddTileHistToTileTopKI64(reduced);

    this->remainK_ = this->kValue_ - this->totalDefinitelyInTopK_;
    LocalTensor<float> tileHistFp32 = resTensor.template ReinterpretCast<float>();
    if (roundId == 0 && this->remainK_ > 0) {
        // 基类为依赖基类，两阶段查找下必须用 this-> 限定（否则 g++ 报 must be available）
        this->HandleLastRoundBoundary(resTensor, tileHistFp32);
    }

    bool isLastRound = (this->remainK_ == 0 || roundId == 0);
    if (isLastRound) {
        this->WriteCoreTopKFromWs(resTensor, tileHistFp32);
    }

    return isLastRound;
}

/**
 * @brief 将 values（原逻辑）与 int64 indices 写回 GM 输出
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::CopyOutResultI64(const int32_t& count,
                                                                    const uint64_t& coreTopKOffset)
{
    LocalTensor<T> outValueLocal = this->outValueBuf_.template Get<T>();
    LocalTensor<int64_t> outIndexLocal = this->outIndexBuf_.template Get<int64_t>();

    if (count > 0) {
        DataCopyPad(this->valueGm_[coreTopKOffset], outValueLocal,
                    DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0});
    }
    SetFlag<HardEvent::MTE3_V>(this->eventIDMTE3ToVForValue_);
    if (count > 0) {
        DataCopyPad(
            indexGm64_[coreTopKOffset], outIndexLocal,
            DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int64_t) * count), 0, 0, 0});
    }
    SetFlag<HardEvent::MTE3_V>(this->eventIDMTE3ToVForIndex_);
}

/**
 * @brief 单 tile 内按 cmpMode 提取并输出：GatherMask 在 int32 域压缩值与索引，
 *        索引经 VectorMergeI64 向量拼装 int64 tileOffset（全 count 向量，无标量兜底）
 * 注：i64 路径不接收预生成索引（vecIndex/maxIndexLen）——tile 内索引一律在 VectorMergeI64 内
 *     经 CreateVecIndexPerf 现算，故这两个参数不再透传。
 */
template <typename T, bool largest>
template <AscendC::CMPMODE cmpMode, typename CT>
__aicore__ inline void RadixTopKWsI64<T, largest>::SubTopKAndCopyOutI64(
    LocalTensor<CT>& xLocal, const CT& boundaryNumber, const uint64_t& curTileLen, int32_t& curTileK,
    uint64_t& coreTopKOffset, LocalTensor<uint8_t>& cmpMaskTensor, const int64_t& tileOffset)
{
    if (curTileK <= 0)
        return;
    LocalTensor<T> outValueLocal = this->outValueBuf_.template Get<T>();
    LocalTensor<float> outValueLocalFp32 = outValueLocal.template ReinterpretCast<float>();

    uint64_t curTileLenAlign = Ops::Base::CeilAlign(curTileLen, REPEAT_SIZE / sizeof(CT));

    PipeBarrier<PIPE_V>();
    if constexpr (largest && cmpMode != AscendC::CMPMODE::EQ &&
                  (IsSameType<CT, float>::value || IsSameType<CT, half>::value)) {
        if (this->IsNaNValue(boundaryNumber)) {
            // 边界值本身是 NaN 时必须换 isNaN 掩码：x <= NaN 对有限值同样为 false，
            // NOT(LE) 会退化成全真并把 tile 前的有限元素错当入选者（与 base 版同因，详见
            // radix_top_k_common.h::SubTopKAndCopyOut 处注释）。
            Compare<CT, uint8_t>(cmpMaskTensor, xLocal, xLocal, AscendC::CMPMODE::EQ, curTileLenAlign);
            PipeBarrier<PIPE_V>();
            Not(cmpMaskTensor.template ReinterpretCast<int16_t>(), cmpMaskTensor.template ReinterpretCast<int16_t>(),
                static_cast<uint32_t>((curTileLenAlign >> 3) / sizeof(int16_t)));
        } else {
            // largest=true 的「必然入选」通过：NaN 的序键在 twiddle 域位于最大端（key ∈ [0xFF81, 0xFFFF]），
            // 直方图按 key 统计时已把 NaN 计入该 tile 的配额，但 x > b 对 NaN 恒为 false，
            // 于是 NaN 不被拷出、配额反被背景元素顶替（S1：被顶替槽数精确等于注入的 NaN 数）。
            // 这里不直接 Compares(GT)，而取 NOT(LE)：LE 对 NaN 恒为 false，逐位取反即为 true，
            // 掩码 = (x > b) ∪ isNaN，与 key 域的配额统计口径一致（逐位运算，全值域有效）。
            // largest=false 不加：key 域中 +NaN 不计入入选，且 torch 语义要求 NaN 不入选；
            // 整型 CT 无 NaN，走下面的直通分支。
            Compares<CT, uint8_t>(cmpMaskTensor, xLocal, boundaryNumber, AscendC::CMPMODE::LE, curTileLenAlign);
            PipeBarrier<PIPE_V>();
            Not(cmpMaskTensor.template ReinterpretCast<int16_t>(), cmpMaskTensor.template ReinterpretCast<int16_t>(),
                static_cast<uint32_t>((curTileLenAlign >> 3) / sizeof(int16_t)));
        }
    } else {
        PipeBarrier<PIPE_V>();
        Compares<CT, uint8_t>(cmpMaskTensor, xLocal, boundaryNumber, cmpMode, curTileLenAlign);
    }
    uint64_t rsvdCntV = 0;
    WaitFlag<HardEvent::MTE3_V>(this->eventIDMTE3ToVForValue_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        GatherMask(outValueLocalFp32, xLocal, cmpMaskTensor.ReinterpretCast<uint32_t>(), true, curTileLen, {1, 1, 0, 0},
                   rsvdCntV);
    } else {
        GatherMask(outValueLocal, xLocal, cmpMaskTensor.template ReinterpretCast<uint16_t>(), true, curTileLen,
                   {1, 1, 0, 0}, rsvdCntV);
    }
    WaitFlag<HardEvent::MTE3_V>(this->eventIDMTE3ToVForIndex_);
    VToSSync();
    // 非 EQ 通过按 tile 配额封顶：NaN 掩码并入后 rsvdCntV 可能略多于直方图统计出的配额
    // （例如混入 key 域未计入的 -NaN），封顶可避免拷出到 k 槽之外的越界写
    uint64_t passRsvdCnt = (cmpMode != AscendC::CMPMODE::EQ) ? rsvdCntV : static_cast<uint64_t>(curTileK);
    uint32_t count = static_cast<uint32_t>(
        passRsvdCnt < static_cast<uint64_t>(curTileK) ? passRsvdCnt : static_cast<uint64_t>(curTileK));
    SToVSync();
    // 索引 int64 合并：全 count 走向量符号偏置链（天然处理跨 2^32 进位）。
    // 原标量兜底（count<2048 时逐元素 for-j SetValue/GetValue）已删除：其 SS 逐元素
    // 访存在小 count 高频 tile 下成本远超向量链增量（前缀 CreateVecIndex/GatherMask
    // 两路径同价，不计入差值；向量链增量仅 count 相关的个位数 repeat）
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        PipeBarrier<PIPE_V>();
        Cast(outValueLocal, outValueLocalFp32, RoundMode::CAST_RINT, count);
    }
    PipeBarrier<PIPE_V>();
    VectorMergeI64(cmpMaskTensor, count, tileOffset, curTileLen);
    VToMTE3Sync();
    CopyOutResultI64(count, coreTopKOffset);
    curTileK -= count;
    coreTopKOffset += count;
}

/**
 * @brief 向量版 int64 索引合并：out64[j] = tileOffset + local[j]（含跨 2^32 进位）
 *
 * 符号偏置链（te_add 的 0x80000000 偏置在本场景（加数为常量基址）的精确化）：
 * threshold = min(2^32 - loBase, curTileLen)（标量，64 位计算，值域 [1, curTileLen]），
 * 进位 ⟺ l >= threshold。l - threshold ∈ (-curTileLen, curTileLen) 绝无 int32 溢出，
 * 负号真实体现在补码位模式上，经 Cast s32->s64 符号扩展免费提取：
 *
 *   ① CreateVecIndexPerf 生成 [0,curTileLen) 到 buf32[0]（GatherMask 读走后被覆盖）
 *   ② GatherMask 按 cmpMask 压缩 tile 内索引到 compact
 *   ③ Adds(compact, -threshold)：compact = l - thr ≡ loSum - 2^32*carry (mod 2^32)
 *   ④ Cast int32->int64：交错 [l-thr, ext]，ext = sign(l-thr) = 无进位 ? -1 : 0
 *   ⑤a Adds 偶lane += (uint32)(loBase+threshold) [0x5555 掩码]
 *       偶 lane 位模式 = (l-thr+loBase+thr) mod 2^32 = loSum（跨界 tile loBase+thr=2^32≡0；
 *       非跨界 thr=clamp 值，补偿同样精确成立——统一无分支）
 *   ⑤b Adds 奇lane += hiBase+1 [0xAAAA 掩码]
 *       无进位: -1+hiBase+1 = hiBase；进位: 0+hiBase+1 = hiBase+1（一条指令完成进位修正）
 *
 * 已排除方案（勿走回头路）：谓词 Adds（vadd 不读谓词寄存器，cmpmask 只被 vsel 消费）、
 * Compares GE(int32)（dav_c220 的 GE/LT 通道仅支持 half/float，int32 只有 EQ，恒 false）、
 * float 视图借位保序（denormal 被 FTZ 刷零，数值语义错乱）、CANN count 版 Compares
 * （repeat 数按 count*sizeof(T)/256 向下取整，尾部静默漏写）。
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::VectorMergeI64(const LocalTensor<uint8_t>& cmpMaskTensor,
                                                                  const uint32_t& count, const int64_t& tileOffset,
                                                                  const uint64_t& curTileLen)
{
    LocalTensor<int32_t> buf32 = this->outIndexBuf_.template Get<int32_t>();
    LocalTensor<int64_t> dst64 = this->outIndexBuf_.template Get<int64_t>();
    LocalTensor<int32_t> compact = buf32[this->tileLen_];

    // ①② 索引生成与紧凑压缩
    CreateVecIndexPerf<int32_t>(buf32, 0, static_cast<uint32_t>(curTileLen));
    PipeBarrier<PIPE_V>();
    uint64_t rsvdCompact = 0;
    GatherMask(compact, buf32, cmpMaskTensor.ReinterpretCast<uint32_t>(), true, static_cast<uint32_t>(curTileLen),
               {1, 1, 0, 0}, rsvdCompact);
    PipeBarrier<PIPE_V>();

    // 布局约束兜底：向量链 ④ 的 dst64 [0, 2*count) 与 compact [tileLen_, tileLen_+count)
    // 仅在 count <= tileLen_/2 时零重叠；count 更大（近全选场景 GT 全量，如 K 接近
    // sortLen）时两区域重叠，向量 repeat 内读写乱序会破坏 compact 未读部分
    // （近全选时实测大面积索引崩坏）——回退标量合并。
    // 前向标量安全：写 buf32[2j..2j+2) 恒低于读 buf32[tileLen_+j]（2j < tileLen_+j
    // 对 j < tileLen_ 恒成立），覆盖到的 compact[2j-tileLen_] 必为已读元素。
    // 注：小 count 不兜底——实测修好的向量链比标量 for-j 快约一倍（全程 172ms vs 305ms）
    if (count > this->tileLen_ / 2) {
        VToSSync();
        for (uint32_t j = 0; j < count; j++) {
            dst64.SetValue(j, tileOffset + static_cast<int64_t>(compact.GetValue(j)));
        }
        SToMTE3Sync();
        return;
    }

    int32_t loBase = static_cast<int32_t>(static_cast<uint32_t>(tileOffset));
    int32_t hiBase = static_cast<int32_t>(static_cast<uint64_t>(tileOffset) >> 32);
    // threshold = 2^32 - loBase（64 位计算防 uint32 回绕：loBase=0 时为 2^32，clamp 到 curTileLen）
    uint64_t thresholdU64 = 0x100000000ULL - static_cast<uint64_t>(static_cast<uint32_t>(tileOffset));
    int32_t threshold = static_cast<int32_t>(thresholdU64 > curTileLen ? curTileLen : thresholdU64);
    // ⑤a 的补偿标量按 uint32 取位模式（loBase 为 int32 负值时防 CPU 侧有符号溢出 UB）
    int32_t evenBias = static_cast<int32_t>(static_cast<uint32_t>(loBase) + static_cast<uint32_t>(threshold));

    // ③ 符号偏置（compact 连续域）：compact[l] += -threshold
    AddsContiguousBias(compact, (count + 63) / 64, -threshold);

    // ④ 原地 Cast int32→int64：负值符号扩展 hi=-1（进位检测）/ 非负 hi=0，交错成 [l-thr, ext]
    CastInt32ToInt64Inplace(dst64, compact, (count + 31) / 32);

    // ⑤ 主体加法：偶 lane 补偿 loSum，奇 lane 一步完成 hiBase+carry
    AddsInterleavedLaneBias(buf32, (2 * count + 63) / 64, evenBias, hiBase + 1);
}

/**
 * @brief ③ 连续域整批加常量：dst[0, repNum*64) += value（原地）
 *
 * 前置条件：
 *   - tensor 为连续排布、可原地写，容量 >= repNum*64 个 int32；
 *   - 走 Level-0 显式 repeat 而非 count 版 API（count 版 repeat 数按 count*sizeof(T)/256
 *     向下取整，尾部静默漏写）；rep 计数为 uint8，故内部按 255 repeat/批分批；
 *   - 每批显式 SetMaskNorm + SetVectorMask 设全 1 掩码（MASK_PLACEHOLDER 模式下不设会沿用
 *     寄存器残留），批结束后 ResetMask。
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::AddsContiguousBias(LocalTensor<int32_t> tensor, uint32_t repNum,
                                                                      const int32_t& value)
{
    for (uint32_t repOff = 0; repOff < repNum; repOff += MAX_REP_PER_BATCH) {
        uint32_t batchRep = AscendC::Std::min<uint32_t>(MAX_REP_PER_BATCH, repNum - repOff);
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0, 0xFFFFFFFFFFFFFFFFULL);
        Adds<int32_t, false>(tensor[repOff * 64], tensor[repOff * 64], value, AscendC::MASK_PLACEHOLDER,
                             static_cast<uint8_t>(batchRep), UnaryRepeatParams(1, 1, 8, 8));
        AscendC::ResetMask();
        PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief ⑤ 交错视图按 lane 奇偶分别加常量：偶 lane += evenBias、奇 lane += oddBias（原地）
 *        （对应 VectorMergeI64 的「偶 lane 补 loSum、奇 lane 补 hiBase+carry」）
 *
 * 前置条件：
 *   - tensor 是「每 2 个 int32 = 1 个 int64（lo, hi）」的交错视图，容量 >= totalRep*64 个 int32；
 *   - totalRep = ceil(2*count/64)；repeat 计数为 uint8，故内部按 255 repeat/批分批；
 *   - 每批两条 Adds 各自 SetMaskNorm + SetVectorMask（0x5555…/0xAAAA… lane 掩码），批结束后 ResetMask。
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::AddsInterleavedLaneBias(LocalTensor<int32_t> tensor,
                                                                           uint32_t totalRep, const int32_t& evenBias,
                                                                           const int32_t& oddBias)
{
    for (uint32_t repOff = 0; repOff < totalRep; repOff += MAX_REP_PER_BATCH) {
        uint32_t batchRep = AscendC::Std::min<uint32_t>(MAX_REP_PER_BATCH, totalRep - repOff);
        LocalTensor<int32_t> seg = tensor[repOff * 64];
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0, 0x5555555555555555ULL);
        Adds<int32_t, false>(seg, seg, evenBias, AscendC::MASK_PLACEHOLDER, static_cast<uint8_t>(batchRep),
                             UnaryRepeatParams(1, 1, 8, 8));
        PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0, 0xAAAAAAAAAAAAAAAAULL);
        Adds<int32_t, false>(seg, seg, oddBias, AscendC::MASK_PLACEHOLDER, static_cast<uint8_t>(batchRep),
                             UnaryRepeatParams(1, 1, 8, 8));
        AscendC::ResetMask();
        PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief ④ 原地 Cast int32→int64（显式 repeat，负值符号扩展 hi = -1）
 *
 * 前置条件：
 *   - src32 连续、dst64 与 src32 同一 buffer 的 int64 视图（就地扩展），repNum = ceil(count/32) 且
 *     <= 255（调用方已用 count <= tileLen_/2 保证）；
 *   - 每 repeat 转换 32 个元素，步长单位为 32 字节：dst 8×32B = 32 个 int64 连续，src 必须
 *     4×32B = 32 个 int32 连续（HALF_DEFAULT_REPEAT_STRIDE，与 C-API count 版一致；曾误用 8
 *     ⇒ src 每 repeat 跳 64 个 int32，count>32 的 tile 索引成对错乱）；
 *   - 不用 Cast 的 count 版：其 repeatTime 硬编码为 1，count>32 时静默截断（只写 32 个）。
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::CastInt32ToInt64Inplace(const LocalTensor<int64_t>& dst64,
                                                                           const LocalTensor<int32_t>& src32,
                                                                           uint32_t repNum)
{
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0, 0xFFFFFFFFFFFFFFFFULL);
    uint64_t castMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0ULL};
    Cast<int64_t, int32_t>(dst64, src32, RoundMode::CAST_NONE, castMask, static_cast<uint8_t>(repNum),
                           UnaryRepeatParams(1, 1, 8, 4));
    AscendC::ResetMask();
    PipeBarrier<PIPE_V>();
}

/**
 * @brief 处理单个 tile 的 TopK 提取（与 base 逻辑一致；tileOffset 参数 int64 化）
 */
template <typename T, bool largest>
template <typename CT>
__aicore__ inline void RadixTopKWsI64<T, largest>::ProcessOneTileTopK(LocalTensor<CT>& xLocalTyped,
                                                                      const uint64_t& curTileLen, int32_t& curTileK,
                                                                      uint64_t& coreTopKOffset,
                                                                      LocalTensor<uint8_t>& cmpMaskTensor,
                                                                      const int64_t& tileOffset)
{
    CT boundaryNumber = this->template ExtractBoundaryNumber<CT>();
    if constexpr (!largest) {
        boundaryNumber = static_cast<CT>(-static_cast<float>(boundaryNumber));
        this->template SubTopKAndCopyOutI64<AscendC::CMPMODE::LT, CT>(xLocalTyped, boundaryNumber, curTileLen, curTileK,
                                                                      coreTopKOffset, cmpMaskTensor, tileOffset);
    } else {
        this->template SubTopKAndCopyOutI64<AscendC::CMPMODE::GT, CT>(xLocalTyped, boundaryNumber, curTileLen, curTileK,
                                                                      coreTopKOffset, cmpMaskTensor, tileOffset);
    }
    this->template SubTopKAndCopyOutI64<AscendC::CMPMODE::EQ, CT>(xLocalTyped, boundaryNumber, curTileLen, curTileK,
                                                                  coreTopKOffset, cmpMaskTensor, tileOffset);
}

/**
 * @brief int64 变体 TileTopK：从 WS 分段读取 tileTopK，提取 TopK 元素并输出（与 ws 逻辑一致）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWsI64<T, largest>::TileTopK(const uint64_t& batchId, const uint64_t& blockOffset)
{
    LocalTensor<int32_t> tmpLocal = this->tempBuf_.template Get<int32_t>();
    uint64_t coreTopKOffset = this->CopyInCoreTopK(tmpLocal);
    uint64_t batchIndexOffset = batchId * this->kValue_;
    coreTopKOffset += batchIndexOffset;

    LocalTensor<int32_t> tileTopK = this->tileHistAndTopKBuf_.template Get<int32_t>();
    uint64_t firstId = this->FindFirstTileWithTopK(tileTopK);

    uint64_t firstLen = (firstId == this->tileNum_ - 1) ? this->tailTileLen_ : this->tileLen_;
    this->CopyIn(this->xBufPing_, firstLen, blockOffset + firstId * this->tileLen_);
    // 与 ws 变体不同，这里不调用 CreateVecIndex4TopK、也不取 vecIndex 句柄：i64 路径的 tile 内索引
    // 一律在 VectorMergeI64 内现算（CreateVecIndex4TopK 会把索引写到 tileHistAndTopKBuf_ 偏移
    // tileNumAlign_ 处，而 tileNum_ > MAX_TILE_NUM_IN_UB 的多段 regime 在 i64 tiling 下恒成立，
    // 会越界写出 6144 元素 buffer）。

    this->InitTopKEvent();
    uint64_t startRepeatId = firstId / MAX_TILE_NUM_IN_UB;
    for (uint64_t repeatId = startRepeatId; repeatId <= this->tileNumRepeatTimes_; repeatId++) {
        uint64_t startTileId = repeatId * MAX_TILE_NUM_IN_UB;
        uint64_t endTileId = AscendC::Std::min(static_cast<uint64_t>(startTileId + MAX_TILE_NUM_IN_UB),
                                               static_cast<uint64_t>(this->tileNum_));
        if (repeatId > startRepeatId) {
            // 末段按剩余 tile 数截断：固定 6144 会使末核读出 workspace 末尾（地址布局相关、
            // 时好时坏的多段潜伏越界；base ws 因 S<=2^31 恒单段不会触发）
            uint32_t segLen = static_cast<uint32_t>(
                AscendC::Std::min<uint64_t>(MAX_TILE_NUM_IN_UB, this->tileNum_ - repeatId * MAX_TILE_NUM_IN_UB));
            this->CopyTileTopKWs2Ub(tileTopK, repeatId * MAX_TILE_NUM_IN_UB, segLen);
            MTE2ToSSync();
        } else {
            startTileId = firstId;
        }
        for (uint64_t tileId = startTileId; tileId < endTileId; tileId++) {
            PINGPONG_TILE_BEGIN(tileId, blockOffset);

            uint64_t tileBaseOffset = blockOffset - batchId * this->sortLen_;
            int64_t tileOffset = static_cast<int64_t>(tileBaseOffset + tileId * this->tileLen_);

            int32_t curTileK = tileTopK.GetValue(tileId - repeatId * MAX_TILE_NUM_IN_UB);
            PipeBarrier<PIPE_V>();
            PROCESS_ONE_TILE_TOPK_OR_SKIP_I64(curTileK);

            PINGPONG_TILE_END(tileId);
        }
    }
    this->FinishTopKEvent();
}

} // namespace RadixTopK
#endif // RADIX_TOP_K_WS_I64_H
