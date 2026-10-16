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
 * \file radix_top_k_common.h
 * \brief Radix TopK base kernel common state and utility methods
 */

#ifndef RADIX_TOPK_COMMON_H
#define RADIX_TOPK_COMMON_H

#include "radix_top_k_struct.h"
#include "radix_top_k_utils.h"

namespace RadixTopK {
using namespace AscendC;

/**
 * @brief PingPong tile 循环前导：选取 curBuf/nextBuf，预取下一 tile，WaitFlag
 *        展开后定义 curBuf, nextBuf, curTileLen 三个局部变量供后处理使用。
 */
#define PINGPONG_TILE_BEGIN(tileId, blockOffset)                                                        \
    uint64_t curTileLen = (tileId == this->tileNum_ - 1) ? this->tailTileLen_ : this->tileLen_;         \
    TBuf<TPosition::VECIN>& curBuf = this->pingPongFlag ? this->xBufPing_ : this->xBufPong_;            \
    TBuf<TPosition::VECIN>& nextBuf = this->pingPongFlag ? this->xBufPong_ : this->xBufPing_;           \
    do {                                                                                                \
        if (tileId < this->tileNum_ - 1) {                                                              \
            this->CopyIn(nextBuf, (tileId == this->tileNum_ - 2) ? this->tailTileLen_ : this->tileLen_, \
                         blockOffset + (tileId + 1) * this->tileLen_);                                  \
        }                                                                                               \
        WaitFlag<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);                                         \
    } while (0)

/**
 * @brief PingPong tile 循环结尾：SetFlag 预取下一 tile，翻转 pingPongFlag
 */
#define PINGPONG_TILE_END(tileId)                                  \
    do {                                                           \
        if (tileId < this->tileNum_ - 1) {                         \
            SetFlag<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_); \
        }                                                          \
        this->pingPongFlag = !this->pingPongFlag;                  \
    } while (0)

/**
 * @brief 处理单个 tile 的 TopK：xLocal/cmpMaskTensor 初始化、bf16/half 分发、
 *        若 curTileK<=0 则推送 MTE3 空事件跳过该 tile
 * @param curTileKVal 当前 tile 的 TopK 元素数
 * @param vecIndexExpr 预生成索引序列表达式（UB/WS 变体不同）
 * 前置条件：curBuf, coreTopKOffset, maxIndexLen, tileOffset, curTileLen 在作用域内
 * @note 提取相与计数相的 NaN 规范化口径**不对称是有意的**（勿在此处"补齐"）：
 *       计数相 CanonicalizeDataForLargest 对 bf16/half、largest 两个方向都做，目的是让 ±NaN 落到
 *       同一个 twiddle 键（largest=true -> 0xFFFF 计入配额；largest=false -> 0x0000 不计入）；
 *       提取相只需保证"掩码口径与计数一致"——bf16 分支把原始数据 Cast 成 fp32 后比较，
 *       Compares(LE, ±NaN) 恒 false ⇒ 取非后恒 true，符号/payload 不影响掩码；largest=false 走
 *       Compares(LT)，NaN 参与比较恒 false ⇒ 天然不入选；元素本身为 NaN 的情形由 IsNaNValue +
 *       Compare(x,x,EQ) 取非得来的 isNaN 掩码覆盖（同样与符号/payload 无关）。
 */
#define PROCESS_ONE_TILE_TOPK_OR_SKIP(curTileKVal, vecIndexExpr)                                                       \
    if (curTileKVal > 0) {                                                                                             \
        auto xLocal = curBuf.template Get<T>();                                                                        \
        auto cmpMaskTensor = this->tempBuf_.template Get<uint8_t>();                                                   \
        auto vecIndex = (vecIndexExpr);                                                                                \
        if constexpr (IsSameType<T, bfloat16_t>::value) {                                                              \
            auto xLocalFp32 = xLocal.template ReinterpretCast<float>();                                                \
            Cast(xLocalFp32, xLocal[this->tileLen_], RoundMode::CAST_NONE, curTileLen);                                \
            this->template ProcessOneTileTopK<float>(xLocalFp32, curTileLen, curTileKVal, coreTopKOffset, maxIndexLen, \
                                                     cmpMaskTensor, vecIndex, tileOffset);                             \
        } else if constexpr (IsSameType<T, half>::value) {                                                             \
            auto xLocalHalf = xLocal[this->tileLen_];                                                                  \
            if constexpr (largest) {                                                                                   \
                this->CanonicalizeDataForLargest(curBuf, curTileLen);                                                  \
            }                                                                                                          \
            this->template ProcessOneTileTopK<half>(xLocalHalf, curTileLen, curTileKVal, coreTopKOffset, maxIndexLen,  \
                                                    cmpMaskTensor, vecIndex, tileOffset);                              \
        }                                                                                                              \
    } else {                                                                                                           \
        PipeBarrier<PIPE_MTE2>();                                                                                      \
    }

/**
 * @brief Radix TopK 基类，包含 UB/WS 两种变体共用的核心计算逻辑
 * @tparam T 输入数据类型（half 或 bfloat16_t）
 * @tparam largest 是否求最大 k 值
 */
template <typename T, bool largest>
class RadixTopKBaseKernel {
public:
    __aicore__ inline RadixTopKBaseKernel(TPipe& tpipe, const RadixTopKTilingData& tilingData)
        : tpipe_(tpipe), tiling_(tilingData)
    {}

protected:
    __aicore__ inline void InitBaseParams();

    __aicore__ inline void CopyIn(TBuf<TPosition::VECIN>& xBuf, const uint64_t& dataNum, const uint64_t& xOffset);

    __aicore__ inline void TwiddleInB16(TBuf<TPosition::VECIN>& xBuf, const uint64_t& curTileLen);

    __aicore__ inline void DoAndMask(TBuf<TPosition::VECIN>& xBuf, const uint64_t& curTileLen);

    __aicore__ inline void CalcCumsumHistogram16(TBuf<TPosition::VECIN>& xBuf, const int32_t& roundId,
                                                 const LocalTensor<int32_t>& tileHist, const int32_t& tileIdValue,
                                                 const int32_t& tileHistBinStride, const int32_t& tileHistBinOffset,
                                                 const uint64_t& curTileLen);

    __aicore__ inline void CanonicalizeDataForLargest(TBuf<TPosition::VECIN>& curBuf, const uint64_t& curTileLen);

    template <bool isInit>
    __aicore__ inline void CopyOut2Ws(const uint64_t& maskLen, const uint64_t& gmOffset);

    __aicore__ inline uint32_t CreateVecIndex4TopK(uint32_t maxIndexLen, LocalTensor<int32_t>& vecIndex);

    template <AscendC::CMPMODE cmpMode, typename CT>
    __aicore__ inline void SubTopKAndCopyOut(LocalTensor<CT>& xLocal, const CT& boundaryNumber,
                                             const uint64_t& curTileLen, int32_t& curTileK, uint64_t& coreTopKOffset,
                                             const uint32_t& maxIndexLen, LocalTensor<uint8_t>& cmpMaskTensor,
                                             LocalTensor<int32_t>& vecIndex, int32_t tileOffset);

    __aicore__ inline void CopyOutResult(const int32_t& count, const uint64_t& coreTopKOffset);

    __aicore__ inline uint64_t CopyInCoreTopK(LocalTensor<int32_t>& tmpLocal);

    __aicore__ inline void InitTopKEvent();

    __aicore__ inline void FinishTopKEvent();

    template <typename CT>
    __aicore__ inline void ProcessOneTileTopK(LocalTensor<CT>& xLocalTyped, const uint64_t& curTileLen,
                                              int32_t& curTileK, uint64_t& coreTopKOffset, const uint32_t& maxIndexLen,
                                              LocalTensor<uint8_t>& cmpMaskTensor, LocalTensor<int32_t>& vecIndex,
                                              int32_t tileOffset);

    __aicore__ inline void ReduceGlobalHist(LocalTensor<int32_t>& globalHist);

    __aicore__ inline void FindBoundaryBin(const LocalTensor<int32_t>& globalHist, int32_t roundId);

    __aicore__ inline void CopyInGlobalHist(LocalTensor<int32_t>& globalHist);

    __aicore__ inline void CopyOutBoundaryBinCumSum(LocalTensor<int32_t>& resTensor);

    /** @brief 前驱 core 边界 bin 累计元素数（int64 累加：总量可达 sortLen > 2^31；int32 形状下同样安全） */
    __aicore__ inline void ComputeCumSumPrev(LocalTensor<int32_t>& resTensor, int64_t& cumSumValuePrev);

    __aicore__ inline void CopyOutCoreTopK(LocalTensor<int32_t>& resTensor, int32_t totalTileTopKInCore);

    template <typename CT>
    __aicore__ inline CT ExtractBoundaryNumber();

    /**
     * @brief 边界值是否为 NaN（位模式判定，不做浮点自比较以免触发浮点相等告警）
     *        NaN ⟺ 指数位全 1 且尾数非 0（对 ±NaN、任意 payload 都成立）
     */
    template <typename CT>
    __aicore__ inline bool IsNaNValue(const CT& value);

    __aicore__ inline uint64_t CalcBlockOffset(uint64_t batchId);

protected:
    TPipe& tpipe_;
    const RadixTopKTilingData& tiling_;

    uint64_t blockIdx_;             /**< 当前 core ID */
    uint64_t xAlign_;               /**< x 数据对齐粒度（BLOCK_SIZE / sizeof(T)） */
    uint64_t tileLen_;              /**< 每个 tile 的数据长度 */
    uint64_t tileNum_;              /**< 当前 core 处理的 tile 数量 */
    uint64_t formerCoreNum_;        /**< 主核数量 */
    uint64_t totalTileNum_;         /**< 总 tile 数 */
    uint64_t formerTileNum_;        /**< 主核 tile 数 */
    uint64_t tailTileNum_;          /**< 尾核 tile 数 */
    uint64_t formerTileLen_;        /**< 主核 tile 长度 */
    uint64_t tailTileLen_;          /**< 尾核 tile 长度（最后一个 tile 可能不足 tileLen_） */
    uint64_t tileNumAlign_;         /**< tileNum_ 对齐到 8 个 int32 的值 */
    uint64_t round_;                /**< Radix 总轮数（16-bit 类型 = 8 轮，每轮 2 bit） */
    uint32_t numValue_;             /**< 每轮 bin 数（= 4，即 2^BITS_PER_ROUND） */
    int16_t involvedMask16_;        /**< 累积的 Radix 前缀 mask */
    int16_t andMask16_;             /**< 当前轮提取位掩码，每轮右移 BITS_PER_ROUND 位 */
    int32_t boundaryBin;            /**< 边界 bin 编号（0~3） */
    int32_t boundaryBinPrev;        /**< 边界 bin 的前一个 bin 编号 */
    int32_t globalHistBoundaryNum_; /**< 边界 bin 对应的全局元素数 */

    int32_t coreNum_;                /**< 总 core 数 */
    uint64_t batch_;                 /**< batch 数 */
    uint64_t sortLen_;               /**< 排序轴长度 */
    uint64_t kValue_;                /**< TopK 值 k */
    uint64_t totalDefinitelyInTopK_; /**< 已确认在 TopK 中的元素总数 */
    uint64_t remainK_;               /**< 当前剩余需取出的元素数 */

    event_t eventIDMTE2ToVForX_;            /**< MTE2→V 同步事件 ID（数据搬入） */
    event_t eventIDMTE3ToVForValue_;        /**< MTE3→V 同步事件 ID（value 输出） */
    event_t eventIDMTE3ToVForIndex_;        /**< MTE3→V 同步事件 ID（index 输出） */
    bool pingPongFlag;                      /**< Ping-Pong 缓冲标志 */
    TBuf<TPosition::VECIN> xBufPing_;       /**< Ping 输入数据缓冲 */
    TBuf<TPosition::VECIN> xBufPong_;       /**< Pong 输入数据缓冲 */
    TBuf<TPosition::VECOUT> outValueBuf_;   /**< 输出 value 缓冲 */
    TBuf<TPosition::VECOUT> outIndexBuf_;   /**< 输出 index 缓冲 */
    TBuf<TPosition::VECOUT> globalHistBuf_; /**< 全局直方图缓冲 */

    GlobalTensor<T> xGm_;                       /**< 输入 x GM Tensor */
    GlobalTensor<T> valueGm_;                   /**< 输出 values GM Tensor */
    GlobalTensor<int32_t> kGm_;                 /**< 输入 k GM Tensor */
    GlobalTensor<int32_t> indexGm_;             /**< 输出 indices GM Tensor */
    GlobalTensor<int32_t> globalHistGm_;        /**< Workspace globalHist 段 GM Tensor */
    GlobalTensor<int32_t> boundaryBinCumSumGm_; /**< Workspace boundaryBinCumSum 段 GM Tensor */
    GlobalTensor<int32_t> coreTopKGm_;          /**< Workspace coreTopK 段 GM Tensor */
};

/**
 * @brief 初始化基础参数，从 TilingData 读取核分配及 tile 切分信息
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::InitBaseParams()
{
    blockIdx_ = GetBlockIdx();
    xAlign_ = BLOCK_SIZE / sizeof(T);
    coreNum_ = tiling_.coreNum;
    formerCoreNum_ = tiling_.formerCoreNum;
    totalTileNum_ = tiling_.totalTileNum;
    formerTileNum_ = tiling_.formerTileNum;
    tailTileNum_ = tiling_.tailTileNum;
    formerTileLen_ = tiling_.formerTileLen;
    tailTileLen_ = blockIdx_ != coreNum_ - 1 ? formerTileLen_ : tiling_.tailTileLen;
    batch_ = tiling_.batch;
    sortLen_ = tiling_.sortLen;
    kValue_ = tiling_.kValue;
    tileNum_ = blockIdx_ < formerCoreNum_ ? formerTileNum_ : tailTileNum_;
    tileLen_ = formerTileLen_;
    tileNumAlign_ = Ops::Base::CeilAlign(tileNum_, BLOCK_SIZE / sizeof(int32_t));
    numValue_ = 1 << BITS_PER_ROUND;
    round_ = sizeof(T) * BYTE_SIZE / BITS_PER_ROUND;
    eventIDMTE2ToVForX_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
}

/**
 * @brief 将输入数据从 GM 搬入 UB（xBuf）
 * @param xBuf 目标 UB 缓冲区
 * @param dataNum 搬运数据量
 * @param xOffset 在 xGm 上的偏移量
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CopyIn(TBuf<TPosition::VECIN>& xBuf, const uint64_t& dataNum,
                                                               const uint64_t& xOffset)
{
    LocalTensor<T> xLocal = xBuf.Get<T>();
    DataCopyPad(xLocal[tileLen_], xGm_[xOffset],
                DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * dataNum),
                                  static_cast<uint32_t>((sortLen_ - dataNum) * sizeof(T)), 0, 0},
                DataCopyPadExtParams<T>{true, 0, static_cast<uint8_t>((xAlign_ - dataNum % xAlign_) % xAlign_), 0});
}

/**
 * @brief 对 half/bf16 做 bit 变换，转换为可按无符号整数比较的序
 *        正数：最高位取反（0→1），变为 0x8000+value
 *        负数：所有位取反，变为 ~value
 * @param xBuf 输入数据缓冲区
 * @param curTileLen 当前 tile 数据长度
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::TwiddleInB16(TBuf<TPosition::VECIN>& xBuf,
                                                                     const uint64_t& curTileLen)
{
    LocalTensor<T> xLocalB16 = xBuf.Get<T>();
    LocalTensor<int16_t> xLocalInt16 = xLocalB16[tileLen_].template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> signMaskTensor = outValueBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensor = outIndexBuf_.Get<int16_t>();

    uint64_t calcLen = curTileLen * (sizeof(T) / sizeof(int16_t));
    uint64_t curTileLenAlign = Ops::Base::CeilAlign(curTileLen, BLOCK_SIZE / sizeof(int16_t));
    LocalTensor<int16_t> temp1 = tempTensor[curTileLenAlign];
    PipeBarrier<PIPE_V>();
    ShiftRight(tempTensor, xLocalInt16, (int16_t)15, curTileLen);
    PipeBarrier<PIPE_V>();
    Duplicate<int16_t>(signMaskTensor, XOR_OP_VALUE_16, curTileLen);
    PipeBarrier<PIPE_V>();
    Or(tempTensor, tempTensor, signMaskTensor, calcLen);

    PipeBarrier<PIPE_V>();
    And(temp1, xLocalInt16, tempTensor, calcLen);
    PipeBarrier<PIPE_V>();
    Or(xLocalInt16, xLocalInt16, tempTensor, calcLen);
    PipeBarrier<PIPE_V>();
    Not(temp1, temp1, calcLen);
    PipeBarrier<PIPE_V>();
    And(xLocalInt16, xLocalInt16, temp1, calcLen);
    if constexpr (!largest && (sizeof(T) == sizeof(int16_t))) {
        // largest=false 的降序由键域取补承担：先做升序键（上面的 twiddle），再整体取补。
        // 对全部 65536 个 16 位模式恒有 ~twiddle(x) == twiddle(bits(-x))，故除 NaN 外本步
        // 与「先取反再 twiddle」逐位一致，序键语义不变；仅 NaN 受影响：
        // CanonicalizeDataForLargest 已把 ±NaN 规范化为同一个正 NaN（实测 fp16 与 bf16 都是
        // 0x7FFF，与输入 NaN 的符号无关），其升序键为 0xFFFF，取补后为 0x0000，低于 -Inf 的
        // 0x007F(bf16)/0x03FF(fp16)，落到序键最底端 ⇒ 在计数相不被计入 k 槽、提取相也不会被选中。
        PipeBarrier<PIPE_V>();
        Not(xLocalInt16, xLocalInt16, calcLen);
    }
}

/**
 * @brief 用当前轮的位掩码提取数据的 2 bit
 * @param xBuf 经过 TwiddleInB16 变换后的数据缓冲区
 * @param curTileLen 当前 tile 数据长度
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::DoAndMask(TBuf<TPosition::VECIN>& xBuf,
                                                                  const uint64_t& curTileLen)
{
    LocalTensor<int16_t> xLocalInt16 = xBuf.Get<int16_t>();
    uint64_t calcLen = curTileLen * (sizeof(T) / sizeof(int16_t));
    LocalTensor<int32_t> maskTensor = outValueBuf_.Get<int32_t>();
    PipeBarrier<PIPE_V>();
    Duplicate<int16_t>(maskTensor.template ReinterpretCast<int16_t>(), andMask16_, curTileLen);
    PipeBarrier<PIPE_V>();
    And(xLocalInt16[tileLen_], xLocalInt16[tileLen_], maskTensor.template ReinterpretCast<int16_t>(), calcLen);
}

/**
 * @brief 16-bit Radix 直方图统计，int16 打包为 int32 逐对比较
 * @param xBuf 输入数据缓冲区
 * @param roundId 当前轮 ID（7~0）
 * @param tileHist tile 直方图输出
 * @param tileIdValue 当前 tile ID
 * @param tileHistBinStride tileHist 各 bin 的 stride
 * @param tileHistBinOffset tileHist 各 bin 的偏移
 * @param curTileLen 当前 tile 数据长度
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CalcCumsumHistogram16(
    TBuf<TPosition::VECIN>& xBuf, const int32_t& roundId, const LocalTensor<int32_t>& tileHist,
    const int32_t& tileIdValue, const int32_t& tileHistBinStride, const int32_t& tileHistBinOffset,
    const uint64_t& curTileLen)
{
    LocalTensor<int16_t> xLocalInt16 = xBuf.Get<int16_t>();
    LocalTensor<int32_t> xLocalInt32 = xLocalInt16.ReinterpretCast<int32_t>();
    LocalTensor<float> xLocalFp32 = xLocalInt16.ReinterpretCast<float>();
    PipeBarrier<PIPE_V>();
    Cast(xLocalFp32, xLocalInt16[tileLen_], RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
    Cast(xLocalInt32, xLocalFp32, RoundMode::CAST_RINT, curTileLen);

    LocalTensor<half> tempHalf = outValueBuf_.Get<half>();
    LocalTensor<uint8_t> cmpMaskTensor = outIndexBuf_.Get<uint8_t>();
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();

    int32_t cumSumOne = 0;
    for (int16_t binMask = numValue_ - 1; binMask > 0; binMask--) {
        int32_t involvedMaskTemp = static_cast<int32_t>(involvedMask16_ |
                                                        static_cast<int16_t>(binMask << (roundId * BITS_PER_ROUND)));
        uint32_t curTileLenAlign = Ops::Base::CeilAlign(curTileLen, REPEAT_SIZE / sizeof(int32_t));
        PipeBarrier<PIPE_V>();
        Compares<int32_t, uint8_t>(cmpMaskTensor, xLocalInt32, involvedMaskTemp, AscendC::CMPMODE::EQ, curTileLenAlign);
        uint64_t rsvdCnt = 0;
        PipeBarrier<PIPE_V>();
        GatherMask(tempHalf, tempHalf, cmpMaskTensor.ReinterpretCast<uint16_t>(), true, curTileLen, {1, 1, 0, 0},
                   rsvdCnt);
        VToSSync();
        cumSumOne += rsvdCnt;
        tileHist[static_cast<int32_t>(binMask * tileHistBinStride - tileHistBinOffset)].SetValue(tileIdValue,
                                                                                                 cumSumOne);
        globalHist.SetValue(binMask, globalHist.GetValue(binMask) + cumSumOne);
    }
}

/**
 * @brief ±NaN 规范化（两个方向都要做）：把 ±NaN 统一变成同一个正 NaN（16 位类型的原域），
 *        使 TwiddleInB16 键域对 NaN 的记账与提取相的掩码口径一致；有限值与 ±Inf 逐位不变
 *        （fp16/bf16 → fp32 → ×1 → 回 16 位是精确往返）。
 *          - largest=false：规范化后 NaN 键为 0xFFFF，经 TwiddleInB16 末尾的键域取补落 0x0000
 *            （低于 -Inf 的 0x007F/0x03FF）⇒ 计数相不计入 k 槽、提取相 <LT> 也不匹配
 *            ⇒ NaN 不入选（与 torch 语义一致）。降序本身由该键域取补承担，故此处不取反。
 *          - largest=true：规范化后 NaN 键为 0xFFFF（最大端）⇒ 计数相把 NaN 计入该 tile 的
 *            k 配额，与提取相 NOT(LE) = (x > b) ∪ isNaN 口径一致 ⇒ NaN 必入选（与 torch 一致）。
 *            **缺了这一步会丢 −NaN**：−NaN 原样 twiddle 后键为 ~0xFE00=0x01FF(fp16) /
 *            ~0xFFC0=0x003F(bf16)，落在 −Inf 之下 ⇒ 直方图不计入配额，但提取掩码仍匹配
 *            ⇒ rsvdCntV 超出该 tile 配额，被 SubTopKAndCopyOut 的 min(rsvdCntV, curTileK)
 *            截断（GatherMask 按 lane 序保留靠前的匹配）⇒ −NaN 槽丢空、背景值顶替。
 *            实测（S=1.2e7+4 / k=1001 / 尾块 8 个 ±NaN / largest=true）：fp16 只中 +NaN 半边、
 *            −NaN 0/4、values 多重集错位 410/1001；bf16 同类（−NaN 0/4，错位 8/1001，量化粗
 *            使错位值多为等值）。全 +NaN 输入（8/8、错位 0）与 largest=false（NaN 0 入选、
 *            错位 0）是对照。
 * @param curBuf 当前数据缓冲区
 * @param curTileLen 当前 tile 数据长度
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CanonicalizeDataForLargest(TBuf<TPosition::VECIN>& curBuf,
                                                                                   const uint64_t& curTileLen)
{
    LocalTensor<T> xCanon = curBuf.Get<T>();
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        // bf16：Muls 不支持 bf16（编译报错），只能借 fp32 域做规范化 —— 实测该往返把 ±NaN 归成
        // 0x7FFF，±Inf/±0/次正规码无损。
        // 注意这里是乘 +1 而非 −1：降序由 TwiddleInB16 末尾的键域取补承担，本步只需把 ±NaN
        // 统一成同一个正 NaN（否则 −NaN 取补后会落到序键顶端而被误选，即 q7 缺陷）。
        // （坑位记录：当年 bf16 的取反是「把位模式 reinterpret 成 half 再乘 −1」，
        //   |x|≥2^121 与 ±Inf/NaN 会落进 half 的非规格区退化为 NaN ⇒ D64/S2。）
        LocalTensor<float> xFp32 = outValueBuf_.Get<float>();
        Cast(xFp32, xCanon[tileLen_], RoundMode::CAST_NONE, curTileLen);
        PipeBarrier<PIPE_V>();
        Muls(xFp32, xFp32, (float)1, curTileLen);
        PipeBarrier<PIPE_V>();
        Cast(xCanon[tileLen_], xFp32, RoundMode::CAST_RINT, curTileLen);
    } else if constexpr (IsSameType<T, half>::value) {
        // fp16：原生同类型 Muls(x, 1)。实测：与「Cast→fp32→Muls(+1)→Cast 回」对
        // ±NaN/±Inf/±0/次正规码等全部边界输入逐位一致（±NaN 都规范化成 0x7FFF）。
        // 省掉 2 条向量指令与 2 次 PipeBarrier，也不再需要 fp32 暂存。
        Muls(xCanon[tileLen_], xCanon[tileLen_], (half)1, curTileLen);
    }
}

/**
 * @brief 将 globalHist 写入 workspace，isInit=true 时记录边界 bin 数并清零
 * @tparam isInit true: 记录边界值并清零；false: 直接写入
 * @param maskLen 数据长度（numValue_）
 * @param gmOffset 在 globalHistGm 上的偏移量
 */
template <typename T, bool largest>
template <bool isInit>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CopyOut2Ws(const uint64_t& maskLen, const uint64_t& gmOffset)
{
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();
    if constexpr (isInit) {
        globalHistBoundaryNum_ = globalHist.GetValue(boundaryBin);
        if (boundaryBin < numValue_ - 1) {
            globalHistBoundaryNum_ -= globalHist.GetValue(boundaryBinPrev);
        }
        PipeBarrier<PIPE_V>();
        SToVSync();
        Duplicate<int32_t>(globalHist, 0, maskLen);
        VToMTE3Sync();
    } else {
        SToMTE3Sync();
    }
    DataCopyPad(globalHistGm_[gmOffset], globalHist,
                DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * maskLen), 0, 0, 0});
}

/**
 * @brief 为 TileTopK 阶段创建预生成索引序列
 * @param maxIndexLen 最大索引长度（不够 512 时置 0）
 * @param vecIndex 输出索引 Tensor
 * @return 实际的 maxIndexLen
 */
template <typename T, bool largest>
__aicore__ inline uint32_t RadixTopKBaseKernel<T, largest>::CreateVecIndex4TopK(uint32_t maxIndexLen,
                                                                                LocalTensor<int32_t>& vecIndex)
{
    if (tileLen_ < maxIndexLen) {
        maxIndexLen = tileLen_;
    } else if (maxIndexLen < 512) {
        maxIndexLen = 0;
    }
    if (maxIndexLen > 0) {
        CreateVecIndexPerf<int32_t>(vecIndex, 0, maxIndexLen);
    }
    return maxIndexLen;
}

/**
 * @brief 判断边界值是否为 NaN（提取相用：边界取到 NaN 时必须换 isNaN 掩码）
 *        位模式判定（指数位全 1 且尾数非 0），对 ±NaN 与任意 payload 都成立；
 *        不用浮点自比较（v != v）是为了避免浮点相等/自比较类告警。
 * @param value 待判定的边界值（float 或 half）
 * @return true 表示 value 是 NaN
 */
template <typename T, bool largest>
template <typename CT>
__aicore__ inline bool RadixTopKBaseKernel<T, largest>::IsNaNValue(const CT& value)
{
    float valueFp32;
    if constexpr (IsSameType<CT, float>::value) {
        valueFp32 = value;
    } else {
        valueFp32 = static_cast<float>(value); // half -> fp32，NaN 语义保持
    }
    // 读 float 的位模式
    union Fp32Bits {
        float f;
        uint32_t u;
    } bits;
    bits.f = valueFp32;
    return (bits.u & 0x7FFFFFFFU) > 0x7F800000U; // 指数位全 1 且尾数非 0 ⇒ NaN
}

/**
 * @brief 单个 tile 内按 cmpMode 提取匹配 boundaryNumber 的元素并输出
 *        先 Compares → GatherMask values/indices → CopyOutResult
 * @param xLocal tile 输入数据
 * @param boundaryNumber 边界值
 * @param curTileLen 当前 tile 数据长度
 * @param curTileK 当前 tile 还需取出的元素数（会被更新）
 * @param coreTopKOffset 当前 core 输出偏移（会被更新）
 * @param maxIndexLen 预生成索引序列长度
 * @param cmpMaskTensor 比较结果 mask buffer
 * @param vecIndex 预生成索引序列
 * @param tileOffset 当前 tile 在全局数据中的偏移
 * @tparam cmpMode 比较模式（GT/EQ/LT）
 * @tparam CT 计算数据类型
 */
template <typename T, bool largest>
template <AscendC::CMPMODE cmpMode, typename CT>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::SubTopKAndCopyOut(
    LocalTensor<CT>& xLocal, const CT& boundaryNumber, const uint64_t& curTileLen, int32_t& curTileK,
    uint64_t& coreTopKOffset, const uint32_t& maxIndexLen, LocalTensor<uint8_t>& cmpMaskTensor,
    LocalTensor<int32_t>& vecIndex, int32_t tileOffset)
{
    if (curTileK <= 0)
        return;
    LocalTensor<T> outValueLocal = outValueBuf_.Get<T>();
    LocalTensor<float> outValueLocalFp32 = outValueLocal.template ReinterpretCast<float>();
    LocalTensor<int32_t> outIndexLocal = outIndexBuf_.Get<int32_t>();

    uint64_t curTileLenAlign = Ops::Base::CeilAlign(curTileLen, REPEAT_SIZE / sizeof(CT));

    PipeBarrier<PIPE_V>();
    if constexpr (largest && cmpMode != AscendC::CMPMODE::EQ &&
                  (IsSameType<CT, float>::value || IsSameType<CT, half>::value)) {
        if (IsNaNValue(boundaryNumber)) {
            // 边界值本身就是 NaN：候选入选集合只有 NaN（key 域 0xFFFF 组，计数相已把 ±NaN 归入同一
            // bin 并按 NaN 个数分配配额）。此时上式的 NOT(LE) 会退化成「全真」——x <= NaN 对**有限值
            // 同样为 false**，取反后掩码全 1，GatherMask 于是把 tile 前 curTileK 个（有限的）元素当成
            // 入选者拷出；实测混合数据 + largest=true 时表现为 top-k 里 NaN 只剩零星几个（1015 个 NaN
            // 只选中 3 个，其余槽被背景元素顶替）。
            // 改取 isNaN 掩码：NaN == NaN 恒为 false，Compare(x, x, EQ) 逐位取反即 isNaN，
            // 对 ±NaN 与任意 payload 都成立，与计数相的配额口径完全一致。
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
            Not(cmpMaskTensor.ReinterpretCast<int16_t>(), cmpMaskTensor.ReinterpretCast<int16_t>(),
                static_cast<uint32_t>((curTileLenAlign >> 3) / sizeof(int16_t)));
        }
    } else {
        PipeBarrier<PIPE_V>();
        Compares<CT, uint8_t>(cmpMaskTensor, xLocal, boundaryNumber, cmpMode, curTileLenAlign);
    }
    uint64_t rsvdCntV = 0, rsvdCntI = 0;
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForValue_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        GatherMask(outValueLocalFp32, xLocal, cmpMaskTensor.ReinterpretCast<uint32_t>(), true, curTileLen, {1, 1, 0, 0},
                   rsvdCntV);
    } else {
        GatherMask(outValueLocal, xLocal, cmpMaskTensor.ReinterpretCast<uint16_t>(), true, curTileLen, {1, 1, 0, 0},
                   rsvdCntV);
    }
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForIndex_);
    if (maxIndexLen >= curTileLen) {
        PipeBarrier<PIPE_V>();
        GatherMask(outIndexLocal, vecIndex, cmpMaskTensor.ReinterpretCast<uint32_t>(), true, curTileLen, {1, 1, 0, 0},
                   rsvdCntI);
    } else if (maxIndexLen > 0) {
        CopyData<int32_t>(outIndexLocal, vecIndex, maxIndexLen);
        CreateVecIndexDataByAdds(outIndexLocal, vecIndex, curTileLen, maxIndexLen);
        PipeBarrier<PIPE_V>();
        GatherMask(outIndexLocal, outIndexLocal, cmpMaskTensor.ReinterpretCast<uint32_t>(), true, curTileLen,
                   {1, 1, 0, 0}, rsvdCntI);
    } else {
        CreateVecIndexPerf<int32_t>(outIndexLocal, 0, curTileLen);
        PipeBarrier<PIPE_V>();
        GatherMask(outIndexLocal, outIndexLocal, cmpMaskTensor.ReinterpretCast<uint32_t>(), true, curTileLen,
                   {1, 1, 0, 0}, rsvdCntI);
    }
    VToSSync();
    // 非 EQ 通过按 tile 配额封顶：NaN 掩码并入后 rsvdCntV 可能略多于直方图统计出的配额
    // （例如混入 key 域未计入的 -NaN），封顶可避免拷出到 k 槽之外的越界写
    uint64_t passRsvdCnt = (cmpMode != AscendC::CMPMODE::EQ) ? rsvdCntV : static_cast<uint64_t>(curTileK);
    uint32_t count = static_cast<uint32_t>(
        passRsvdCnt < static_cast<uint64_t>(curTileK) ? passRsvdCnt : static_cast<uint64_t>(curTileK));
    SToVSync();
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        PipeBarrier<PIPE_V>();
        Cast(outValueLocal, outValueLocalFp32, RoundMode::CAST_RINT, count);
    }
    PipeBarrier<PIPE_V>();
    Adds(outIndexLocal, outIndexLocal, tileOffset, count);

    VToMTE3Sync();
    CopyOutResult(count, coreTopKOffset);
    curTileK -= count;
    coreTopKOffset += count;
}

/**
 * @brief 将 UB 中 values 和 indices 写回 GM 输出
 * @param count 输出元素数量
 * @param coreTopKOffset 本 core 在输出 buffer 中的起始偏移
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CopyOutResult(const int32_t& count,
                                                                      const uint64_t& coreTopKOffset)
{
    LocalTensor<T> outValueLocal = outValueBuf_.Get<T>();
    LocalTensor<int32_t> outIndexLocal = outIndexBuf_.Get<int32_t>();
    if (count > 0) {
        DataCopyPad(valueGm_[coreTopKOffset], outValueLocal,
                    DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0});
    }
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForValue_);
    if (count > 0) {
        DataCopyPad(
            indexGm_[coreTopKOffset], outIndexLocal,
            DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * count), 0, 0, 0});
    }
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForIndex_);
}

/**
 * @brief 从 workspace 读入所有前驱 core 的 TopK 计数，计算本 core 输出起始偏移
 * @param tmpLocal 临时 Tensor
 * @return 本 core 的输出起始偏移
 */
template <typename T, bool largest>
__aicore__ inline uint64_t RadixTopKBaseKernel<T, largest>::CopyInCoreTopK(LocalTensor<int32_t>& tmpLocal)
{
    uint64_t coreTopKOffset = 0;
    if (blockIdx_) {
        DataCopyPad(
            tmpLocal, coreTopKGm_,
            DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * blockIdx_), 0, 0, 0},
            DataCopyPadExtParams<int32_t>{true, 0, 0, 0});
        MTE2ToSSync();
        for (int32_t coreId = 0; coreId < blockIdx_; coreId++) {
            coreTopKOffset += static_cast<uint64_t>(tmpLocal.GetValue(coreId));
        }
    }
    return coreTopKOffset;
}

/**
 * @brief 将所有 core 的 globalHist 归约到单个 core 的全局直方图
 * @param globalHist 全局直方图（输入/输出）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::ReduceGlobalHist(LocalTensor<int32_t>& globalHist)
{
    int32_t coresInBlock = BLOCK_SIZE / sizeof(int32_t) / numValue_;
    int32_t splitPoint = 32;
    while (splitPoint >= coreNum_ && splitPoint > coresInBlock) {
        splitPoint /= 2;
    }
    int32_t remain = coreNum_;
    while (remain > coresInBlock) {
        PipeBarrier<PIPE_V>();
        Add(globalHist, globalHist, globalHist[splitPoint * numValue_], (remain - splitPoint) * numValue_);
        remain = splitPoint;
        splitPoint /= 2;
    }
    VToSSync();
    for (int32_t binMask = 0; binMask < numValue_; binMask++) {
        int32_t sum = 0;
        for (int32_t i = 0; i < AscendC::Std::min(remain, coresInBlock); i++) {
            sum += globalHist(binMask + i * numValue_);
        }
        globalHist.SetValue(binMask, sum);
    }
}

/**
 * @brief 在全局直方图中找到累积量首次 >= remainK_ 的 bin，更新 involvedMask16_
 * @param globalHist 归约后的全局直方图
 * @param roundId 当前轮 ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::FindBoundaryBin(const LocalTensor<int32_t>& globalHist,
                                                                        int32_t roundId)
{
    boundaryBin = -1;
    boundaryBinPrev = -1;
    for (int32_t binMask = numValue_ - 1; binMask >= 0; binMask--) {
        if (globalHist(binMask) >= remainK_) {
            boundaryBin = binMask;
            boundaryBinPrev = binMask + 1;
            break;
        }
    }
    involvedMask16_ |= static_cast<int16_t>(boundaryBin) << (roundId * BITS_PER_ROUND);
}

/**
 * @brief 从 workspace 读入所有 core 的 globalHist
 * @param globalHist 接收 UB Tensor
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CopyInGlobalHist(LocalTensor<int32_t>& globalHist)
{
    DataCopyPad(globalHist, globalHistGm_,
                DataCopyExtParams{static_cast<uint16_t>(1),
                                  static_cast<uint32_t>(sizeof(int32_t) * numValue_ * coreNum_), 0, 0, 0},
                DataCopyPadExtParams<int32_t>{true, 0, 0, 0});
}

/**
 * @brief 将本 core 边界 bin 累计和写入 workspace
 * @param resTensor 包含累计和的 Tensor
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CopyOutBoundaryBinCumSum(LocalTensor<int32_t>& resTensor)
{
    DataCopyPad(boundaryBinCumSumGm_[blockIdx_], resTensor,
                DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0, 0, 0});
}

/**
 * @brief 计算排在当前 core 之前所有 core 的边界 bin 累计元素数
 * @param resTensor 临时 Tensor
 * @param cumSumValuePrev 输出：前驱 core 的累计边界 bin 元素数
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::ComputeCumSumPrev(LocalTensor<int32_t>& resTensor,
                                                                          int64_t& cumSumValuePrev)
{
    DataCopyPad(resTensor, boundaryBinCumSumGm_,
                DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * coreNum_), 0, 0, 0},
                DataCopyPadExtParams<int32_t>{true, 0, 0, 0});
    MTE2ToSSync();
    cumSumValuePrev = 0;
    for (int32_t coreId = 0; coreId < blockIdx_; coreId++) {
        // 每 core 的边界 bin 计数 < 2^31（per-core 元素数上限），int64 累加对 int32 形状同样安全，
        // 故 UB/WS 两个变体共用本函数（不再各留一份只差累加器类型的实现）
        cumSumValuePrev += resTensor.GetValue(coreId);
    }
}

/**
 * @brief 将本 core 的总 tileTopK 计数写入 workspace
 * @param resTensor 临时 Tensor
 * @param totalTileTopKInCore 本 core 的 TopK 总计数
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::CopyOutCoreTopK(LocalTensor<int32_t>& resTensor,
                                                                        int32_t totalTileTopKInCore)
{
    resTensor.SetValue(0, totalTileTopKInCore);
    SToMTE3Sync();
    DataCopyPad(coreTopKGm_[blockIdx_], resTensor,
                DataCopyExtParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0, 0, 0});
}

/**
 * @brief 将 involvedMask16_ 逆 Twiddle 变换，还原为原始 half/bf16 边界值
 * @tparam CT 目标类型（half 或 float）
 * @return 还原后的边界值
 */
template <typename T, bool largest>
template <typename CT>
__aicore__ inline CT RadixTopKBaseKernel<T, largest>::ExtractBoundaryNumber()
{
    int16_t boundaryNumber16 = involvedMask16_ ^ ((~(involvedMask16_ >> 15)) | 0x8000);
    if constexpr (IsSameType<CT, float>::value) {
        bfloat16_t boundaryBf16 = *reinterpret_cast<bfloat16_t*>(&boundaryNumber16);
        return AscendC::Cast(boundaryBf16);
    } else {
        return *reinterpret_cast<CT*>(&boundaryNumber16);
    }
}

/**
 * @brief 计算当前 core 在输入数据中的起始偏移
 * @param batchId 当前 batch ID
 * @return 当前 core 在 xGm 中的起始偏移
 */
template <typename T, bool largest>
__aicore__ inline uint64_t RadixTopKBaseKernel<T, largest>::CalcBlockOffset(uint64_t batchId)
{
    uint64_t blockOffset = batchId * sortLen_;
    if (blockIdx_ < formerCoreNum_) {
        blockOffset += blockIdx_ * formerTileNum_ * formerTileLen_;
    } else {
        blockOffset += formerCoreNum_ * formerTileNum_ * formerTileLen_ +
                       (blockIdx_ - formerCoreNum_) * tailTileNum_ * formerTileLen_;
    }
    return blockOffset;
}

/**
 * @brief 初始化 TileTopK 阶段的同步事件
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::InitTopKEvent()
{
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);
    eventIDMTE3ToVForValue_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    eventIDMTE3ToVForIndex_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForValue_);
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForIndex_);
    pingPongFlag = true;
}

/**
 * @brief 等待 TileTopK 阶段异步操作完成并释放事件 ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::FinishTopKEvent()
{
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForValue_);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToVForIndex_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMTE3ToVForValue_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMTE3ToVForIndex_);
}

/**
 * @brief 处理单个 tile 的 TopK 提取：先 GT 取确定元素，再 EQ 补齐
 * @param xLocalTyped tile 输入数据
 * @param curTileLen 当前 tile 数据长度
 * @param curTileK 当前 tile 还需取出的元素数
 * @param coreTopKOffset 当前 core 输出偏移
 * @param maxIndexLen 预生成索引序列长度
 * @param cmpMaskTensor 比较结果 mask buffer
 * @param vecIndex 预生成索引序列
 * @param tileOffset tile 在全局数据中的偏移
 * @tparam CT 数据类型（half 或 float）
 */
template <typename T, bool largest>
template <typename CT>
__aicore__ inline void RadixTopKBaseKernel<T, largest>::ProcessOneTileTopK(
    LocalTensor<CT>& xLocalTyped, const uint64_t& curTileLen, int32_t& curTileK, uint64_t& coreTopKOffset,
    const uint32_t& maxIndexLen, LocalTensor<uint8_t>& cmpMaskTensor, LocalTensor<int32_t>& vecIndex,
    int32_t tileOffset)
{
    CT boundaryNumber = this->template ExtractBoundaryNumber<CT>();
    if constexpr (!largest) {
        boundaryNumber = static_cast<CT>(-static_cast<float>(boundaryNumber));
        this->template SubTopKAndCopyOut<AscendC::CMPMODE::LT, CT>(xLocalTyped, boundaryNumber, curTileLen, curTileK,
                                                                   coreTopKOffset, maxIndexLen, cmpMaskTensor, vecIndex,
                                                                   tileOffset);
    } else {
        this->template SubTopKAndCopyOut<AscendC::CMPMODE::GT, CT>(xLocalTyped, boundaryNumber, curTileLen, curTileK,
                                                                   coreTopKOffset, maxIndexLen, cmpMaskTensor, vecIndex,
                                                                   tileOffset);
    }
    this->template SubTopKAndCopyOut<AscendC::CMPMODE::EQ, CT>(xLocalTyped, boundaryNumber, curTileLen, curTileK,
                                                               coreTopKOffset, maxIndexLen, cmpMaskTensor, vecIndex,
                                                               tileOffset);
}

} // namespace RadixTopK
#endif // RADIX_TOPK_COMMON_H
