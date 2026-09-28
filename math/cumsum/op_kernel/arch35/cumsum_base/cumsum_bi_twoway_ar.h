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
 * ile cumsum_bi_twoway_ar.h
 * rief 3012 紧凑特化（N=1 AR）：紧凑 inner + 组合类。
 * 除紧凑 VF 加法树外全部复用存量模板：Ub 层整层复用（IsTwowayInner trait 分派），
 * 数据流沿通用 TWOWAY 分段缓冲范式（MTE2→oriFold→经典 VEC 桥→fold→VF 树）。
 */

#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_BI_TWOWAY_AR_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_BI_TWOWAY_AR_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base.h"
#include "cumsum_twoway_sklansky.h"
#include "cumsum_ub_sklansky.h"

namespace Cumsum {
using namespace AscendC;

/* 本 inner 属 TWOWAY 家族（Ub 层装配门经此放行，见 cumsum_ub_sklansky.h） */
template <typename DataType, typename PromoteDataType>
class CumsumBiTwowayArSklansky;

template <typename DataType, typename PromoteDataType>
struct IsTwowayInner<CumsumBiTwowayArSklansky<DataType, PromoteDataType>> : std::true_type {};

/* ============================================================================
 * 紧凑 inner：oriFold --MTE2--> --经典VEC桥--> fold --VF树--> res
 * ==========================================================================*/
template <typename DataType, typename PromoteDataType>
class CumsumBiTwowayArSklansky : public CumsumBase<DataType> {
public:
    constexpr static uint32_t VL_ELEM = Ops::Base::GetVRegSize() / sizeof(PromoteDataType);
    constexpr static uint32_t SCAN_LEVELS = 6; /* log2(VL_ELEM)=6，fp32 域 VL */

    __aicore__ inline CumsumBiTwowayArSklansky(TPipe& pipe) : CumsumBase<DataType>(pipe) {}

    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TwowaySklanskyInitData& initData)
    {
        CumsumBase<DataType>::Init(x, y);
        initData_ = initData;
        /* 双缓冲常驻：oriFold=U·dt（MTE2 落盘），fold=U·4（桥/树/res），host 按此预算 */
        int32_t dataBytes = Ops::Base::CeilAlign(static_cast<int32_t>(initData_.xBufferSize),
                                                 static_cast<int32_t>(BLOCK_SIZE_));
        this->pipe_.InitBuffer(inBufX_, dataBytes);
        this->pipe_.InitBuffer(inCastBufX_, dataBytes * 2);
        oriFoldBuffer_ = inBufX_.template Get<DataType>();
        foldBuffer_ = inCastBufX_.template Get<PromoteDataType>();
    }

    __aicore__ inline void BaseProcessPre(const TwowaySklanskyProcessData& processData)
    {
        processData_ = processData;
        inputOffset_ = processData_.offsetM * initData_.lenR * initData_.lenN + processData_.offsetR * initData_.lenN +
                       processData_.offsetN;
        outputOffset_ = inputOffset_;
        if (processData_.isExclusive) {
            processData_.offsetR = processData_.isReverse ? processData_.offsetR + 1 : processData_.offsetR - 1;
            inputOffset_ = processData_.offsetM * initData_.lenR * initData_.lenN +
                           processData_.offsetR * initData_.lenN + processData_.offsetN;
        }
    }

    __aicore__ inline void BaseProcess()
    {
        CopyInCompact();
        ComputeCompact();
    }

    __aicore__ inline void BaseCopyOut();

    __aicore__ inline LocalTensor<PromoteDataType>& GetBaseResTensor()
    {
        return foldBuffer_; /* res = fold（V 域，树/AddCarry/BetweenUb 工作区） */
    }

private:
    template <int K, bool REV>
    __aicore__ inline void ScanLayer(__ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt);
    __aicore__ inline void BridgeVfCopy(__ubuf__ PromoteDataType* dstPtr, __ubuf__ PromoteDataType* srcPtr,
                                        uint16_t blockCnt);
    __aicore__ inline void CopyInCompact();
    __aicore__ inline void CastFoldCompact();
    __aicore__ inline void ComputeCompact();
    template <bool REV>
    __aicore__ inline void ScanLayersWindow(__ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt);
    template <bool REV>
    __aicore__ inline void CombineVlBlocks(__ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt);
    template <bool REV>
    __aicore__ inline void CombineOneLevel(__ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt, uint16_t d);

    constexpr static uint32_t BLOCK_SIZE_ = Ops::Base::GetUbBlockSize();

    TwowaySklanskyInitData initData_;
    TwowaySklanskyProcessData processData_;

    TBuf<> inBufX_;
    TBuf<> inCastBufX_;
    LocalTensor<DataType> oriFoldBuffer_;
    LocalTensor<PromoteDataType> foldBuffer_;
    int64_t inputOffset_ = 0;
    int64_t outputOffset_ = 0;
};

/* CopyIn：MTE2 GM→oriFold；首 V_MTE2 护上一窗口 cast-back，尾 MTE2_V 护桥读 */
template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::CopyInCompact()
{
    uint32_t n = static_cast<uint32_t>(processData_.ubFactorR);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);

    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
    /* exclusive 边界窗源收缩至行内 + DataCopyPad padding 补零，避免越界读 */
    if (processData_.isExclusive) {
        int64_t rowEnd = processData_.offsetM * initData_.lenR * initData_.lenN + initData_.lenR * initData_.lenN +
                         processData_.offsetN;
        if (!processData_.isReverse) {
            if (processData_.offsetR < 0) { /* 正向首窗：源起点 clamp 行首 */
                if (n > 1) {
                    cp.blockLen = (n - 1) * sizeof(DataType);
                    DataCopyPadExtParams<DataType> padL{true, 1, 0, 0};
                    DataCopyPad(oriFoldBuffer_[0], this->xGm_[inputOffset_ + 1], cp, padL);
                } /* n==1：无数据搬，w[0] 由后置零覆盖 */
            } else {
                cp.blockLen = n * sizeof(DataType);
                DataCopyPad(oriFoldBuffer_[0], this->xGm_[inputOffset_], cp, padParams);
            }
        } else {
            if (inputOffset_ + static_cast<int64_t>(n) > rowEnd) { /* 反向尾窗：截尾 */
                if (n > 1) {
                    cp.blockLen = (n - 1) * sizeof(DataType);
                    DataCopyExtParams cp2;
                    cp2.blockCount = 1;
                    cp2.blockLen = (n - 1) * sizeof(DataType);
                    cp2.srcStride = 0;
                    cp2.dstStride = 0;
                    DataCopyPadExtParams<DataType> padR{true, 0, 1, 0};
                    DataCopyPad(oriFoldBuffer_[0], this->xGm_[inputOffset_], cp2, padR);
                } /* n==1 同上 */
            } else {
                cp.blockLen = n * sizeof(DataType);
                DataCopyPad(oriFoldBuffer_[0], this->xGm_[inputOffset_], cp, padParams);
            }
        }
    } else {
        cp.blockLen = n * sizeof(DataType);
        DataCopyPad(oriFoldBuffer_[0], this->xGm_[inputOffset_], cp, padParams);
    }

    eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
    CastFoldCompact();
}

/* 桥：MTE2 数据经经典 VEC（fp16/bf16 Cast 循环；fp32 无桥）落到 fold 后，
 * 再经 VF 拷贝（恒等 Gather+StoreAlign）改写，使 VF 树的 LoadAlign 可见 */
template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::CastFoldCompact()
{
    uint32_t n = static_cast<uint32_t>(processData_.ubFactorR);
    uint16_t blockCnt = static_cast<uint16_t>((n + VL_ELEM - 1) / VL_ELEM); /* ceil：尾块整块进树 */
    if constexpr (!std::is_same<DataType, PromoteDataType>::value) {
        uint64_t mask[2] = {UINT64_MAX, 0};
        int32_t total = static_cast<int32_t>(n);
        constexpr int32_t ONCE = 255 * 64; /* 255 repeat × 64 elem/repeat，与基类常量同义 */
        int32_t fullCount = total / ONCE;
        for (int32_t i = 0; i < fullCount; i++) { /* 固定步长，与基类 fold cast 循环同构 */
            int32_t offset = i * ONCE;
            Cast(foldBuffer_[offset], oriFoldBuffer_[offset], RoundMode::CAST_NONE, mask, 255, {1, 1, 8, 4});
        }
        int32_t tailCast = total - fullCount * ONCE;
        if (tailCast > 0) {
            int32_t offset = fullCount * ONCE;
            Cast(foldBuffer_[offset], oriFoldBuffer_[offset], RoundMode::CAST_NONE, mask, (tailCast + 63) / 64,
                 {1, 1, 8, 4});
        }
        /* 原地触染：fold(classic 写) → fold(VF 写) */
        BridgeVfCopy((__ubuf__ PromoteDataType*)foldBuffer_.GetPhyAddr(),
                     (__ubuf__ PromoteDataType*)foldBuffer_.GetPhyAddr(), blockCnt);
    } else {
        BridgeVfCopy((__ubuf__ PromoteDataType*)foldBuffer_.GetPhyAddr(),
                     (__ubuf__ PromoteDataType*)oriFoldBuffer_.GetPhyAddr(), blockCnt);
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::BridgeVfCopy(
    __ubuf__ PromoteDataType* dstPtr, __ubuf__ PromoteDataType* srcPtr, uint16_t blockCnt)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg pAll = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<PromoteDataType> reg;
        AscendC::Reg::RegTensor<int32_t> idReg;
        AscendC::Reg::Arange(idReg, 0);           /* 恒等索引 */
        for (uint16_t b = 0; b < blockCnt; b++) { /* 固定步长，归纳变量 uint16_t */
            uint32_t off = static_cast<uint32_t>(b) * VL_ELEM;
            AscendC::Reg::Gather(reg, srcPtr + off, (AscendC::Reg::RegTensor<uint32_t>&)idReg, pAll);
            AscendC::Reg::StoreAlign(dstPtr + off, reg, pAll);
        }
    }
}

/* 第一级：VL 内 6 层 Sklansky，层在外块在内（索引/mask 出块循环），寻址 ptr+运行时偏移 */
/* 层函数（template<int K> 常量层参）：VF 算术链构造索引/mask，块循环在内 */
/* reverse 镜像（Sklansky 对称翻转）：前半 += 后半首，idx=组首+2^K，mask bit==0 */
template <typename DataType, typename PromoteDataType>
template <int K, bool REV>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::ScanLayer(__ubuf__ PromoteDataType* foldPtr,
                                                                                      uint16_t blockCnt)
{
    /* 正向：mask 目标 1（后半加），idx 偏移 2^K−1（前半末）；
     * 反向：mask 目标 0（前半加），idx 偏移 2^K（后半首）。
     * 方向是 tiling 期属性，模板参数化后 if constexpr 在 __VEC_SCOPE__ 外编译期消解，
     * scope 内零分支；指令标量操作数与通用版 VfGatherBeforeSecondSkalansky 同款合法。 */
    int32_t maskTarget = 0;
    int32_t idxOff = 0;
    if constexpr (REV) {
        maskTarget = 0;
        idxOff = static_cast<int32_t>(1u << K);
    } else {
        maskTarget = 1;
        idxOff = static_cast<int32_t>((1u << K) - 1);
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg pAll = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<PromoteDataType> curReg;
        AscendC::Reg::RegTensor<PromoteDataType> carryReg;
        AscendC::Reg::RegTensor<PromoteDataType> sumReg;
        AscendC::Reg::RegTensor<int32_t> idxReg;
        AscendC::Reg::RegTensor<int32_t> bitReg;
        AscendC::Reg::RegTensor<int32_t> scratchReg;
        AscendC::Reg::RegTensor<int32_t> groupReg;
        AscendC::Reg::RegTensor<int32_t> oneReg;
        AscendC::Reg::MaskReg pHalf;

        AscendC::Reg::Duplicate(oneReg, maskTarget, pAll);
        AscendC::Reg::Arange(idxReg, 0);
        AscendC::Reg::Duplicate(groupReg, static_cast<int32_t>(1u << (K + 1)), pAll);
        AscendC::Reg::Div(idxReg, idxReg, groupReg, pAll);
        AscendC::Reg::Muls(idxReg, idxReg, static_cast<int32_t>(1u << (K + 1)), pAll);
        AscendC::Reg::Adds(idxReg, idxReg, idxOff, pAll);
        /* mask = bit0(lane / 2^K)：And 无标量版，恒等式 bit0(v)=v−(v/2)·2 */
        AscendC::Reg::Arange(bitReg, 0);
        AscendC::Reg::Duplicate(groupReg, static_cast<int32_t>(1u << K), pAll);
        AscendC::Reg::Div(bitReg, bitReg, groupReg, pAll);
        AscendC::Reg::Duplicate(groupReg, 2, pAll);
        AscendC::Reg::Div(scratchReg, bitReg, groupReg, pAll);
        AscendC::Reg::Muls(scratchReg, scratchReg, 2, pAll);
        AscendC::Reg::Sub(scratchReg, bitReg, scratchReg, pAll);
        AscendC::Reg::Compare<int32_t, CMPMODE::EQ>(pHalf, scratchReg, oneReg, pAll);
        for (uint16_t b = 0; b < blockCnt; b++) { /* 块在内：固定步长，归纳变量 uint16_t */
            uint32_t off = static_cast<uint32_t>(b) * VL_ELEM;
            /* 半掩码 Add 会清零未选中 lane：全掩码算 sum + Select 覆盖选中 lane */
            AscendC::Reg::Gather(carryReg, foldPtr + off, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pAll);
            AscendC::Reg::LoadAlign(curReg, foldPtr + off);
            AscendC::Reg::Add(sumReg, curReg, carryReg, pAll);
            AscendC::Reg::Select(curReg, sumReg, curReg, pHalf);
            AscendC::Reg::StoreAlign(foldPtr + off, curReg, pAll);
        }
    }
}

template <typename DataType, typename PromoteDataType>
template <bool REV>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::ScanLayersWindow(
    __ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt)
{
    ScanLayer<0, REV>(foldPtr, blockCnt);
    ScanLayer<1, REV>(foldPtr, blockCnt);
    ScanLayer<2, REV>(foldPtr, blockCnt);
    ScanLayer<3, REV>(foldPtr, blockCnt);
    ScanLayer<4, REV>(foldPtr, blockCnt);
    ScanLayer<5, REV>(foldPtr, blockCnt);
}

/* ---------------------------------------------------------------------------
 * 第二级：VL 块间组合——VfFirstAdd 同构（级 j/m 双循环固定步长 1，级距 d 循环体尾
 * 标量递推；级 j：块 b（bit j 置位）+= 块 b−2^j 整 VL 窗口）。
 * -------------------------------------------------------------------------*/
/* 跨块组合：正向后半块 += 前半末块 lane63；反向镜像前半块 += 后半首块 lane0 */
template <typename DataType, typename PromoteDataType>
template <bool REV>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::CombineOneLevel(
    __ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt, uint16_t d)
{
    /* 方向派生量（广播源 lane / 源块修正 / 目标块基偏移）if constexpr 编译期消解；
     * 残组段拆独立 __VEC_SCOPE__，其触发条件 rem>d 为运行时值、置于 scope 外 */
    int32_t bcastLane = 0;
    uint16_t srcAdj = 0;
    uint16_t dstSel = 0;
    if constexpr (REV) {
        bcastLane = 0;
        srcAdj = 0;
        dstSel = 0;
    } else {
        bcastLane = static_cast<int32_t>(VL_ELEM - 1);
        srcAdj = 1;
        dstSel = d;
    }
    uint16_t grpCnt = static_cast<uint16_t>(blockCnt / (2u * d)); /* 完整 2d 块组数 */
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg pAll = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<PromoteDataType> curReg;
        AscendC::Reg::RegTensor<PromoteDataType> carryReg;
        AscendC::Reg::RegTensor<int32_t> bcastIdxReg;
        AscendC::Reg::Duplicate(bcastIdxReg, bcastLane, pAll);
        for (uint16_t g = 0; g < grpCnt; g++) { /* 组枚举固定步长，块偏移纯算术 */
            uint32_t srcOff = static_cast<uint32_t>(g * (2u * d) + d - srcAdj) * VL_ELEM;
            uint16_t dstBase = static_cast<uint16_t>(g * (2u * d) + dstSel);
            for (uint16_t t = 0; t < d; t++) {
                uint32_t dstOff = static_cast<uint32_t>(dstBase + t) * VL_ELEM;
                AscendC::Reg::Gather(carryReg, foldPtr + srcOff, (AscendC::Reg::RegTensor<uint32_t>&)bcastIdxReg, pAll);
                AscendC::Reg::LoadAlign(curReg, foldPtr + dstOff);
                AscendC::Reg::Add(curReg, curReg, carryReg, pAll);
                AscendC::Reg::StoreAlign(foldPtr + dstOff, curReg, pAll);
            }
        }
    }
    uint16_t rem = static_cast<uint16_t>(blockCnt - grpCnt * (2u * d));
    if (rem > d) { /* 残组：scope 外运行时判定，段内与完整组同构 */
        uint16_t remDst = 0;
        if constexpr (REV) {
            remDst = d;
        } else {
            remDst = static_cast<uint16_t>(rem - d);
        }
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg pAll = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::RegTensor<PromoteDataType> curReg;
            AscendC::Reg::RegTensor<PromoteDataType> carryReg;
            AscendC::Reg::RegTensor<int32_t> bcastIdxReg;
            AscendC::Reg::Duplicate(bcastIdxReg, bcastLane, pAll);
            uint32_t srcOff = static_cast<uint32_t>(grpCnt * (2u * d) + d - srcAdj) * VL_ELEM;
            uint16_t dstBase = static_cast<uint16_t>(grpCnt * (2u * d) + dstSel);
            for (uint16_t t = 0; t < remDst; t++) {
                uint32_t dstOff = static_cast<uint32_t>(dstBase + t) * VL_ELEM;
                AscendC::Reg::Gather(carryReg, foldPtr + srcOff, (AscendC::Reg::RegTensor<uint32_t>&)bcastIdxReg, pAll);
                AscendC::Reg::LoadAlign(curReg, foldPtr + dstOff);
                AscendC::Reg::Add(curReg, curReg, carryReg, pAll);
                AscendC::Reg::StoreAlign(foldPtr + dstOff, curReg, pAll);
            }
        }
    }
}

template <typename DataType, typename PromoteDataType>
template <bool REV>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::CombineVlBlocks(
    __ubuf__ PromoteDataType* foldPtr, uint16_t blockCnt)
{
    uint16_t jn = 0;
    while ((1u << jn) < blockCnt) {
        jn++; /* log2(blockCnt)，纯标量域 */
    }
    uint16_t d = 1;
    for (uint16_t j = 0; j < jn; j++) { /* 级循环在作用域外（基类同构） */
        CombineOneLevel<REV>(foldPtr, blockCnt, d);
        d = static_cast<uint16_t>(d << 1); /* 级距标量递推 */
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::ComputeCompact()
{
    __ubuf__ PromoteDataType* foldPtr = (__ubuf__ PromoteDataType*)foldBuffer_.GetPhyAddr();
    uint32_t n = static_cast<uint32_t>(processData_.ubFactorR);
    /* ceil 块数：不足 VL 的尾块整块进树（正序 gather idx<lane，垃圾 lane 不被读） */
    uint16_t blockCnt = static_cast<uint16_t>((n + VL_ELEM - 1) / VL_ELEM);
    /* exclusive 边界元素（正向首窗头/反向末窗尾）语义值为 0：树前 VF 置零 */
    if (processData_.isExclusive) {
        bool zeroHead = !processData_.isReverse && processData_.offsetR < 0; /* 正向首窗 */
        int64_t rowEnd = processData_.offsetM * initData_.lenR * initData_.lenN + initData_.lenR * initData_.lenN +
                         processData_.offsetN;
        bool zeroTail = processData_.isReverse && inputOffset_ + static_cast<int64_t>(n) > rowEnd;
        if (zeroHead || zeroTail) {
            /* 置零目标 lane/offset 为逐窗运行时值，scope 外求值后 scope 内零分支 */
            int32_t tgtLane = 0;
            uint32_t zeroOff = 0;
            if (zeroHead) {
                tgtLane = 0;
                zeroOff = 0;
            } else {
                tgtLane = static_cast<int32_t>((n - 1) % VL_ELEM);
                zeroOff = static_cast<uint32_t>((n - 1) / VL_ELEM) * VL_ELEM;
            }
            __VEC_SCOPE__
            {
                AscendC::Reg::MaskReg
                    pAll = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::RegTensor<PromoteDataType> curReg;
                AscendC::Reg::RegTensor<PromoteDataType> zeroReg;
                AscendC::Reg::RegTensor<int32_t> laneReg;
                AscendC::Reg::RegTensor<int32_t> tgtReg;
                AscendC::Reg::MaskReg pZero;
                AscendC::Reg::Duplicate(zeroReg, static_cast<PromoteDataType>(0), pAll);
                AscendC::Reg::Arange(laneReg, 0);
                AscendC::Reg::Duplicate(tgtReg, tgtLane, pAll);
                AscendC::Reg::Compare<int32_t, CMPMODE::EQ>(pZero, laneReg, tgtReg, pAll);
                AscendC::Reg::LoadAlign(curReg, foldPtr + zeroOff);
                AscendC::Reg::Select(curReg, zeroReg, curReg, pZero);
                AscendC::Reg::StoreAlign(foldPtr + zeroOff, curReg, pAll);
            }
        }
    }
    /* 反向尾块垃圾清零：镜像 idx>lane 会读垃圾 lane，清零后 +0 无害 */
    if (processData_.isReverse != 0) {
        uint32_t padEnd = static_cast<uint32_t>(blockCnt) * VL_ELEM;
        if (padEnd > n) {
            __VEC_SCOPE__
            {
                AscendC::Reg::MaskReg
                    pAll = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::RegTensor<PromoteDataType> curReg;
                AscendC::Reg::RegTensor<PromoteDataType> zeroReg;
                AscendC::Reg::RegTensor<int32_t> laneReg;
                AscendC::Reg::RegTensor<int32_t> validReg;
                AscendC::Reg::MaskReg pGarb;
                AscendC::Reg::Duplicate(zeroReg, static_cast<PromoteDataType>(0), pAll);
                AscendC::Reg::Arange(laneReg, 0);
                AscendC::Reg::Duplicate(validReg, static_cast<int32_t>(n % VL_ELEM), pAll);
                AscendC::Reg::Compare<int32_t, CMPMODE::GE>(pGarb, laneReg, validReg, pAll);
                uint32_t tailOff = static_cast<uint32_t>(blockCnt - 1) * VL_ELEM;
                AscendC::Reg::LoadAlign(curReg, foldPtr + tailOff);
                AscendC::Reg::Select(curReg, zeroReg, curReg, pGarb);
                AscendC::Reg::StoreAlign(foldPtr + tailOff, curReg, pAll);
            }
        }
    }
    if (processData_.isReverse != 0) {
        ScanLayersWindow<true>(foldPtr, blockCnt);
        CombineVlBlocks<true>(foldPtr, blockCnt);
    } else {
        ScanLayersWindow<false>(foldPtr, blockCnt);
        CombineVlBlocks<false>(foldPtr, blockCnt);
    }
}

/* CopyOut：出桥 → V_MTE3 栅栏 → MTE3 → GM → 双栅栏护下一窗口写 */
template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiTwowayArSklansky<DataType, PromoteDataType>::BaseCopyOut()
{
    uint32_t n = static_cast<uint32_t>(processData_.ubFactorR);
    if constexpr (!std::is_same<DataType, PromoteDataType>::value) {
        Cast(oriFoldBuffer_, GetBaseResTensor(), RoundMode::CAST_RINT, static_cast<int32_t>(n));
    }
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = n * sizeof(DataType);
    cp.srcStride = 0;
    cp.dstStride = 0;
    if constexpr (std::is_same<DataType, PromoteDataType>::value) {
        DataCopyPad(this->yGm_[outputOffset_], foldBuffer_[0], cp); /* fp32 直读 fold */
    } else {
        DataCopyPad(this->yGm_[outputOffset_], oriFoldBuffer_[0], cp); /* fp16 读 cast 回区 */
    }
    event_t reuseID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(reuseID);
    WaitFlag<HardEvent::MTE3_MTE2>(reuseID);
    event_t reuseVID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(reuseVID);
    WaitFlag<HardEvent::MTE3_V>(reuseVID);
}

/* ============================================================================
 * 组合类：复制 CumsumUbSsTwowaySs 的 Init/Process 装配，inner 换紧凑类。
 * Ub 层（memo Fenwick/SkalanskyBetweenUb/窗口循环/行分核）经继承整层复用。
 * ==========================================================================*/
template <typename DataType, typename PromoteDataType>
class CumsumBiUbSsTwowayArSs
    : public CumsumUbSklansky<DataType, PromoteDataType, CumsumBiTwowayArSklansky<DataType, PromoteDataType>> {
public:
    using Base = CumsumUbSklansky<DataType, PromoteDataType, CumsumBiTwowayArSklansky<DataType, PromoteDataType>>;
    __aicore__ inline CumsumBiUbSsTwowayArSs(TPipe& pipe) : Base(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    constexpr static int32_t BLK_SIZE_ = Ops::Base::GetUbBlockSize();
    const CumsumSklanskyTilingData* tiling_;
};

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiUbSsTwowayArSs<DataType, PromoteDataType>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace)
{
    tiling_ = tilingData;
    /* 与存量 CumsumUbSsTwowaySs::Init 逐行同构（字段全集） */
    UbSklanskyInitData initData;
    initData.lenM = tiling_->lenM;
    initData.lenR = tiling_->lenR;
    initData.lenN = tiling_->lenN;
    initData.reverse = tiling_->reverse == 1 ? true : false;
    initData.xBufferSize = tiling_->xBufSize;
    initData.xUnfoldBufferSize = tiling_->xUnfoldBufSize;
    initData.rFoldAfterSize = tiling_->tailCoreMainUbFoldPara.foldLen;
    initData.rFoldAfterCount = tiling_->tailCoreMainUbFoldPara.sklanskyIter;
    initData.rFoldCount = tiling_->tailCoreMainUbFoldPara.foldCount;
    initData.ubFactorR = tiling_->rUbPara.tailCoreUbPara.ubFactor;
    initData.ubTailFactorR = tiling_->rUbPara.tailCoreUbPara.ubTailFactor;
    initData.ubCountR = tiling_->rUbPara.tailCoreUbPara.ubCount;
    initData.rTailFoldAfterSize = tiling_->tailCoreTailUbFoldPara.foldLen;
    initData.rTailFoldAfterCount = tiling_->tailCoreTailUbFoldPara.sklanskyIter;
    initData.rTailFoldCount = tiling_->tailCoreTailUbFoldPara.foldCount;
    initData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubTailFactor;
    initData.memoLen = __CumsumUtil::CalLog2(tiling_->rUbPara.tailCoreUbPara.ubCount);
    initData.ubSklanskyBufSize = tiling_->ubSklanskyBufSize;
    initData.perMemoSize = Ops::Base::CeilAlign(initData.ubFactorN, static_cast<int32_t>(BLK_SIZE_ / sizeof(DataType)));
    Base::BaseInit(x, y, workspace, initData); /* trait 放行 TWOWAY 装配 → 紧凑 inner + memo */
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumBiUbSsTwowayArSs<DataType, PromoteDataType>::Process()
{
    if (this->blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    UbSklanskyProcessData processData;
    processData.offsetR = 0;
    processData.offsetN = 0;
    processData.isExclusive = tiling_->exclusive;

    int32_t fullCoreNum = tiling_->mBlockPara.blockCount - 1;
    if (this->blockIdx_ < fullCoreNum) {
        for (int32_t i = 0; i < tiling_->mBlockPara.blockFactor; i++) {
            processData.offsetM = this->blockIdx_ * tiling_->mBlockPara.blockFactor + i;
            Base::BaseProcessPre(processData);
            Base::BaseProcess();
        }
    } else {
        for (int32_t i = 0; i < tiling_->mBlockPara.blockTailFactor; i++) {
            processData.offsetM = fullCoreNum * tiling_->mBlockPara.blockFactor + i;
            Base::BaseProcessPre(processData);
            Base::BaseProcess();
        }
    }
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_BI_TWOWAY_AR_H
