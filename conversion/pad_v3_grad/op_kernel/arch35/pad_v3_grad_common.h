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
 * \file pad_v3_grad_common.h
 * \brief pad v3 grad common utilities
 */

#ifndef ASCENDC_PAD_V3_GRAD_COMMON_H
#define ASCENDC_PAD_V3_GRAD_COMMON_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "pad_v3_grad_struct.h"

namespace PadV3Grad {

constexpr uint32_t BUFFER_NUM = 2;                     ///< 队列 buffer 数量
constexpr uint32_t VL_SIZE = Ops::Base::GetVRegSize(); ///< 向量寄存器大小（256B）
constexpr uint32_t UB_BLOCK = Ops::Base::GetUbBlockSize();

constexpr static int64_t CONST2 = 2;
constexpr static int64_t CONST3 = 3;
constexpr static int64_t CONST4 = 4;
constexpr static int32_t CONST5 = 5;

using PromoteDataT = float; ///< 计算过程使用 float32，避免精度损失

// pad_v3_grad 计算升级类型：bfloat16/float16 用 float32 计算，其余原样
template <typename T>
using PadV3GradCastType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float32_t,
                                             std::conditional_t<std::is_same_v<T, float16_t>, float32_t, T>>;
// GM 偏移类型：tiling 模板参数 U 为 int64 时用 uint64，否则用 uint32
template <typename U>
using PadV3GradGmOffsetType = std::conditional_t<std::is_same_v<U, int64_t>, uint64_t, uint32_t>;

// 读取 tiling 数据填充 SIMT 侧 UB 数组（shape/stride/pad、快速除参数、裁剪边界）并计算输出元素数，
// 返回 false 表示本核无需参与计算（核号越界或输出为空）。blockNum 由调用方获取（后续启动 kernel 仍需使用）
template <typename U, typename GmOffsetType>
__aicore__ inline bool PrepareSimtGradArrays(const PadV3GradACTilingData* td, uint32_t blockNum,
                                             __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                                             __ubuf__ U* inShapes, __ubuf__ U* outShapes, __ubuf__ U* inStrides,
                                             __ubuf__ U* outStrides, __ubuf__ U* leftPads, __ubuf__ U* rightPads,
                                             __ubuf__ U* cutBounds, GmOffsetType& outputSize)
{
    if (AscendC::GetBlockIdx() >= blockNum) {
        return false;
    }

    outputSize = 1;
    for (uint8_t i = 0; i < td->dimNum; i++) {
        outputSize *= td->outShape[i];
    }
    if (outputSize == 0) {
        return false;
    }

    GmOffsetType m = 0, s = 0;
    for (int i = 0; i < td->dimNum; i++) {
        inShapes[i] = static_cast<U>(td->inShape[i]);
        outShapes[i] = static_cast<U>(td->outShape[i]);
        inStrides[i] = static_cast<U>(td->inStride[i]);
        outStrides[i] = static_cast<U>(td->outStride[i]);
        leftPads[i] = td->leftPad[i];
        rightPads[i] = td->rightPad[i];
        AscendC::GetUintDivMagicAndShift(m, s, static_cast<GmOffsetType>(td->outStride[i]));
        magics[i] = m;
        shifts[i] = s;
        cutBounds[i] = static_cast<U>(td->inShape[i]) * static_cast<U>(td->inStride[i]);
    }
    return true;
}

constexpr static AscendC::Reg::CastTrait CAST_TRAIT_0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait CAST_TRAIT_1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};

// 双缓冲排队同步：偶数号缓冲用 EVENT_ID0，奇数号用 EVENT_ID1
template <HardEvent EVENT>
__aicore__ inline void SetEvent(uint32_t bufIdx)
{
    if (bufIdx & 1) {
        SetFlag<EVENT>(EVENT_ID1);
    } else {
        SetFlag<EVENT>(EVENT_ID0);
    }
}

template <HardEvent EVENT>
__aicore__ inline void WaitEvent(uint32_t bufIdx)
{
    if (bufIdx & 1) {
        WaitFlag<EVENT>(EVENT_ID1);
    } else {
        WaitFlag<EVENT>(EVENT_ID0);
    }
}

// 将 UB 上的 pad 梯度结果搬回 GM：非 fp32 先 Cast 回 T（RINT）再走 MTE3，fp32 直接搬；
// copyInParams 按引用传入，函数内设置 blockLen 后用于本次搬出
template <typename T, typename CalType>
__aicore__ inline void CopyGradOutputToGM(GlobalTensor<T>& outputGm, DataCopyExtParams& copyInParams,
                                          LocalTensor<T> src, LocalTensor<CalType> res, uint32_t inSrcStart,
                                          uint32_t inResStart, uint32_t idx, uint64_t outSelfAddr, uint32_t dataLen)
{
    copyInParams.blockLen = dataLen * sizeof(T);
    if constexpr (sizeof(T) != sizeof(float32_t)) {
        Cast<T, CalType>(src, res, RoundMode::CAST_RINT, dataLen);

        SetEvent<HardEvent::V_MTE3>(idx);
        WaitEvent<HardEvent::V_MTE3>(idx);
        // 同步：MTE3_V
        DataCopyPad(outputGm[outSelfAddr], src[inSrcStart], copyInParams);

        SetEvent<HardEvent::MTE3_MTE2>(idx);
    } else {
        // 如果是fp32就可以直接往外搬
        SetEvent<HardEvent::V_MTE3>(idx);
        WaitEvent<HardEvent::V_MTE3>(idx);

        DataCopyPad(outputGm[outSelfAddr], res[inResStart], copyInParams);

        SetEvent<HardEvent::MTE3_V>(idx);
    }
}

} // namespace PadV3Grad

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};

#endif
