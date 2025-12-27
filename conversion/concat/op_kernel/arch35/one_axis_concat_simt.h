/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_axis_concat_simt.h
 * \brief
 */
#ifndef ONE_AXIS_CONCAT_SIMT_H
#define ONE_AXIS_CONCAT_SIMT_H

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {

using namespace AscendC;
constexpr uint32_t MAX_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREAD_NUM) inline void ConcatSimt(
    GM_ADDR x, GM_ADDR y, uint32_t tensorIdx, uint32_t colsSize, uint32_t colsOffset, uint32_t m, uint32_t n,
    uint32_t magic, uint32_t shift);

template <typename T>
class OneAxisConcatSimt {
public:
    __aicore__ inline OneAxisConcatSimt(const ConcatTilingDataForSimt& tilingData) : tilingData_(tilingData){};
    __aicore__ inline void ProcessForSimt(GM_ADDR x, GM_ADDR dst);

private:
    const ConcatTilingDataForSimt& tilingData_;
};

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREAD_NUM) inline void ConcatSimt(
    GM_ADDR x, GM_ADDR y, uint32_t tensorIdx, uint32_t colsSize, uint32_t colsOffset, uint32_t m, uint32_t n,
    uint32_t magic, uint32_t shift)
{
    ListTensorDesc tensorList = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    auto src = tensorList.GetDataPtr<T>(tensorIdx);
    __gm__ T* dst = (__gm__ T*)y;
    for (uint32_t i = Simt::GetThreadIdx(); i < m * colsSize; i += Simt::GetThreadNum()) {
        uint32_t t1 = Simt::MulHi(i, magic);
        uint32_t numRows = (t1 + i) >> shift;
        uint32_t numCols = i - numRows * colsSize + colsOffset;
        dst[numRows * n + numCols] = src[i];
    }
}

template <typename T>
__aicore__ inline void OneAxisConcatSimt<T>::ProcessForSimt(GM_ADDR x, GM_ADDR y)
{
    for (uint32_t i = 0; i < tilingData_.tensorNumPerCore; i++) {
        uint32_t tensorIdx = i * GetBlockNum() + GetBlockIdx();
        if (tensorIdx >= tilingData_.tensorNum) {
            return;
        }
        uint32_t colsOffset = tensorIdx == 0 ? 0 : tilingData_.tensorColsOffset[tensorIdx - 1];
        uint32_t colsSize = tilingData_.tensorColsOffset[tensorIdx] - colsOffset;
        uint32_t magic = 0;
        uint32_t shift = 0;

        GetUintDivMagicAndShift(magic, shift, colsSize);
        Simt::VF_CALL<ConcatSimt<T>>(
            Simt::Dim3(MAX_THREAD_NUM), x, y, tensorIdx, colsSize, colsOffset, tilingData_.catDim0, tilingData_.catDim1,
            magic, shift);
    }
}
} // namespace Concat
#endif // ONE_AXIS_CONCAT_SIMT_H