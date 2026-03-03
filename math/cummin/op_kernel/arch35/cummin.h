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
 * \file Cummin_regbase_common.h
 * \brief Cummin
 */

#ifndef Cummin_REGBASE_COMMON_H
#define Cummin_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "cummin_struct.h"
#include "cummin_tiling_key.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"
namespace CumminRegbase {

using namespace AscendC;

constexpr static uint32_t BLOCK_SIZE = 32;
constexpr static uint32_t VREGTENSOR_SIZE = 256;
constexpr static uint32_t BUFFER_NUM = 1;

template <typename T>
class Cummin {
public:
    __aicore__ inline Cummin(TPipe& tpipe, CumminRegbaseTilingData& tilingData) : tpipe(tpipe), tiling(tilingData){};

public:
    __aicore__ inline void InitGMBuffers(GM_ADDR x, GM_ADDR y, GM_ADDR argmin, int64_t offset);
    __aicore__ inline void InitBuffers(int64_t yBufferSize, int64_t rows, bool isNeedArgminTensor);
    __aicore__ inline void InitBuffersForN(int64_t yBufferSize);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmin);
    static __aicore__ inline void InsertSync(const HardEvent& event);

    __aicore__ inline void CopyIn(LocalTensor<T>& tensor, uint16_t copyRows, uint32_t dataNum, int64_t xOffset);

    template <typename C>
    __aicore__ inline void CopyOut(
        GlobalTensor<C>& gm, LocalTensor<C>& tensor, uint16_t copyRows, uint32_t dataNum, int64_t offset);

    __aicore__ inline void ComputeM();
    __aicore__ inline void ComputeR(CumminSplitInfo info);
    __aicore__ inline void ComputeN(CumminSplitInfo info);
    __aicore__ inline void ComputeReservedM(GM_ADDR x, GM_ADDR y, GM_ADDR argmin);
    __aicore__ inline void DoCompare(
        LocalTensor<int32_t>& argminTensor, int64_t computeTimes, int64_t offset, int64_t regComputeTimes,
        uint32_t reservedRegComputeTimes, int32_t duplicate);

public:
    TPipe& tpipe;
    // tilingData
    CumminRegbaseTilingData& tiling;
    int64_t M = 0;
    int64_t N = 0;
    int64_t R = 0;
    int64_t AlignedN = 0;

    int64_t splitM = 0;
    int64_t computeM = 0;
    int64_t ReservedM = 0;
    int64_t mRowsPerCore = 0;
    int64_t corenum = 0;
    int64_t perGroupCoreNum = 0;
    int64_t gmOffset = 0;
    int64_t computeLength = 0;
    int64_t allocLength = 0;
    int64_t stride = 0;
    int64_t blockId = 0;
    int64_t blockCount = 0;
    uint32_t dSize = sizeof(T);
    uint32_t regTensorPerComputeLength = VREGTENSOR_SIZE / sizeof(T);
    CumminSplitInfo generalProcessInfo;
    CumminSplitInfo resevedMgeneralProcessInfo;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int32_t> argminGm;

    TQue<TPosition::VECIN, BUFFER_NUM> xTmpQue;
    TQue<TPosition::VECIN, BUFFER_NUM> argminTmpQue;
    TQue<TPosition::VECIN, BUFFER_NUM> yQue;
    TQue<TPosition::VECIN, BUFFER_NUM> argminQue;

    LocalTensor<T> tmpXTensor;
    LocalTensor<int32_t> tmpArgminTensor;
    LocalTensor<T> yTensor;
    LocalTensor<int32_t> argminTensor;
};

template <typename T>
__aicore__ inline void Cummin<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmin)
{
    blockId = GetBlockIdx();
    blockCount = GetBlockNum();
    corenum = tiling.coreNum;
    perGroupCoreNum = tiling.perGroupCoreNum;
    M = tiling.M;
    N = tiling.N;
    R = tiling.R;
    ;

    AlignedN = Ops::Base::CeilAlign(N, static_cast<int64_t>(BLOCK_SIZE / dSize));

    splitM = tiling.splitM;
    computeM = tiling.computeM;
    ReservedM = tiling.ReservedM;

    generalProcessInfo = tiling.generalProcessInfo;
    mRowsPerCore = tiling.mRowsPerCore;

    if (blockId < corenum && blockId % perGroupCoreNum < tiling.formerCore) {
        resevedMgeneralProcessInfo = tiling.formerCoreProcessInfo;
        gmOffset = blockCount * mRowsPerCore * R * N + (blockId / perGroupCoreNum) * R * N +
                   (blockId % perGroupCoreNum) * tiling.formerCoreComputeLength;
        computeLength = tiling.formerCoreComputeLength;

    } else {
        resevedMgeneralProcessInfo = tiling.tailCoreProcessInfo;
        gmOffset = blockCount * mRowsPerCore * R * N + (blockId / perGroupCoreNum) * R * N +
                   (blockId % perGroupCoreNum) * tiling.tailCoreComputeLength + tiling.formerCore;
        computeLength = tiling.tailCoreComputeLength;
    }

    auto offset = blockId * mRowsPerCore * N * R;
    InitGMBuffers(x, y, argmin, offset);
}

template <typename T>
__aicore__ inline void Cummin<T>::InitGMBuffers(GM_ADDR x, GM_ADDR y, GM_ADDR argmin, int64_t offset)
{
    xGm.SetGlobalBuffer((__gm__ T*)x + offset);
    yGm.SetGlobalBuffer((__gm__ T*)y + offset);
    argminGm.SetGlobalBuffer((__gm__ int32_t*)argmin + offset);
}
template <typename T>
__aicore__ inline void Cummin<T>::InitBuffers(int64_t yBufferSize, int64_t rows, bool isNeedArgminTensor)
{
    auto argminBuffersize = Ops::Base::CeilAlign(yBufferSize, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
    tpipe.InitBuffer(xTmpQue, BUFFER_NUM, yBufferSize * dSize);
    tpipe.InitBuffer(argminTmpQue, BUFFER_NUM, argminBuffersize * sizeof(int32_t));
    tpipe.InitBuffer(yQue, BUFFER_NUM, yBufferSize * rows * dSize);
    tmpXTensor = xTmpQue.AllocTensor<T>();
    tmpArgminTensor = argminTmpQue.AllocTensor<int32_t>();
    yTensor = yQue.AllocTensor<T>();

    if (isNeedArgminTensor) {
        tpipe.InitBuffer(argminQue, BUFFER_NUM, yBufferSize * rows * sizeof(int32_t));
        argminTensor = argminQue.AllocTensor<int32_t>();
    }
}

template <typename T>
static __aicore__ inline void Cummin<T>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    switch (event) {
        case HardEvent::V_MTE3:
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            break;
        case HardEvent::V_MTE2:
            SetFlag<HardEvent::V_MTE2>(eventID);
            WaitFlag<HardEvent::V_MTE2>(eventID);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            break;
        case HardEvent::MTE2_V:
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            break;
        case HardEvent::MTE3_V:
            SetFlag<HardEvent::MTE3_V>(eventID);
            WaitFlag<HardEvent::MTE3_V>(eventID);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void Cummin<T>::CopyIn(LocalTensor<T>& tensor, uint16_t copyRows, uint32_t dataNum, int64_t xOffset)
{
    InsertSync(HardEvent::MTE3_MTE2);
    DataCopyPadExtParams<T> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams{copyRows, dataNum * dSize, stride * dSize, 0, 0};
    DataCopyPad(tensor, this->xGm[xOffset], copyParams, dataCopyPadExtParams);
    InsertSync(HardEvent::MTE2_V);
}

template <typename T>
template <typename C>
__aicore__ inline void Cummin<T>::CopyOut(
    GlobalTensor<C>& gm, LocalTensor<C>& tensor, uint16_t copyRows, uint32_t dataNum, int64_t offset)
{
    InsertSync(HardEvent::V_MTE3);
    DataCopyExtParams copyParams{copyRows, dataNum * dSize, 0, stride * dSize, 0};
    DataCopyPad(gm[offset], tensor, copyParams);
    InsertSync(HardEvent::MTE3_MTE2);
    InsertSync(HardEvent::MTE3_V);
}

template <typename T>
__aicore__ inline void Cummin<T>::ComputeM()
{
    InitBuffers(AlignedN, computeM * R, true);
    allocLength = AlignedN;
    auto rows = computeM;
    auto regComputeTimes = Ops::Base::CeilDiv(AlignedN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
    auto reservedRegComputeTimes = AlignedN % regTensorPerComputeLength;

    for (auto i = 0; i < splitM; i++) {
        if (i == splitM - 1 && ReservedM > 0) {
            rows = ReservedM;
        }

        auto offset = i * R * N * computeM;
        CopyIn(yTensor, static_cast<uint16_t>(rows * R), N, offset);

        for (auto j = 0; j < rows; j++) {
            DataCopy(tmpXTensor, yTensor[j * R * AlignedN], AlignedN);

            auto dupLen = Ops::Base::CeilAlign(AlignedN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
            Duplicate<int32_t>(tmpArgminTensor, 0, dupLen);
            Duplicate<int32_t>(argminTensor[j * R * AlignedN], 0, dupLen);

            DoCompare(argminTensor, R - 1, j * R * AlignedN + AlignedN, regComputeTimes, reservedRegComputeTimes, 1);
        }
        CopyOut(yGm, yTensor, R * rows, N, offset);
        CopyOut(argminGm, argminTensor, R * rows, N, offset);
    }
}

template <typename T>
__aicore__ inline void Cummin<T>::ComputeR(CumminSplitInfo info)
{
    InitBuffers(info.allocLength, info.computeR, true);
    allocLength = info.allocLength;

    auto regComputeTimes = Ops::Base::CeilDiv(info.allocLength, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
    auto reservedRegComputeTimes = info.allocLength % regTensorPerComputeLength;

    for (auto i = 0; i < mRowsPerCore; i++) {
        auto rows = info.computeR;
        for (auto j = 0; j < info.splitR; j++) {
            if (j == info.splitR - 1 && info.reservedR > 0) {
                rows = info.reservedR;
            }

            auto offset = i * R * N + j * info.computeR * N;
            CopyIn(yTensor, static_cast<uint16_t>(rows), info.computeLength, offset);

            if (j == 0) {
                auto dupLen = Ops::Base::CeilAlign(info.allocLength, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
                DataCopy(tmpXTensor, yTensor, info.allocLength);
                Duplicate<int32_t>(argminTensor, 0, dupLen);
                Duplicate<int32_t>(tmpArgminTensor, 0, dupLen);
            }

            DoCompare(argminTensor, rows, 0, regComputeTimes, reservedRegComputeTimes, j * info.computeR);

            CopyOut(yGm, yTensor, rows, info.computeLength, offset);
            CopyOut(argminGm, argminTensor, rows, info.computeLength, offset);
        }
    }
}

template <typename T>
__aicore__ inline void Cummin<T>::ComputeN(CumminSplitInfo info)
{
    InitBuffers(info.computeN, 1, false);
    allocLength = info.computeN;

    for (auto i = 0; i < mRowsPerCore; i++) {
        auto length = info.computeN;
        auto regComputeTimes = Ops::Base::CeilDiv(info.computeN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
        auto reservedRegComputeTimes = info.computeN % regTensorPerComputeLength;

        for (auto j = 0; j < info.splitN; j++) {
            if (j == info.splitN - 1 && info.reservedN > 0) {
                length = info.reservedN;
                regComputeTimes = Ops::Base::CeilDiv(info.reservedN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
                reservedRegComputeTimes = info.reservedN % regTensorPerComputeLength;
            }
            for (auto k = 0; k < R; k++) {
                auto offset = i * R * N + j * info.computeN + k * N;
                if (k == 0) {
                    CopyIn(tmpXTensor, static_cast<uint16_t>(1), length, offset);
                    Duplicate<int32_t>(
                        tmpArgminTensor, 0, Ops::Base::CeilAlign(length, static_cast<int64_t>(VREGTENSOR_SIZE / dSize)));
                    CopyOut(yGm, tmpXTensor, 1, length, offset);
                    CopyOut(argminGm, tmpArgminTensor, 1, length, offset);
                    continue;
                }
                CopyIn(yTensor, static_cast<uint16_t>(1), length, offset);
                DoCompare(tmpArgminTensor, 1, 0, regComputeTimes, reservedRegComputeTimes, k);
                CopyOut(yGm, tmpXTensor, 1, length, offset);
                CopyOut(argminGm, tmpArgminTensor, 1, length, offset);
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cummin<T>::ComputeReservedM(GM_ADDR x, GM_ADDR y, GM_ADDR argmin)
{
    InitGMBuffers(x, y, argmin, gmOffset);
    tpipe.Reset();
    stride = N - computeLength;
    mRowsPerCore = 1;
    if (tiling.reservedRows && GetBlockIdx() < corenum) {
        if (resevedMgeneralProcessInfo.isNFullyLoad) {
            ComputeR(resevedMgeneralProcessInfo);
        } else {
            ComputeN(resevedMgeneralProcessInfo);
        }
    }
}

template <typename T>
__aicore__ inline void Cummin<T>::DoCompare(
    LocalTensor<int32_t>& argminTensor, int64_t computeTimes, int64_t offset, int64_t regComputeTimes,
    uint32_t reservedRegComputeTimes, int32_t duplicate)
{
    __VEC_SCOPE__
    {
        __local_mem__ T* selectXAddr = (__local_mem__ T*)tmpXTensor.GetPhyAddr();
        __local_mem__ int32_t* selectargminAddr = (__local_mem__ int32_t*)tmpArgminTensor.GetPhyAddr();
        __local_mem__ T* yAddr = (__local_mem__ T*)yTensor.GetPhyAddr() + offset;
        __local_mem__ int32_t* argminAddr = (__local_mem__ int32_t*)argminTensor.GetPhyAddr() + offset;

        if constexpr (sizeof(T) == 4) {
            MicroAPI::RegTensor<T> xReg;
            MicroAPI::RegTensor<T> yReg;
            MicroAPI::RegTensor<int32_t> argminReg;
            MicroAPI::RegTensor<int32_t> dupReg;
            AscendC::MicroAPI::MaskReg yMaskReg =
                AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

            AscendC::MicroAPI::MaskReg cmpReg;
            for (int i = 0; i < regComputeTimes; i++) {
                if (i == regComputeTimes - 1 && reservedRegComputeTimes != 0) {
                    yMaskReg = MicroAPI::UpdateMask<T>(reservedRegComputeTimes);
                }
                int64_t tmpOffset = i * VREGTENSOR_SIZE / dSize;
                AscendC::MicroAPI::DataCopy(xReg, selectXAddr + tmpOffset);
                AscendC::MicroAPI::DataCopy(argminReg, selectargminAddr + tmpOffset);
                for (int j = 0; j < computeTimes; j++) {
                    auto yOffset = tmpOffset + j * allocLength;
                    AscendC::MicroAPI::DataCopy(yReg, yAddr + yOffset);
                    AscendC::MicroAPI::Duplicate(dupReg, j + duplicate);
                    AscendC::MicroAPI::Compare<T, CMPMODE::LT>(cmpReg, xReg, yReg, yMaskReg);
                    AscendC::MicroAPI::Select(xReg, xReg, yReg, cmpReg);
                    AscendC::MicroAPI::Select(argminReg, argminReg, dupReg, cmpReg);
                    AscendC::MicroAPI::DataCopy(yAddr + yOffset, xReg, yMaskReg);
                    AscendC::MicroAPI::DataCopy(argminAddr + yOffset, argminReg, yMaskReg);
                }
                AscendC::MicroAPI::DataCopy(selectXAddr + tmpOffset, xReg, yMaskReg);
                AscendC::MicroAPI::DataCopy(selectargminAddr + tmpOffset, argminReg, yMaskReg);
            }
        }
    }
}

} // namespace CumminRegbase

#endif