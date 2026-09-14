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
 * \file equal.h
 * \brief Ascend C implementation of elementwise logical equality.
 */
#ifndef EXPERIMENTAL_MATH_EQUAL_OP_KERNEL_EQUAL_H_
#define EXPERIMENTAL_MATH_EQUAL_OP_KERNEL_EQUAL_H_

#include "equal_tiling_data.h"
#include "equal_tiling_key.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "adv_api/pad/broadcast.h"

namespace NsEqual {

using namespace AscendC;

constexpr uint32_t CONTIGUOUS_QUEUE_BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = Ops::Base::GetUbBlockSize();
constexpr uint32_t REPEAT_BYTES = 256;
constexpr uint32_t UINT32_PACKED_BYTE_COUNT = sizeof(uint32_t) / sizeof(uint8_t);
constexpr uint32_t REPEATED_BYTE_MASK = 0x01010101U;

template <typename T, uint32_t SCHEDULE_MODE>
class KernelEqual {
public:
    __aicore__ inline KernelEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const EqualTilingData& tiling)
    {
        const int64_t blockIndex = GetBlockIdx();
        tileLength_ = tiling.tileLength;
        broadcastMode_ = tiling.broadcastMode;
        rank_ = tiling.rank;
        fastBroadcastInput_ = tiling.fastBroadcastInput;
        fastMiddle_ = tiling.fastMiddle;
        fastTail_ = tiling.fastTail;
        fastPlaneLength_ = tiling.fastPlaneLength;
        fastSourceLength_ = tiling.fastSourceLength;
        for (uint32_t index = 0; index < EQUAL_MAX_BROADCAST_DIM; ++index) {
            outShape_[index] = tiling.outShape[index];
            x1Stride_[index] = tiling.x1Stride[index];
            x2Stride_[index] = tiling.x2Stride[index];
        }

        if constexpr (SCHEDULE_MODE == ELEMENTWISE_TPL_SCH_MODE_0) {
            blockLength_ = (blockIndex + 1 == tiling.blockNum) ? tiling.tailBlockLength : tiling.blockLength;
            blockOffset_ = blockIndex * tiling.blockLength;
            x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + blockOffset_, blockLength_);
            x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + blockOffset_, blockLength_);
            yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y) + blockOffset_, blockLength_);
            blockOffset_ = 0;
        } else {
            x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1), tiling.x1Length);
            x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), tiling.x2Length);
            yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), tiling.totalLength);
            if (broadcastMode_ == EQUAL_BROADCAST_SANDWICH || broadcastMode_ == EQUAL_BROADCAST_TAIL_REUSE) {
                const int64_t baseOuter = tiling.fastOuter / tiling.blockNum;
                const int64_t remainderOuter = tiling.fastOuter % tiling.blockNum;
                const int64_t outerCount = baseOuter + static_cast<int64_t>(blockIndex < remainderOuter);
                const int64_t outerStart = blockIndex * baseOuter +
                                           (blockIndex < remainderOuter ? blockIndex : remainderOuter);
                blockOffset_ = outerStart * fastPlaneLength_;
                blockLength_ = outerCount * fastPlaneLength_;
            } else {
                blockLength_ = (blockIndex + 1 == tiling.blockNum) ? tiling.tailBlockLength : tiling.blockLength;
                blockOffset_ = blockIndex * tiling.blockLength;
            }
        }

        if constexpr (SCHEDULE_MODE == ELEMENTWISE_TPL_SCH_MODE_1) {
            if (broadcastMode_ == EQUAL_BROADCAST_SANDWICH || broadcastMode_ == EQUAL_BROADCAST_TAIL_REUSE) {
                pipe_.InitBuffer(x1Queue_, 1, tileLength_ * sizeof(T));
            } else {
                pipe_.InitBuffer(x1Queue_, 1, tileLength_ * sizeof(T));
                pipe_.InitBuffer(x2Queue_, 1, tileLength_ * sizeof(T));
            }
        } else {
            pipe_.InitBuffer(x1Queue_, CONTIGUOUS_QUEUE_BUFFER_NUM, tileLength_ * sizeof(T));
            pipe_.InitBuffer(x2Queue_, CONTIGUOUS_QUEUE_BUFFER_NUM, tileLength_ * sizeof(T));
        }
        if constexpr (SCHEDULE_MODE == ELEMENTWISE_TPL_SCH_MODE_1) {
            pipe_.InitBuffer(yQueue_, 1, tileLength_ * sizeof(int8_t));
        } else {
            pipe_.InitBuffer(yQueue_, CONTIGUOUS_QUEUE_BUFFER_NUM, tileLength_ * sizeof(int8_t));
        }

        if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int16_t>) {
            pipe_.InitBuffer(tmp1_, tileLength_ * sizeof(float));
            pipe_.InitBuffer(tmp2_, tileLength_ * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            pipe_.InitBuffer(tmp1_, tileLength_ * sizeof(half));
            pipe_.InitBuffer(tmp2_, tileLength_ * sizeof(half));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            pipe_.InitBuffer(tmp1_, tileLength_ * sizeof(uint32_t));
            pipe_.InitBuffer(tmp2_, tileLength_ * sizeof(uint32_t));
            pipe_.InitBuffer(tmp3_, tileLength_ * sizeof(uint32_t));
            pipe_.InitBuffer(tmp4_, tileLength_ * sizeof(uint32_t));
        } else {
            pipe_.InitBuffer(tmp1_, tileLength_ * sizeof(half));
        }

        if constexpr (SCHEDULE_MODE == ELEMENTWISE_TPL_SCH_MODE_1) {
            if (broadcastMode_ != EQUAL_BROADCAST_SANDWICH && broadcastMode_ != EQUAL_BROADCAST_TAIL_REUSE) {
                pipe_.InitBuffer(x1ScalarQueue_, 1, BLOCK_BYTES);
                pipe_.InitBuffer(x2ScalarQueue_, 1, BLOCK_BYTES);
            }
            if (broadcastMode_ == EQUAL_BROADCAST_SANDWICH) {
                cachePlaneCount_ = tileLength_ / fastPlaneLength_;
                const int64_t sourceBytes = (cachePlaneCount_ * fastSourceLength_ * sizeof(T) + BLOCK_BYTES - 1) /
                                            BLOCK_BYTES * BLOCK_BYTES;
                pipe_.InitBuffer(broadcastSourceQueue_, 1, sourceBytes);
                pipe_.InitBuffer(broadcastTmp_, tiling.fastTmpBytes);
                pipe_.InitBuffer(broadcastCache_, tileLength_ * sizeof(T));
            } else if (broadcastMode_ == EQUAL_BROADCAST_TAIL_REUSE) {
                pipe_.InitBuffer(broadcastSourceQueue_, 1, tileLength_ * sizeof(T));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (SCHEDULE_MODE == ELEMENTWISE_TPL_SCH_MODE_1) {
            if (broadcastMode_ == EQUAL_BROADCAST_TAIL_REUSE) {
                ProcessTailReuseBroadcast();
                return;
            }
            if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t> ||
                          std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
                if (broadcastMode_ == EQUAL_BROADCAST_SANDWICH) {
                    ProcessSandwichBroadcast();
                    return;
                }
            }
            ProcessBroadcast();
            return;
        }
        const int64_t loopCount = blockLength_ / tileLength_ + static_cast<int64_t>(blockLength_ % tileLength_ != 0);
        for (int64_t progress = 0; progress < loopCount; ++progress) {
            const int64_t offset = progress * tileLength_;
            const uint32_t validLength = static_cast<uint32_t>(min(tileLength_, blockLength_ - offset));
            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t validLength)
    {
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        if ((static_cast<int64_t>(validLength) * sizeof(T)) % BLOCK_BYTES == 0) {
            DataCopy(x1Local, x1Gm_[offset], validLength);
            DataCopy(x2Local, x2Gm_[offset], validLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(x1Local, x1Gm_[offset], copyParams, padParams);
            DataCopyPad(x2Local, x2Gm_[offset], copyParams, padParams);
        }
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline int64_t CalcInputOffset(int64_t outIndex, const int64_t* stride) const
    {
        int64_t inputOffset = 0;
        int64_t remain = outIndex;
        for (int32_t index = static_cast<int32_t>(rank_) - 1; index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            const int64_t dim = outShape_[dimIndex];
            const int64_t coord = remain % dim;
            remain /= dim;
            inputOffset += coord * stride[dimIndex];
        }
        return inputOffset;
    }

    __aicore__ inline uint32_t GetComputeLength(uint32_t validLength) const
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int16_t>) {
            return AlignComputeLength<float>(validLength);
        }
        return AlignComputeLength<half>(validLength);
    }

    __aicore__ inline void CopyContiguousToLocal(LocalTensor<T> dst, const GlobalTensor<T>& src, int64_t srcOffset,
                                                 uint32_t length)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void FillScalar(LocalTensor<T> dst, T value, uint32_t length)
    {
        if constexpr (std::is_same_v<T, int64_t>) {
            for (uint32_t index = 0; index < length; ++index) {
                dst.SetValue(index, value);
            }
            TEventID scalarToVectorEvent = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToVectorEvent);
            WaitFlag<HardEvent::S_V>(scalarToVectorEvent);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint32_t byteValue = static_cast<uint8_t>(value);
            const uint32_t repeatedValue = byteValue * REPEATED_BYTE_MASK;
            const uint32_t packedLength = (length + UINT32_PACKED_BYTE_COUNT - 1) / UINT32_PACKED_BYTE_COUNT;
            Duplicate(dst.template ReinterpretCast<uint32_t>(), repeatedValue, packedLength);
        } else {
            Duplicate(dst, value, length);
        }
    }

    __aicore__ inline T ReadScalar(const GlobalTensor<T>& src, int64_t srcOffset,
                                   TQue<QuePosition::VECIN, 1>& scalarQueue)
    {
        LocalTensor<T> scalarLocal = scalarQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(scalarLocal, src[srcOffset], copyParams, padParams);
        scalarQueue.EnQue(scalarLocal);
        scalarLocal = scalarQueue.DeQue<T>();
        TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        const T value = scalarLocal.GetValue(0);
        scalarQueue.FreeTensor(scalarLocal);
        return value;
    }

    __aicore__ inline void BroadcastScalarToLocal(LocalTensor<T> dst, const GlobalTensor<T>& src, int64_t srcOffset,
                                                  uint32_t fillLength, TQue<QuePosition::VECIN, 1>& scalarQueue)
    {
        FillScalar(dst, ReadScalar(src, srcOffset, scalarQueue), fillLength);
    }

    __aicore__ inline void CopyBroadcastInput(int64_t outStart, uint32_t validLength, uint32_t computeLength)
    {
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        if (broadcastMode_ == EQUAL_BROADCAST_X1_SCALAR) {
            FillScalar(x1Local, x1ScalarValue_, computeLength);
            CopyContiguousToLocal(x2Local, x2Gm_, outStart, validLength);
        } else if (broadcastMode_ == EQUAL_BROADCAST_X2_SCALAR) {
            CopyContiguousToLocal(x1Local, x1Gm_, outStart, validLength);
            FillScalar(x2Local, x2ScalarValue_, computeLength);
        } else {
            const int64_t x1Offset = CalcInputOffset(outStart, x1Stride_);
            const int64_t x2Offset = CalcInputOffset(outStart, x2Stride_);
            if (x1Stride_[rank_ - 1] == 0) {
                BroadcastScalarToLocal(x1Local, x1Gm_, x1Offset, computeLength, x1ScalarQueue_);
            } else {
                CopyContiguousToLocal(x1Local, x1Gm_, x1Offset, validLength);
            }
            if (x2Stride_[rank_ - 1] == 0) {
                BroadcastScalarToLocal(x2Local, x2Gm_, x2Offset, computeLength, x2ScalarQueue_);
            } else {
                CopyContiguousToLocal(x2Local, x2Gm_, x2Offset, validLength);
            }
        }
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void ProcessBroadcast()
    {
        if (broadcastMode_ == EQUAL_BROADCAST_X1_SCALAR) {
            x1ScalarValue_ = ReadScalar(x1Gm_, 0, x1ScalarQueue_);
        } else if (broadcastMode_ == EQUAL_BROADCAST_X2_SCALAR) {
            x2ScalarValue_ = ReadScalar(x2Gm_, 0, x2ScalarQueue_);
        }
        int64_t outStart = blockOffset_;
        int64_t remainLength = blockLength_;
        while (remainLength > 0) {
            int64_t segmentLength = min(tileLength_, remainLength);
            if (broadcastMode_ == EQUAL_BROADCAST_GENERAL && rank_ > 0) {
                const int64_t tailDim = outShape_[rank_ - 1];
                segmentLength = min(segmentLength, tailDim - outStart % tailDim);
            }
            const uint32_t validLength = static_cast<uint32_t>(segmentLength);
            const uint32_t computeLength = GetComputeLength(validLength);
            CopyBroadcastInput(outStart, validLength, computeLength);
            Compute(validLength);
            CopyOut(outStart, validLength);
            outStart += validLength;
            remainLength -= validLength;
        }
    }

    __aicore__ inline void PrepareSandwichBroadcast()
    {
        LocalTensor<T> sourceLocal = broadcastSourceQueue_.AllocTensor<T>();
        const GlobalTensor<T>& sourceGm = fastBroadcastInput_ == 1 ? x1Gm_ : x2Gm_;
        CopyContiguousToLocal(sourceLocal, sourceGm, 0, static_cast<uint32_t>(fastSourceLength_));
        broadcastSourceQueue_.EnQue(sourceLocal);
        sourceLocal = broadcastSourceQueue_.DeQue<T>();
        if (cachePlaneCount_ > 1) {
            TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(eventId);
            WaitFlag<HardEvent::MTE2_S>(eventId);
            for (int64_t plane = 1; plane < cachePlaneCount_; ++plane) {
                for (int64_t index = 0; index < fastSourceLength_; ++index) {
                    sourceLocal.SetValue(plane * fastSourceLength_ + index, sourceLocal.GetValue(index));
                }
            }
            TEventID scalarToVectorEvent = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(scalarToVectorEvent);
            WaitFlag<HardEvent::S_V>(scalarToVectorEvent);
        }
        const uint32_t repeatedMiddle = static_cast<uint32_t>(cachePlaneCount_ * fastMiddle_);
        const uint32_t dstShape[2] = {repeatedMiddle, static_cast<uint32_t>(fastTail_)};
        const uint32_t srcShape[2] = {repeatedMiddle, 1};
        LocalTensor<uint8_t> broadcastTmp = broadcastTmp_.Get<uint8_t>();
        LocalTensor<T> broadcastLocal = broadcastCache_.Get<T>();
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
            AscendC::Broadcast<T, 2, 1>(broadcastLocal, sourceLocal, dstShape, srcShape, broadcastTmp);
        } else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int16_t>) {
            AscendC::Broadcast<half, 2, 1>(broadcastLocal.template ReinterpretCast<half>(),
                                           sourceLocal.template ReinterpretCast<half>(), dstShape, srcShape,
                                           broadcastTmp);
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
            AscendC::Broadcast<float, 2, 1>(broadcastLocal.template ReinterpretCast<float>(),
                                            sourceLocal.template ReinterpretCast<float>(), dstShape, srcShape,
                                            broadcastTmp);
        }
        broadcastSourceQueue_.FreeTensor(sourceLocal);
    }

    __aicore__ inline void ProcessSandwichBroadcast()
    {
        PrepareSandwichBroadcast();
        LocalTensor<T> broadcastLocal = broadcastCache_.Get<T>();
        int64_t outOffset = blockOffset_;
        const int64_t planeCount = blockLength_ / fastPlaneLength_;
        int64_t plane = 0;
        while (plane < planeCount) {
            const int64_t currentPlaneCount = min(cachePlaneCount_, planeCount - plane);
            const uint32_t validLength = static_cast<uint32_t>(currentPlaneCount * fastPlaneLength_);
            LocalTensor<T> contiguousLocal = x1Queue_.AllocTensor<T>();
            const GlobalTensor<T>& contiguousGm = fastBroadcastInput_ == 1 ? x2Gm_ : x1Gm_;
            CopyContiguousToLocal(contiguousLocal, contiguousGm, outOffset, validLength);
            x1Queue_.EnQue(contiguousLocal);
            contiguousLocal = x1Queue_.DeQue<T>();
            LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();
            // Keep the broadcast cache as the second operand. Equality is commutative,
            // and the compute paths either preserve it or use independent scratch.
            ComputeLocal(yLocal, contiguousLocal, broadcastLocal, validLength);
            x1Queue_.FreeTensor(contiguousLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(outOffset, validLength);
            outOffset += validLength;
            plane += currentPlaneCount;
        }
    }

    __aicore__ inline void ProcessTailReuseBroadcast()
    {
        const GlobalTensor<T>& sourceGm = fastBroadcastInput_ == 1 ? x1Gm_ : x2Gm_;
        const GlobalTensor<T>& contiguousGm = fastBroadcastInput_ == 1 ? x2Gm_ : x1Gm_;
        const int64_t* sourceStride = fastBroadcastInput_ == 1 ? x1Stride_ : x2Stride_;
        const int64_t firstRow = blockOffset_ / fastTail_;
        const int64_t rowCount = blockLength_ / fastTail_;
        int64_t localRow = 0;
        while (localRow < rowCount) {
            const int64_t globalRow = firstRow + localRow;
            const int64_t sourceBase = CalcInputOffset(globalRow * fastTail_, sourceStride);
            int64_t repeatedRows = 1;
            while (localRow + repeatedRows < rowCount) {
                const int64_t nextRow = globalRow + repeatedRows;
                if (CalcInputOffset(nextRow * fastTail_, sourceStride) != sourceBase) {
                    break;
                }
                ++repeatedRows;
            }

            int64_t tailOffset = 0;
            while (tailOffset < fastTail_) {
                const uint32_t validLength = static_cast<uint32_t>(min(tileLength_, fastTail_ - tailOffset));
                LocalTensor<T> sourceLocal = broadcastSourceQueue_.AllocTensor<T>();
                CopyContiguousToLocal(sourceLocal, sourceGm, sourceBase + tailOffset, validLength);
                broadcastSourceQueue_.EnQue(sourceLocal);
                sourceLocal = broadcastSourceQueue_.DeQue<T>();

                for (int64_t repeat = 0; repeat < repeatedRows; ++repeat) {
                    const int64_t outOffset = (globalRow + repeat) * fastTail_ + tailOffset;
                    LocalTensor<T> contiguousLocal = x1Queue_.AllocTensor<T>();
                    CopyContiguousToLocal(contiguousLocal, contiguousGm, outOffset, validLength);
                    x1Queue_.EnQue(contiguousLocal);
                    contiguousLocal = x1Queue_.DeQue<T>();
                    LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();
                    // Equal is commutative. Keep the cached source as x1Local
                    // because the INT64 compute path reuses x2Local as result
                    // storage after its words have been gathered.
                    ComputeLocal(yLocal, sourceLocal, contiguousLocal, validLength);
                    x1Queue_.FreeTensor(contiguousLocal);
                    yQueue_.EnQue(yLocal);
                    CopyOut(outOffset, validLength);
                }
                broadcastSourceQueue_.FreeTensor(sourceLocal);
                tailOffset += validLength;
            }
            localRow += repeatedRows;
        }
    }

    template <typename ComputeT>
    __aicore__ inline void CompareAndMaterialize(LocalTensor<int8_t>& yLocal, const LocalTensor<ComputeT>& x1Local,
                                                 const LocalTensor<ComputeT>& x2Local,
                                                 LocalTensor<ComputeT>& resultLocal, uint32_t computeLength)
    {
        LocalTensor<uint8_t> compareMask = yLocal.ReinterpretCast<uint8_t>();
        Compare(compareMask, x1Local, x2Local, CMPMODE::EQ, computeLength);
        Duplicate(resultLocal, static_cast<ComputeT>(1), computeLength);
        Select(resultLocal, compareMask, resultLocal, static_cast<ComputeT>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               computeLength);
    }

    __aicore__ inline void ComputeLocal(LocalTensor<int8_t>& yLocal, LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
                                        uint32_t validLength)
    {
        if constexpr (std::is_same_v<T, half>) {
            const uint32_t computeLength = AlignComputeLength<half>(validLength);
            LocalTensor<half> resultLocal = tmp1_.Get<half>();
            LocalTensor<uint8_t> compareMask = yLocal.ReinterpretCast<uint8_t>();
            Compare(compareMask, x1Local, x2Local, CMPMODE::EQ, computeLength);
            Duplicate(resultLocal, static_cast<half>(1), computeLength);
            Select(resultLocal, compareMask, resultLocal, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   computeLength);
            Cast(yLocal, resultLocal, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<T, float>) {
            const uint32_t computeLength = AlignComputeLength<float>(validLength);
            LocalTensor<half> resultHalf = tmp1_.Get<half>();
            LocalTensor<uint8_t> compareMask = yLocal.ReinterpretCast<uint8_t>();
            Compare(compareMask, x1Local, x2Local, CMPMODE::EQ, computeLength);
            Duplicate(x1Local, 1.0F, computeLength);
            Select(x1Local, compareMask, x1Local, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
            Cast(resultHalf, x1Local, RoundMode::CAST_NONE, computeLength);
            Cast(yLocal, resultHalf, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int16_t>) {
            // Every INT16 value is exactly representable as float. BF16 also
            // widens losslessly, so equality is preserved by these casts.
            const uint32_t computeLength = AlignComputeLength<float>(validLength);
            LocalTensor<float> x1Float = tmp1_.Get<float>();
            LocalTensor<float> x2Float = tmp2_.Get<float>();
            Cast(x1Float, x1Local, RoundMode::CAST_NONE, computeLength);
            Cast(x2Float, x2Local, RoundMode::CAST_NONE, computeLength);
            LocalTensor<uint8_t> compareMask = yLocal.ReinterpretCast<uint8_t>();
            Compare(compareMask, x1Float, x2Float, CMPMODE::EQ, computeLength);
            LocalTensor<half> resultHalf = x1Float.ReinterpretCast<half>();
            LocalTensor<half> onesHalf = x2Float.ReinterpretCast<half>();
            Duplicate(onesHalf, static_cast<half>(1), computeLength);
            Select(resultHalf, compareMask, onesHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   computeLength);
            Cast(yLocal, resultHalf, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            const uint32_t computeLength = AlignComputeLength<half>(validLength);
            LocalTensor<half> x1Half = tmp1_.Get<half>();
            LocalTensor<half> x2Half = tmp2_.Get<half>();
            if constexpr (std::is_same_v<T, bool>) {
                Cast(x1Half, x1Local.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, computeLength);
                Cast(x2Half, x2Local.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, computeLength);
            } else {
                Cast(x1Half, x1Local, RoundMode::CAST_NONE, computeLength);
                Cast(x2Half, x2Local, RoundMode::CAST_NONE, computeLength);
            }
            CompareAndMaterialize(yLocal, x1Half, x2Half, x1Half, computeLength);
            Cast(yLocal, x1Half, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
            const uint32_t computeLength = AlignComputeLength<int32_t>(validLength);
            LocalTensor<int32_t> x1Int = x1Local.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> x2Int = x2Local.template ReinterpretCast<int32_t>();
            LocalTensor<half> resultHalf = tmp1_.Get<half>();
            LocalTensor<uint8_t> compareMask = yLocal.ReinterpretCast<uint8_t>();
            Compare(compareMask, x1Int, x2Int, CMPMODE::EQ, computeLength);
            Duplicate(resultHalf, static_cast<half>(1), computeLength);
            Select(resultHalf, compareMask, resultHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   computeLength);
            Cast(yLocal, resultHalf, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            // A2 has no native int64 vector compare. Gather the low and high
            // 32-bit words and compare each word independently. Combining the
            // two boolean results is bit-exact for the complete int64 domain.
            const uint32_t wordLength = validLength * 2;
            const uint32_t resultComputeLength = AlignComputeLength<half>(validLength);
            LocalTensor<uint32_t> x1Low = tmp1_.Get<uint32_t>();
            LocalTensor<uint32_t> x1High = tmp2_.Get<uint32_t>();
            LocalTensor<uint32_t> x2Low = tmp3_.Get<uint32_t>();
            LocalTensor<uint32_t> x2High = tmp4_.Get<uint32_t>();
            uint64_t gatheredCount = 0;
            constexpr uint32_t wordsPerRepeat = REPEAT_BYTES / sizeof(uint32_t);
            const uint16_t gatherRepeats = static_cast<uint16_t>((wordLength + wordsPerRepeat - 1) / wordsPerRepeat);
            const GatherMaskParams gatherParams{1, gatherRepeats, 8, 0};
            GatherMask(x1Low, x1Local.template ReinterpretCast<uint32_t>(), static_cast<uint8_t>(1), true,
                       wordsPerRepeat, gatherParams, gatheredCount);
            GatherMask(x1High, x1Local.template ReinterpretCast<uint32_t>(), static_cast<uint8_t>(2), true,
                       wordsPerRepeat, gatherParams, gatheredCount);
            PipeBarrier<PIPE_V>();
            GatherMask(x2Low, x2Local.template ReinterpretCast<uint32_t>(), static_cast<uint8_t>(1), true,
                       wordsPerRepeat, gatherParams, gatheredCount);
            GatherMask(x2High, x2Local.template ReinterpretCast<uint32_t>(), static_cast<uint8_t>(2), true,
                       wordsPerRepeat, gatherParams, gatheredCount);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint8_t> compareMask = yLocal.ReinterpretCast<uint8_t>();
            LocalTensor<half> resultStorage = x2Local.template ReinterpretCast<half>();
            LocalTensor<half> lowEqual = resultStorage;
            LocalTensor<half> highEqual = resultStorage[resultComputeLength];
            Compare(compareMask, x1Low.ReinterpretCast<int32_t>(), x2Low.ReinterpretCast<int32_t>(), CMPMODE::EQ,
                    resultComputeLength);
            Duplicate(lowEqual, static_cast<half>(1), resultComputeLength);
            Select(lowEqual, compareMask, lowEqual, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   resultComputeLength);
            PipeBarrier<PIPE_V>();
            Compare(compareMask, x1High.ReinterpretCast<int32_t>(), x2High.ReinterpretCast<int32_t>(), CMPMODE::EQ,
                    resultComputeLength);
            Duplicate(highEqual, static_cast<half>(1), resultComputeLength);
            Select(highEqual, compareMask, highEqual, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   resultComputeLength);
            Mul(lowEqual, lowEqual, highEqual, resultComputeLength);
            Cast(yLocal, lowEqual, RoundMode::CAST_NONE, resultComputeLength);
        }
    }

    __aicore__ inline void Compute(uint32_t validLength)
    {
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();
        ComputeLocal(yLocal, x1Local, x2Local, validLength);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t validLength)
    {
        LocalTensor<int8_t> yLocal = yQueue_.template DeQue<int8_t>();
        if (validLength % BLOCK_BYTES == 0) {
            DataCopy(yGm_[offset], yLocal, validLength);
        } else {
            DataCopyExtParams copyParams{1, validLength, 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        yQueue_.FreeTensor(yLocal);
    }

    template <typename ComputeT>
    __aicore__ inline uint32_t AlignComputeLength(uint32_t validLength) const
    {
        constexpr uint32_t alignElements = REPEAT_BYTES / sizeof(ComputeT);
        return (validLength + alignElements - 1) / alignElements * alignElements;
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECIN, 1> x1ScalarQueue_;
    TQue<QuePosition::VECIN, 1> x2ScalarQueue_;
    TQue<QuePosition::VECIN, 1> broadcastSourceQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<QuePosition::VECCALC> tmp1_;
    TBuf<QuePosition::VECCALC> tmp2_;
    TBuf<QuePosition::VECCALC> tmp3_;
    TBuf<QuePosition::VECCALC> tmp4_;
    TBuf<QuePosition::VECCALC> broadcastTmp_;
    TBuf<QuePosition::VECCALC> broadcastCache_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<int8_t> yGm_;
    int64_t blockLength_ = 0;
    int64_t blockOffset_ = 0;
    int64_t tileLength_ = 0;
    uint32_t broadcastMode_ = EQUAL_BROADCAST_CONTIGUOUS;
    uint32_t rank_ = 0;
    uint32_t fastBroadcastInput_ = 0;
    int64_t outShape_[EQUAL_MAX_BROADCAST_DIM] = {};
    int64_t x1Stride_[EQUAL_MAX_BROADCAST_DIM] = {};
    int64_t x2Stride_[EQUAL_MAX_BROADCAST_DIM] = {};
    int64_t fastMiddle_ = 0;
    int64_t fastTail_ = 0;
    int64_t fastPlaneLength_ = 0;
    int64_t fastSourceLength_ = 0;
    int64_t cachePlaneCount_ = 1;
    T x1ScalarValue_{};
    T x2ScalarValue_{};
};

} // namespace NsEqual
#endif // EXPERIMENTAL_MATH_EQUAL_OP_KERNEL_EQUAL_H_
