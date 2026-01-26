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
 * \file truncated_normal_v2_simt.h
 * \brief truncated_normal_v2_simt
 */

#ifndef RANDOM_TRUNCATED_NORMAL_V2_SIMT_H_
#define RANDOM_TRUNCATED_NORMAL_V2_SIMT_H_

#include "kernel_operator.h"
#include "truncated_normal_v2_base.h"

namespace TruncatedNormalV2 {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD = 128;
constexpr uint32_t THREAD_LAUNCH = 512;
#else
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t THREAD_LAUNCH = 512;
#endif

template <typename Y_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void SimtCompute(
    __gm__ Y_T* yGm, int64_t offsetValue, int64_t outputNum, uint32_t key0, uint32_t key1, uint32_t counter2,
    uint32_t counter3)
{
    uint32_t key[ALG_KEY_SIZE] = {0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0};
    key[0] = key0;
    key[1] = key1;
    counter[IDX_2] = counter2;
    counter[IDX_3] = counter3;

    int64_t groupIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    int64_t totalThreadCount = Simt::GetBlockNum() * Simt::GetThreadNum();
    int64_t offset = groupIndex * GROUP_SIZE;

    // the op is stateful, offsetGm saved the previous offset of counter
    if (offsetValue > 0) {
        Skip(offsetValue, counter);
    }
    while (offset < outputNum) {
        // Skip will update counterThread
        uint32_t counterThread[ALG_COUNTER_SIZE];
        CopyArray4(counterThread, counter);
        Skip(groupIndex * RESERVED_SAMPLES_PER_OUTPUT, counterThread);

        float results[ALG_COUNTER_SIZE];
        // GenSamples will update results
        GenSamples(results, key, counterThread);

        for (int32_t i = 0; i < GROUP_SIZE; ++i) {
            if (offset >= outputNum) {
                return;
            }
            if constexpr (IsSameType<Y_T, bfloat16_t>::value || IsSameType<Y_T, half>::value) {
                yGm[offset] = static_cast<Y_T>(results[i]);
            } else {
                yGm[offset] = results[i];
            }
            ++offset;
        }
        offset += (totalThreadCount - 1) * GROUP_SIZE;
        groupIndex += totalThreadCount;
    }
}

template <typename Y_T, typename OFFSET_T>
class TruncatedNormalV2Simt {
public:
    __aicore__ inline TruncatedNormalV2Simt(const TruncatedNormalV2TilingData* tiling) : tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR offset);
    __aicore__ inline void Process();

private:
    GlobalTensor<Y_T> y_;
    GlobalTensor<OFFSET_T> offset_;
    const TruncatedNormalV2TilingData* tilingData_;
};

template <typename Y_T, typename OFFSET_T>
__aicore__ inline void TruncatedNormalV2Simt<Y_T, OFFSET_T>::Init(GM_ADDR y, GM_ADDR offset)
{
    y_.SetGlobalBuffer((__gm__ Y_T*)(y));
    offset_.SetGlobalBuffer((__gm__ OFFSET_T*)(offset));
}

template <typename Y_T, typename OFFSET_T>
__aicore__ inline void TruncatedNormalV2Simt<Y_T, OFFSET_T>::Process()
{
    uint32_t key0 = static_cast<uint32_t>(tilingData_->newSeed);
    uint32_t key1 = static_cast<uint32_t>(tilingData_->newSeed >> ALG_RGIHT_BIT);
    uint32_t counter2 = static_cast<uint32_t>(tilingData_->newSeed2);
    uint32_t counter3 = static_cast<uint32_t>(tilingData_->newSeed2 >> ALG_RGIHT_BIT);
    int64_t outputNum = tilingData_->outputNum;
    int64_t offsetValue = offset_(0);

    SyncAll();
    AscendC::Simt::VF_CALL<SimtCompute<Y_T, OFFSET_T>>(
        AscendC::Simt::Dim3{USED_THREAD}, (__gm__ Y_T*)(y_.GetPhyAddr()), offsetValue, outputNum, key0, key1, counter2,
        counter3);
    if (GetBlockIdx() == 0) {
        // update output offset
        offset_(0) = offset_(0) + static_cast<OFFSET_T>(outputNum * RESERVED_SAMPLES_PER_OUTPUT);
        AscendC::DataCacheCleanAndInvalid<
            OFFSET_T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(offset_[0]);
    }
}

} // namespace TruncatedNormalV2

#endif // RANDOM_TRUNCATED_NORMAL_V2_SIMT_H_
