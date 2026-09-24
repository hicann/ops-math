/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARALLEL_CONCAT_TILING_STRUCT_H
#define PARALLEL_CONCAT_TILING_STRUCT_H

#include <cstdint>

constexpr uint64_t SIMT_ROW_BYTES_THRESHOLD = 64;

struct ParallelConcatTilingData {
    // —— 合轴视图 [N, L] 标量（L = 单输入元素数 = ∏(d_1..d_k)；k=0 时 L=1；任一 d_m=0 时 L=0 → 空张量）——
    uint64_t n;          // 动态输入个数 N（host 已校验 N == shape[0] == len(values)） [count]
    uint64_t rowElems;   // 单输入元素数 L（展平行宽度） [elements]
    uint64_t rowBytes;   // 单输入字节数 = L × dtypeSize [bytes]
    uint64_t totalBytes; // 输出总字节 = N × rowBytes（空张量判据 totalBytes == 0） [bytes]
    uint8_t dtypeSize;   // 元素字节宽（1/2/4/8） [bytes/element]

    // —— 多核切分（N 主序展平 chunk 切分 · baseC/remC 非均匀满核；两分支统一多核）——
    uint32_t numActiveCores; // 实际参与核数 = min(totalChunks, coreNum)（= SetBlockDim 值；
                             // chunksPerRow / remC 由 kernel 现算不设字段） [cores]
    uint32_t perCoreChunks;  // 每核基础 chunk 数 baseC = totalChunks / numActiveCores；
                             // 前 remC = totalChunks % numActiveCores 个核处理 baseC+1 个 [chunks]

    // —— UB buffer（双缓冲）——
    uint32_t bufferSize; // 单块字节数（不含双缓冲）；SIMD 分支 InitBuffer(dataQue, 2, bufferSize)
                         // （TQueBind VECIN→VECOUT 共享双缓冲）；
                         // host 自适应三档，硬约束：32B 对齐、2 × bufferSize <= ubSize、
                         // bufferSize >= SIMT_ROW_BYTES_THRESHOLD [bytes]
};

#endif
