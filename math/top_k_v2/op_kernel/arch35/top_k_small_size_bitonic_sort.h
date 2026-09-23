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
 * \file top_k_small_size_bitonic_sort.h
 * \brief BITONIC-compatible candidate finalize helpers for sorted TopK with 2 <= k <= 32.
 *
 * 文件整体架构概述：
 * 本文件实现了 TopK 算子在 k 较小 (2 <= k <= 32) 时的"收尾排序"逻辑。
 * 双调排序网络 (Bitonic Sort Network) 规模固定为 32，利用 SIMD 向量寄存器或
 * SIMT warp 级通信完成并行 compare-swap，将候选元素排成有序序列。
 *
 * 两条实现路线：
 *   1. Reg-based (SIMD) 路线：基于 Reg:: 向量寄存器 API，直接操作 UB 内存，
 *      针对不同位宽 (B16/B32/B64) 做特化优化，性能最优。
 *   2. SIMT (warp 级) 路线：基于 asc_shfl_xor/asc_ballot/__popc 标量线程模型，
 *      作为 1 字节类型或非 uint32 索引类型的回退路径。
 * sizeof(T) == 1U (int8/uint8) 不能走 Reg 路径的原因：
 *   arch35 的 SIMD Reg:: API (Gather/Compare/Select/LoadAlign 等) 最小操作位宽为
 *   16 位，不支持 8 位 RegTensor。因此 1 字节类型只能回退到 SIMT 标量线程路径，
 *   用 asc_shfl_xor 做 warp 内通信替代 Reg::Gather。这是硬件 ISA 的功能限制，
 *   而非性能选择。
 */
#ifndef TOP_K_SMALL_SIZE_BITONIC_SORT_H
#define TOP_K_SMALL_SIZE_BITONIC_SORT_H

#include "top_k_small_bitonic_common.h"
#include "top_k_small_bitonic_reg_finalize.h"
#include "top_k_small_bitonic_packed_simt.h"

#endif // TOP_K_SMALL_SIZE_BITONIC_SORT_H
