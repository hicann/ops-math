/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file reduce_nansum_tiling_data.h
 * \brief ReduceNansum TilingData 结构体定义
 */

#ifndef _REDUCE_NANSUM_TILING_DATA_H_
#define _REDUCE_NANSUM_TILING_DATA_H_

struct ReduceNansumTilingData {
    // 3D 抽象参数
    int64_t a1Count = 0;          // A1 维度大小
    int64_t rCount = 0;           // R（归约）维度大小
    int64_t a0Count = 0;          // A0 维度大小

    // 多核切分参数
    int64_t usedCoreNum = 0;      // 使用的核数
    int64_t tilesPerCore = 0;     // 每核处理的 tile 数（AR: 行数; ARA: tile数）
    int64_t tailCoreTiles = 0;    // 尾核处理的 tile 数

    // UB 切分参数
    int64_t tileA0Len = 0;        // ARA 模板：每个 tile 处理的 A0 个数
    int64_t rChunkSize = 0;       // 分载模式：每个 chunk 处理的 R 行数
    int64_t numChunks = 0;        // 分载模式：chunk 总数
    int64_t lastChunkSize = 0;    // 分载模式：最后一个 chunk 的 R 行数

    // 对齐参数
    int64_t rLengthAlign = 0;     // AR 模板：R 对齐到 32 字节后的长度
    int64_t alignedCols = 0;      // ARA 模板：tileA0Len 对齐到 32 字节后的长度
    int64_t a0Outer = 0;          // ARA 模板：A0 方向的 tile 数

    // 数据类型信息
    int64_t originalA0 = 0;       // 原始 A0 大小（用于 stride 计算）
    int64_t tmpBufSize = 0;       // ReduceSum 临时 buffer 大小

    // 非连续多轴归约参数（AR 模板专用）
    // 当归约轴不连续时（如 dim=[0,2]），每行在 GM 中不是连续的，
    // 而是由 copyBlockCount 个大小为 copyBlockLen 的连续块组成，
    // 相邻块之间的 GM 间距为 copySrcStride 字节
    int64_t copyBlockCount = 0;    // DataCopyPad blockCount（0 表示不使用，用默认连续模式）
    int64_t copyBlockLen = 0;      // DataCopyPad blockLen（字节）
    int64_t copySrcStride = 0;     // DataCopyPad srcStride（字节）
    int64_t outputStride = 0;      // 输出元素之间的 GM 步长（元素数）- 仅单非归约维度有效

    // 非归约维度的 GM 步长信息（用于多个非归约维度的 GM 偏移计算）
    // nonReduceDimCount: 非归约维度的数量（在首尾归约轴之间 + 之前 + 之后的非归约维度）
    // nonReduceDimSizes[i]: 第 i 个非归约维度的大小
    // nonReduceGmStrides[i]: 第 i 个非归约维度在 GM 中的步长（元素数）
    int64_t nonReduceDimCount = 0;         // 非归约维度数
    int64_t nonReduceDimSizes[8] = {0};    // 非归约维度大小（最多 8 维）
    int64_t nonReduceGmStrides[8] = {0};   // 非归约维度 GM 步长（元素数）

    // 归约维度的 GM 步长信息（用于非连续多轴归约的 GM 偏移计算）
    // 当归约轴在 GM 中不连续时，kernel 需要逐个归约元素计算 GM 地址。
    // reduceDimCount: 归约维度的数量
    // reduceDimSizes[i]: 第 i 个归约维度的大小
    // reduceGmStrides[i]: 第 i 个归约维度在 GM 中的步长（元素数）
    int64_t reduceDimCount = 0;            // 归约维度数
    int64_t reduceDimSizes[8] = {0};       // 归约维度大小（最多 8 维）
    int64_t reduceGmStrides[8] = {0};      // 归约维度 GM 步长（元素数）

    // AtomicAdd 多核归约参数
    int64_t useAtomicAdd = 0;     // 是否使用原子加模式（0=否, 1=是）
    int64_t totalRForSplit = 0;   // 需要被多核切分的总 R 值（仅 AtomicAdd 模式）
    int64_t rPerCore = 0;         // 每核处理的 R 数量（仅 AtomicAdd 模式）
};

#endif // _REDUCE_NANSUM_TILING_DATA_H_
