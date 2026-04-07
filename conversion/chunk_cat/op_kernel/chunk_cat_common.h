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
 * \file chunk_cat_common.h
 * \brief
 */

#ifndef _CHUNK_CAT_COMMON_DATA_H_
#define _CHUNK_CAT_COMMON_DATA_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "chunk_cat_tiling_data.h"

constexpr uint32_t UB_BLOCK_SIZE = 32; // UB块大小
constexpr uint32_t TRANS_BLOCK = 16; // 转置行数
constexpr uint32_t HALF = 2; // 半对齐/UB对半切分

struct TensorInfo {
    bool isSplit{false};
    int64_t chunkDimSize{0};
    int64_t chunkCol{0};
    int64_t chunkRow{0};
    int64_t chunkRowAlign{0};
    int64_t originCol{1};
    int64_t tensorCol{0};
    int64_t splitCol{0};
    int64_t splitColAlign{0};
    int64_t startOffset{0};
};

struct UbLoopInfo {
    bool isAllZero{true};
    int64_t count{0};
    int64_t currentUbRowFactor{0};
    int64_t currentUbColFactor{0};
    int64_t ubRowGroup{0};
    int64_t ubColGroup{0};
    int64_t totalUbCol{0};
    int64_t totalUbColAlign{0};
    int64_t colStart{0};
    int64_t rowStart{0};
    int64_t preCatCol{0};
    int64_t* inputCol;
};

#endif // _CHUNK_CAT_COMMON_DATA_H_