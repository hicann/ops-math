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
 * \file random_unified_tiling_data_arch35.h
 * \brief tiling base data
 */

#ifndef RANDOM_UNIFIED_TILING_DATA_STRUCT_H
#define RANDOM_UNIFIED_TILING_DATA_STRUCT_H

#include <sstream>
class RandomUnifiedTilingDataStruct {
public:
    int64_t usedCoreNum = 0;
    int64_t normalCoreProNum = 0;
    int64_t tailCoreProNum = 0;
    int64_t singleBufferSize = 0;
    uint32_t key[2] = {0};
    uint32_t counter[4] = {0};
    int64_t outputSize = 1;
    int64_t probTensorSize = 0;
    int64_t sharedTmpBufSize = 0;
    float keepProb = 0;
    uint32_t reserved = 0;
    uint32_t v3KernelMode = 0;
    
    std::string DumpTilingInfo() const {
        std::ostringstream info;
        info << "[RandomUnifiedTilingData] "
            << "usedCoreNum: " << usedCoreNum
            << ", normalCoreProNum: " << normalCoreProNum
            << ", tailCoreProNum: " << tailCoreProNum
            << ", singleBufferSize: " << singleBufferSize
            << ", key: [" << key[0] << ", " << key[1] << "]"
            << ", counter: [" << counter[0] << ", " << counter[1]
            << ", " << counter[2] << ", " << counter[3] << "]"
            << ", outputSize: " << outputSize
            << ", probTensorSize: " << probTensorSize
            << ", sharedTmpBufSize: " << sharedTmpBufSize
            << ", keepProb: " << keepProb
            << ", v3KernelMode: " << v3KernelMode;
        return info.str();
    }
};

class RandomUnifiedSimtTilingDataStruct {
public:
    int64_t usedCoreNum = 0;
    int64_t outputSize = 0;
    int64_t seed = 0;
    int64_t offset = 0;
    int64_t ubSize = 0;
    int64_t extraInt64Param1 = 0; // 扩展字段：供需要额外参数的算子使用
    float prob = 0;
    float extraFloat32Param1 = 0; // 扩展字段：供需要额外参数的算子使用

    std::string DumpTilingInfo() const {
        std::ostringstream info;
        info << "[RandomUnifiedSimtTilingData] "
            << "usedCoreNum: " << usedCoreNum
            << ", outputSize: " << outputSize
            << ", seed: " << seed
            << ", offset: " << offset
            << ", ubSize: " << ubSize
            << ", extraInt64Param1: " << extraInt64Param1
            << ", prob: " << prob
            << ", extraFloat32Param1: " << extraFloat32Param1;
        return info.str();
    }
};
#endif
