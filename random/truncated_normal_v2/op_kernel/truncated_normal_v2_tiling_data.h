/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file truncated_normal_v2_tiling_data.h
 * \brief truncated_normal_v2_tiling_data
 */

#ifndef RANDOM_TRUNCATED_NORMAL_V2_TILING_DATA_H_
#define RANDOM_TRUNCATED_NORMAL_V2_TILING_DATA_H_

class TruncatedNormalV2TilingData {
public:
    int64_t newSeed = 0;
    int64_t newSeed2 = 0;
    int64_t outputNum = 0;
};
#endif // RANDOM_TRUNCATED_NORMAL_V2_TILING_DATA_H_
