/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_data_def.h
 * \brief
 */
#ifndef TILING_DATA_DEF_H
#define TILING_DATA_DEF_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

inline void InitPadV3GradReplicationTilingData(uint8_t* tiling, PadV3GradReplicationTilingData* data)
{
    memcpy(data, tiling, sizeof(PadV3GradReplicationTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    PadV3GradReplicationTilingData tilingData; \
    InitPadV3GradReplicationTilingData(tilingArg, &tilingData)
#endif