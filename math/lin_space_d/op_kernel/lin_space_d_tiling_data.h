/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_d_tiling_data.h
 * \brief tiling data struct
 */
#ifndef LIN_SPACE_D_TILING_DATA_H
#define LIN_SPACE_D_TILING_DATA_H

struct LinSpaceDTilingData {
    uint32_t totalLength;   
    uint32_t formerNum;
    uint32_t formerLength;        
    uint32_t formerTileNum;  
    uint32_t formerLastTileLength;
    uint32_t tailLength;          
    uint32_t tailTileNum;
    uint32_t tailLastTileLength; 
};
#endif // LIN_SPACE_D_TILING_DATA_H