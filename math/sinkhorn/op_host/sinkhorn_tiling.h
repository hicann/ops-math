/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sinkhorn.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SINKHORN_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SINKHORN_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SinkhornTilingData)
TILING_DATA_FIELD_DEF(uint64_t, formerNum);    // former 数量
TILING_DATA_FIELD_DEF(uint64_t, formerRow);    // former cost行数
TILING_DATA_FIELD_DEF(uint64_t, formerLength); // former cost总长

TILING_DATA_FIELD_DEF(uint64_t, formerTileNum);        // former Tile数量
TILING_DATA_FIELD_DEF(uint64_t, formerLastTileRow);    // fomer last Tile行数
TILING_DATA_FIELD_DEF(uint64_t, formerLastTileLength); // fomer last Tile长度

TILING_DATA_FIELD_DEF(uint64_t, tailNum);    // tail 数量
TILING_DATA_FIELD_DEF(uint64_t, tailRow);    // tail cost行数
TILING_DATA_FIELD_DEF(uint64_t, tailLength); // tail cost总长

TILING_DATA_FIELD_DEF(uint64_t, tailTileNum);        // tail Tile数量
TILING_DATA_FIELD_DEF(uint64_t, tailLastTileRow);    // tail last Tile行数
TILING_DATA_FIELD_DEF(uint64_t, tailLastTileLength); // tail last Tile长度

TILING_DATA_FIELD_DEF(uint64_t, tileRow);    // Tile行数(非Last)
TILING_DATA_FIELD_DEF(uint64_t, tileLength); // Tile长度(非Last)

TILING_DATA_FIELD_DEF(uint64_t, totalRow);        // 总行数
TILING_DATA_FIELD_DEF(uint64_t, totalCol);        // 总列数
TILING_DATA_FIELD_DEF(uint64_t, totalColAligned); // 对齐后的总列数

TILING_DATA_FIELD_DEF(float, tol); // 误差
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Sinkhorn, SinkhornTilingData)
struct SinkhornCompileInfo {
    uint64_t aivNum = 40;                         // AIV核数
    uint64_t sysWorkspaceSize = 16 * 1024 * 1024; // 系统WorkSpace大小
    uint64_t ubSize = 196608;                     // UB大小
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SINKHORN_H