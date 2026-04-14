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
 * \file drop_out_v3_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DROP_OUT_V3_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DROP_OUT_V3_H_

#include <cstdint>
#include <vector>
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_api/op_util.h"

namespace optiling {

constexpr int64_t TILING_ARRAY_LEN_EIGHT = 8;
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr int64_t DCACHE_SIZE = 32768;
constexpr int64_t ALIGNMENT_32 = 32;

BEGIN_TILING_DATA_DEF(DropOutV3TilingData)
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, ubSize);
  TILING_DATA_FIELD_DEF(int64_t, tilingKey);
  TILING_DATA_FIELD_DEF(int64_t, seed);
  TILING_DATA_FIELD_DEF(float, prob);
  TILING_DATA_FIELD_DEF(int64_t, offset);
  TILING_DATA_FIELD_DEF(int64_t, elementNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DropOutV3, DropOutV3TilingData)

struct DropOutV3CompileInfo {
  int32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DROP_OUT_V3_H_