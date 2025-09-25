/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hans_decode_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_DECODED_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_DECODED_H
#include "register/tilingdata_base.h"

namespace optiling {
struct HansDecodeCompileInfo {
};

BEGIN_TILING_DATA_DEF(HansDecodeTilingData)
TILING_DATA_FIELD_DEF(int64_t, mantissaByteSize);
TILING_DATA_FIELD_DEF(int64_t, fixedByteSize);
TILING_DATA_FIELD_DEF(int64_t, recoverExpByteSize);
TILING_DATA_FIELD_DEF(int64_t, recoverByteSize);
TILING_DATA_FIELD_DEF(bool, reshuff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HansDecode, HansDecodeTilingData)
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_DECODED_H