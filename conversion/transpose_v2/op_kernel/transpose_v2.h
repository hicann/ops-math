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
 * \file transpose_v2.h
 * \brief
 */

#ifndef ASCEND_TRANSPOSE_V2_H
#define ASCEND_TRANSPOSE_V2_H

#include "kernel_operator.h"

namespace TransposeV2 {
constexpr uint32_t MAX_UB_SIZE = 192 * 1024 - 256;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t TRANS_BLOCK = 16;
} // namespace TransposeV2
#endif
