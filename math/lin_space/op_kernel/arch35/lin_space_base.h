/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_base.h
 * \brief
 */

#ifndef LIN_SPACE_BASE_H
#define LIN_SPACE_BASE_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"

namespace LinSpace
{
using namespace AscendC;
constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t DB_BUFFER = 2;
}  // namespace LinSpace
#endif  // LIN_SPACE_BASE_H
