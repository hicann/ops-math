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
 * \file sort_with_index.h
 * \brief
 */
#ifndef MATH_SORT_WITH_INDEX_OP_HOST_OP_API_SORT_WITH_INDEX_H_
#define MATH_SORT_WITH_INDEX_OP_HOST_OP_API_SORT_WITH_INDEX_H_

#include "opdev/op_executor.h"
#include "opdev/fast_vector.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> SortWithIndex(const aclTensor* self, const aclTensor* index, int64_t axis, bool descending, bool stable,
    aclOpExecutor* executor);
}

#endif // MATH_SORT_WITH_INDEX_OP_HOST_OP_API_SORT_WITH_INDEX_H_
