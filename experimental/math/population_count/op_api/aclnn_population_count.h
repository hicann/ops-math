/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_population_count.h
 * @brief ACLNN L2 API declaration for PopulationCount
 */

#ifndef ACLNN_POPULATION_COUNT_H_
#define ACLNN_POPULATION_COUNT_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size required by aclnnPopulationCount.
 * @param x [in]  Input tensor, INT16 or UINT16.
 * @param y [in]  Output tensor, UINT8, same shape as x.
 * @param workspaceSize [out]
 * @param executor [out]
 */
ACLNN_API aclnnStatus aclnnPopulationCountGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);

/**
 * @brief Execute PopulationCount.
 */
ACLNN_API aclnnStatus aclnnPopulationCount(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_POPULATION_COUNT_H_
