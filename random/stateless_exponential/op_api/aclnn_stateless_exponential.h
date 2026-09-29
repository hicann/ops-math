/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_STATELESS_EXPONENTIAL_H_
#define OP_API_INC_STATELESS_EXPONENTIAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @param [in] self: npu device侧的aclTensor。
 * 数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和out一样，数据格式支持ND，shape必须和out一样，支持非连续的Tensor。
 * @param [in] seedTenso: npu device侧的aclTensor。生成随机数的种子，数据类型支持INT64。
 * @param [in] offsetTensor: npu device侧的aclTensor。生成随机数的偏移，数据类型支持INT64。
 * @param [in] offset: host侧的整型，随机数生成器的偏移量，它影响生成的随机数序列的位置。输入为INT64_T数据类型。
 * @param [in] lambd: 指数分布速率参数，数据类型支持DOUBLE。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnStatelessExponentialTensorGetWorkspaceSize(aclTensor* self, const aclTensor* seedTensor,
                                                                      const aclTensor* offsetTensor, int64_t offset,
                                                                      double lambd, uint64_t* workspaceSize,
                                                                      aclOpExecutor** executor);

/**
 * @brief aclnnDropoutV3的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnDropoutV3GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnStatelessExponentialTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif //  OP_API_INC_STATELESS_EXPONENTIAL_H_
