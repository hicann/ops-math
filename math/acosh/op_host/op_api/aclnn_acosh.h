/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACOSH_H_
#define OP_API_INC_ACOSH_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAcosh的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对输入Tensor中的每个元素进行反双曲余弦操作后输出。
 * 计算公式：
 * $$ out_{i}=cosh^{-1}(self_{i}) $$
 *
 * @param [in] self: 输入Tensor，数据类型支持整型，浮点，复数类型，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: 输出Tensor，数据类型支持浮点，复数类型，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAcoshGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief aclnnAcosh的第二段接口，用于执行计算。
 *
 * 算子功能：对输入Tensor中的每个元素进行反双曲余弦操作后输出。
 * 计算公式：
 * $$ out_{i}=cosh^{-1}(self_{i}) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAcoshGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAcosh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceAcosh的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对输入Tensor中的每个元素进行反双曲余弦操作后输出。
 * 计算公式：
 * $$ selfRef_{i}=cosh^{-1}(selfRef_{i}) $$
 *
 * @param [in] selfRef: 输入Tensor，数据类型支持浮点，复数类型，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAcoshGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnInplaceAcosh的第二段接口，用于执行计算。
 *
 * 算子功能：对输入Tensor中的每个元素进行反双曲余弦操作后输出。
 * 计算公式：
 * $$ selfRef_{i}=cosh^{-1}(selfRef_{i}) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceAcoshGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAcosh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ACOSH_H_
