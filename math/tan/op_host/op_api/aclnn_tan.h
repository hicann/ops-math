/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_TAN_H_
#define OP_API_INC_TAN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTan的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算输入张量self中每个元素的正切值，并将结果存储在张量out中
 * 计算公式：$$ out[i] = tan(self[i]) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持 FLOAT、BFLOAT16、FLOAT16、INT32、DOUBLE、COMPLEX64、COMPLEX128, 支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor，
 * 数据类型支持 FLOAT、BFLOAT16、FLOAT16、INT32、DOUBLE、COMPLEX64、COMPLEX128, 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnTanGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceTan的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算输入张量selfRef中每个元素的正切值，并将结果存储在张量selfRef中
 * 计算公式：$$ out[i] = tan(self[i]) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(selfRef)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(selfRef)]
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持 FLOAT、FLOAT16、INT32、DOUBLE、COMPLEX64、COMPLEX128, 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceTanGetWorkspaceSize(const aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnTan的第二段接口，用于执行计算。
 *
 * 算子功能：计算输入张量self中每个元素的正切值，并将结果存储在张量out中
 * 计算公式：$$ out[i] = tan(self[i]) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnTanhGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnTan(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnInplaceTan的第二段接口，用于执行计算。
 *
 * 算子功能：计算输入张量selfRef中每个元素的正切值，并将结果存储在张量selfRef中
 * 计算公式：$$ out[i] = tan(self[i]) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(selfRef)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(selfRef)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnTanhGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceTan(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_TANH_H