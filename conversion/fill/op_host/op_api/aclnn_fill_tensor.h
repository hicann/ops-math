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

#ifndef OP_API_INC_LEVEL2_ACLNN_FILL_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_FILL_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：对Tensor各个位置赋予value值
 * 计算公式：
 *   不涉及
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *    A[(selfRef)] --> B(生成AclTensor: dims) -->C([l0op::Fill])
 *    D[(value)] --> C
 *    C --> E([l0op::ViewCopy])--> F[(out)]
 * ```
 */

/**
 * @brief aclnnInplaceFillTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、
 * COMPLEX64、COMPLEX128、BOOL、BFLOAT16，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] value: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、
 * COMPLEX64、COMPLEX128、BOOL、BFLOAT16，且数据类型需要能转换成selfRef的数据类型，数据格式支持ND，数据维度只能是0D或者
 * size=1的1D。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* value, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceFillTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspacSize: 在npu device侧申请的workspace大小，由第一段接口aclnnFillTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceFillTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FILL_TENSOR_H_