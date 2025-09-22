/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OP_API_INC_GTSCALAR_H_
#define OP_API_INC_GTSCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGt的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：判断输入Tensor中的每个元素是否大于other
 * Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于判断是否成立 计算公式： $$ out_{i}= (self_i >
 * other) ? True : False $$
 *
 * @param [in] self: 待进行gt计算的入参,npu device侧的aclTensor，
 * 数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: 待进行gt计算的入参,aclScalar。
 * @param [in] out: npu device侧的aclTensor，输出一个数据类型为BOOL类型的Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGtScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGtScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGtScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGtScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceGtScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：判断输入Tensor中的每个元素是否大于other Scalar的值
 * 计算公式： $$ selfRef_{i} = (selfRef_{i} > other_{i}) ? True : False $$
 *
 * @param [in] selfRef: 待进行gt本地计算的入参,npu device侧的aclTensor，
 * 数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: 待进行gt计算的入参,aclScalar。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceGtScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceGtScalar的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceGtScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceGtScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GTTENSOR_H_
