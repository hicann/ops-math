/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_REPLICATION_PAD1D_H_
#define OP_API_INC_REPLICATION_PAD1D_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnReplicationPad1d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：使用输入边界填充输入tensor。
 * @param [in] self: 数据类型支持FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8,
 * COMPLEX64, COMPLEX128，支持非连续的Tensor，数据格式支持ND，维度支持二维或三维，在最后一维做pad。
 * @param [in] padding: 数据类型为INT64，长度为2，数值依次代表左右两边需要填充的值。
 * @param [in] out: 数据类型、数据格式、维度与self一致，最后一维度的数值等于self最后一维度的数值加padding的两个值。
 * 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnReplicationPad1dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding,
                                                            aclTensor* out, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

/**
 * @brief: aclnnReplicationPad1d的第二段接口，用于执行计算
 *
 * 算子功能： 使用输入边界填充输入tensor。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnReplicationPad1dGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnReplicationPad1d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REPLICATION_PAD1D_H_