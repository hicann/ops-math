/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MAXIMUM_H_
#define OP_API_INC_MAXIMUM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnmaximum的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算两个张量中每个元素的最大值，并返回一个新的张量。
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，
 * 且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，
 * 且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，
 * 且数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other
 * broadcast之后的shape。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaximumGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMaximum的第二段接口，用于执行计算。
 *
 * 算子功能：计算两个张量中每个元素的最大值，并返回一个新的张量。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMaximumGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaximum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MAXIMUM_H_
