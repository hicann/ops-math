/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ASIN_H_
#define OP_API_INC_ASIN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAsin的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 功能描述：从输入矩阵的每一个元素进行反余弦操作后输出。
 * 计算公式：
 * out_{i}=sin^{-1}(input_{i})
 * 参数描述：
 * @param [in]   input
 * 输入Tensor，数据类型支持INT8，INT16，INT32，INT64，UINT8，BOOL，FLOAT，BFLOAT16,
 * FLOAT16，DOUBLE。支持非连续Tensor，数据格式支持ND。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，BFLOAT16, FLOAT16，DOUBLE。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnAsinGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnAsin的第二段接口，用于执行计算。
 * 功能描述：从输入矩阵的每一个元素进行反余弦操作后输出。。
 * 计算公式：
 * out_{i}=sin^{-1}(input_{i})
 * 实现说明：
 * api计算的基本路径：
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B --> C([l0op::Asin])
    C --> G([l0op::Cast])
    G --> E([l0op::ViewCopy])
    E --> S[(Out)]
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAsinGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAsin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceAsin的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 功能描述：从输入矩阵的每一个元素进行反余弦操作后输出。
 * 计算公式：
 * out_{i}=sin^{-1}(input_{i})
 * 参数描述：
 * @param [in]   input
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，DOUBLE。支持非连续Tensor，数据格式支持ND。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16，DOUBLE。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceAsinGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspace_size,
                                                       aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceAsin的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor原地完成asin操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAsinGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAsin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
