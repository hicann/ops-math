/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LINSPACE_H_
#define OP_API_INC_LINSPACE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLinspace的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 功能描述：创建一个大小为steps的1维向量，其值从start起始到end结束（包含）线性均匀分布。
 *
 * 计算公式：$$ out = (start, start + \frac{end - start}{steps - 1},...,
 *                    tart + (steps -2) * \frac{end - start}{steps - 1}, end) $$
 *
 * 参数描述：
 * @param [in]   start
 * 获取值的范围的起始位置：host侧的aclScalar，数据类型支持整型，浮点数据类型。数据格式支持ND。
 * @param [in]   end
 * 获取值的范围的结束位置：host侧的aclScalar，数据类型支持整型，浮点数据类型。数据格式支持ND。
 * @param [in]   steps
 * 张量的大小：host侧的aclScalar，数据类型支持整型。数据格式支持ND。需要满足steps大于等于0。
 * @param [in]   out              指定的输出tensor：npu
 * device侧的aclTensor，数据类型支持整型，浮点数据类型，数据格式支持ND。
 * @param [out]  workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar* start, const aclScalar* end, int64_t steps,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnLinspace的第二段接口，用于执行计算。
 *
 *
 * 功能描述：创建一个大小为steps的1维向量，其值从start起始到end结束（包含）线性均匀分布。
 *
 * 计算公式：$$ out = (start, start + \frac{end - start}{steps - 1},...,
 *                    start + (steps -2) * \frac{end - start}{steps - 1}, end) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLinspaceGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLinspace(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif