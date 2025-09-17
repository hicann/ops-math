/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LEVEL2_BASE_H_
#define LEVEL2_BASE_H_

#include "common/op_api_def.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace op {

// 检查1个输入和1个输出是否是空指针
[[maybe_unused]] static bool CheckNotNull2Tensor(const aclTensor* t0, const aclTensor* t1)
{
    OP_CHECK_NULL(t0, return false);
    OP_CHECK_NULL(t1, return false);

    return true;
}

/**
 * 1. 1个输入1个输出
 * 2. 输入输出的shape必须一致
 * 3. 输入的维度必须小于等于8
 */
[[maybe_unused]] static bool CheckSameShape1In1Out(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    // self的维度必须小于等于8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

// 检查1个输入和1个输出的数据类型是否在算子的支持列表内
[[maybe_unused]] static bool CheckDtypeValid1In1Out(
    const aclTensor* self, const aclTensor* out, const std::initializer_list<op::DataType>& dtypeSupportList,
    const std::initializer_list<op::DataType>& dtypeOutList)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    // 检查输出的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeOutList, return false);

    return true;
}

/**
 * l1: ASCEND910B 或者 ASCEND910_93芯片，该算子支持的数据类型列表
 * l2: 其他芯片，该算子支持的数据类型列表
 */
[[maybe_unused]] static const std::initializer_list<DataType>& GetDtypeSupportListV1(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return l1;
    } else {
        return l2;
    }
}

/**
 * l1: ASCEND910B ~ ASCEND910E芯片，该算子支持的数据类型列表
 * l2: 其他芯片，该算子支持的数据类型列表
 */
[[maybe_unused]] static const std::initializer_list<DataType>& GetDtypeSupportListV2(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return l1;
    } else {
        return l2;
    }
}

} // namespace op
#ifdef __cplusplus
}
#endif
#endif // LEVEL2_BASE_H_