/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

namespace l0op {

OP_TYPE_REGISTER(StridedSlice);

// 910D int8,uint8,int16,uint16,int32,uint32,int64,uint64,float32,float16,bfloat16,complex32,complex64,bool
// hifloat8,float8_e5m2,float8_e4m3fn,
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910D = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,     op::DataType::DT_INT32,    op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,      op::DataType::DT_INT8,      op::DataType::DT_INT16,    op::DataType::DT_INT64,
    op::DataType::DT_UINT16,    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,   op::DataType::DT_BF16,
    op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN};

// 910B float16,float,int32,uint8,bool,int8,int64,int16,uint16,uint32,uint64,bfloat16,complex32,complex64,
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,    op::DataType::DT_INT32,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,      op::DataType::DT_INT8,     op::DataType::DT_INT16,  op::DataType::DT_INT64,
    op::DataType::DT_UINT16,    op::DataType::DT_UINT32,   op::DataType::DT_UINT64, op::DataType::DT_BF16,
    op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64};

// 910A float16,float,int32,uint8,bool,int8,int64,int16,uint16,uint32,uint64,complex32,complex64,
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT,  op::DataType::DT_INT32,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,     op::DataType::DT_INT8,   op::DataType::DT_INT16,  op::DataType::DT_INT64,
    op::DataType::DT_UINT16,   op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_COMPLEX32,
    op::DataType::DT_COMPLEX64};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto dataType = self->GetDataType();
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == op::SocVersion::ASCEND910B || socVersion == op::SocVersion::ASCEND910_93) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_910B);
    } else if (socVersion == op::SocVersion::ASCEND910_95) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_910D);
    }
    return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* StridedSliceAiCore(
    const aclTensor* x, const aclTensor* y, const aclTensor* begin, const aclTensor* end, const aclTensor* strides,
    const aclScalar* beginMask, const aclScalar* endMask, const aclScalar* ellipsisMask, const aclScalar* newAxisMask,
    const aclScalar* shrinkAxisMask, aclOpExecutor* executor)
{
    L0_DFX(StridedSliceAiCore, x, y, begin, end, strides);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        StridedSlice, OP_INPUT(x, begin, end, strides), OP_OUTPUT(y),
        OP_ATTR(beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "StridedSlice ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

const aclTensor* StridedSliceAiCpu(
    const aclTensor* x, const aclTensor* y, const aclTensor* begin, const aclTensor* end, const aclTensor* strides,
    const aclScalar* beginMask, const aclScalar* endMask, const aclScalar* ellipsisMask, const aclScalar* newAxisMask,
    const aclScalar* shrinkAxisMask, aclOpExecutor* executor)
{
    L0_DFX(StridedSliceAiCpu, x, y, begin, end, strides);
    static op::internal::AicpuTaskSpace space("StridedSlice", ge::DEPEND_CONST_VALUE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        StridedSlice,
        OP_ATTR_NAMES({"T", "Index", "begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"}),
        OP_INPUT(x, begin, end, strides), OP_OUTPUT(y),
        OP_ATTR(
            x->GetDataType(), begin->GetDataType(), beginMask->ToInt64(), endMask->ToInt64(), ellipsisMask->ToInt64(),
            newAxisMask->ToInt64(), shrinkAxisMask->ToInt64()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return y;
}

const aclTensor* StridedSlice(
    const aclTensor* x, const aclTensor* y, const aclTensor* begin, const aclTensor* end, const aclTensor* strides,
    const aclScalar* beginMask, const aclScalar* endMask, const aclScalar* ellipsisMask, const aclScalar* newAxisMask,
    const aclScalar* shrinkAxisMask, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(x)) {
        return StridedSliceAiCore(
            x, y, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, executor);
    }
    return StridedSliceAiCpu(
        x, y, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, executor);
}

const aclTensor* StridedSlice(
    const aclTensor* x, const aclTensor* y, const aclTensor* begin, const aclTensor* end, const aclTensor* strides,
    aclOpExecutor* executor)
{
    int64_t default_mask = 0;
    auto beginMask = executor->AllocScalar(default_mask);
    auto endMask = executor->AllocScalar(default_mask);
    auto ellipsisMask = executor->AllocScalar(default_mask);
    auto newAxisMask = executor->AllocScalar(default_mask);
    auto shrinkAxisMask = executor->AllocScalar(default_mask);
    return StridedSlice(
        x, y, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, executor);
}

const aclTensor* StridedSlice(
    aclTensor* x, const aclIntArray* begin, const aclIntArray* end, const aclIntArray* strides,
    const aclScalar* beginMask, const aclScalar* endMask, const aclScalar* ellipsisMask, const aclScalar* newAxisMask,
    const aclScalar* shrinkAxisMask, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    auto beginTensor = executor->ConvertToTensor(begin, op::ToOpDataType(ACL_INT64));
    auto endTensor = executor->ConvertToTensor(end, op::ToOpDataType(ACL_INT64));
    auto stridesTensor = executor->ConvertToTensor(strides, op::ToOpDataType(ACL_INT64));

    INFER_SHAPE(
        StridedSlice, OP_INPUT(x, beginTensor, endTensor, stridesTensor), OP_OUTPUT(out),
        OP_ATTR(beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask));
    return StridedSlice(
        x, out, beginTensor, endTensor, stridesTensor, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask,
        executor);
}

} // namespace l0op
