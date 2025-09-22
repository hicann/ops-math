/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort.h"
#include "math/zero_op/op_host/op_api/zero_op.h"
#include "conversion/tensor_move/op_host/op_api/tensor_move.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/platform.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sort);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DATA_LIMIT = 100000;
static const int64_t AXIS_LIMIT = 8;

// 根据排序轴的数据量大小判断是否支持aicore
static bool SocSupportDimSize(const aclTensor *self)
{
    // 该维度数据量为1 或数据量>100000 走AICPU
    auto shapeSize = (int64_t)(self->GetViewShape().GetDimNum());
    auto lastDimSize = (self->GetViewShape())[shapeSize-1];
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND310 || socVersion == SocVersion::ASCEND310B) {
        // sort轴数据量大于100k
        if (lastDimSize > DATA_LIMIT) {
            return false;
        }
    } else {
        // sort轴数据量为1
        if (lastDimSize == 1) {
            return false;
        }
    }
    return true;
}

// 根据dtype判断是否支持aicore
static bool SocSupportDtype(const aclTensor *self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    // AiCore只支持FLOAT16 + FLOAT32
    if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        // 910和310芯片 + tensor为FLOAT32 则不支持AiCore
        if (((socVersion == SocVersion::ASCEND910) || (socVersion == SocVersion::ASCEND310)) &&
            (self->GetDataType()==op::DataType::DT_FLOAT || self->GetDataType()==op::DataType::DT_BF16)) {
            return false;
        }
        return true;
    }
    return false;
}

static bool IsAiCoreSupport(const aclTensor *self, bool stable, bool descending)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95){
        return true;
    } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310B && stable && !descending) {
        return false;
    } else {
        return (SocSupportDimSize(self) && SocSupportDtype(self));
    }
}

void SortAiCore(const aclTensor *self, bool stable, int64_t dim, bool descending, aclTensor *values, aclTensor *indices,
    aclOpExecutor* executor)
{
    L0_DFX(SortAiCore, self, stable, dim, descending, values, indices);

    auto dimSize = (int64_t)(self->GetViewShape().GetDimNum());
    if ((dimSize!= dim + 1) && (dim != -1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must equal to the (number of dimensions - 1 ) or -1.");
    }

    ADD_TO_LAUNCHER_LIST_AICORE(Sort, OP_INPUT(self), OP_OUTPUT(values, indices),
        OP_ATTR(dim, descending, stable));
}

void SortAiCoreForDavid(const aclTensor *self, bool stable, int64_t dim, bool descending, aclTensor *values,
    aclTensor *indices, op::DataType indicesType, aclOpExecutor* executor)
{
    L0_DFX(SortAiCoreForDavid, self, stable, dim, descending, values, indices, indicesType);
    
    auto dimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if ((dimSize!= dim + 1) && (dim != -1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must equal to the (number of dimensions - 1 ) or -1.");
    }

    ADD_TO_LAUNCHER_LIST_AICORE(Sort, OP_INPUT(self), OP_OUTPUT(values, indices),
        OP_ATTR(dim, descending, stable, indicesType));
}

std::tuple<aclTensor*, aclTensor*> SortAiCpu(const aclTensor *self, bool stable, int64_t dim, bool descending,
                                             aclTensor *values, aclTensor *indices, aclOpExecutor* executor)
{
    L0_DFX(SortAiCpu, self, stable, dim, descending, values, indices);

    auto dimSize = (int64_t)(self->GetViewShape().GetDimNum());
    if ((dim > (dimSize-1)) || (dim + dimSize < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be in range [-N, N-1]. Current dim is %ld.", dim);
    }

    static internal::AicpuTaskSpace space("Sort");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Sort, OP_ATTR_NAMES({"axis", "descending", "stable"}), OP_INPUT(self),
        OP_OUTPUT(values, indices), OP_ATTR(dim, descending, stable));
    if (ret != ACLNN_SUCCESS) {
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tie(values, indices);
}

// 处理0维场景：self cast以后viewcopy, indices是为0
static aclnnStatus HandleDimZeroTensor(const aclTensor *self, aclTensor *valuesOut, aclTensor *indicesOut,
    aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // indices是返回0
    auto zeroIndices = l0op::ZerosLike(selfContiguous, executor);
    CHECK_RET(zeroIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valuesCast = l0op::Cast(selfContiguous, valuesOut->GetDataType(), executor);
    CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesCast = l0op::Cast(zeroIndices, indicesOut->GetDataType(), executor);
    CHECK_RET(indicesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyValues = l0op::TensorMove(valuesCast, valuesOut, executor);
    CHECK_RET(viewCopyValues != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyIndices = l0op::TensorMove(indicesCast, indicesOut, executor);
    CHECK_RET(viewCopyIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

const std::tuple<aclTensor*, aclTensor*> Sort(const aclTensor *self, int64_t dim, bool descending, bool stable,
                                              op::DataType indicesType, aclOpExecutor* executor)
{
    L0_DFX(Sort, self, dim, descending, stable, indicesType);
    auto selfShape = self->GetViewShape();
    auto selfFormat = self->GetViewFormat();

    auto dimSize = (int64_t)(selfShape.GetDimNum());
    if (dimSize < 0 || dimSize > AXIS_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor self dimension size must be in range [0, 8]. Current size is [%ld].",
            dimSize);
    }
    // handle dimsize=0
    if (dimSize == 0) { 
        if (dim == 0 || dim == -1) {
            auto valuesOut = executor->AllocTensor(selfShape, self->GetDataType(), selfFormat);
            aclTensor* indicesOut = nullptr;
            if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
                indicesOut = executor->AllocTensor(selfShape, indicesType, selfFormat);
            } else {
                indicesOut = executor->AllocTensor(selfShape, op::DataType::DT_INT32, selfFormat);
            }
            auto res = HandleDimZeroTensor(self, valuesOut, indicesOut, executor);
            if (res != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_INNER, "HandleDimZeroTensor error.");
                return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
            } else {
                return std::tie(valuesOut, indicesOut); 
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When dimSize == 0 , dim must be 0 or -1.");
            return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
        }
    }
    auto lastDimSize = selfShape[dimSize - 1];
    // The Sort Op not support sort axis is 1 when input type is BF16..
    bool  isNotSupport = (1 == lastDimSize && op::DataType::DT_BF16 == self->GetDataType());
    if (isNotSupport && GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The sort axis value is not support 1 when input type is BF16.");
      return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    auto values = executor->AllocTensor(selfShape, self->GetDataType(), selfFormat);
    aclTensor* indices = nullptr;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        indices = executor->AllocTensor(selfShape, indicesType, selfFormat);
    } else {
        indices = executor->AllocTensor(selfShape, op::DataType::DT_INT32, selfFormat);
    }
    if (IsAiCoreSupport(self, stable, descending)) {
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
            SortAiCoreForDavid(self, stable, dim, descending, values, indices, indicesType, executor);
        } else {
            SortAiCore(self, stable, dim, descending, values, indices, executor);
        }
    } else {
        return SortAiCpu(self, stable, dim, descending, values, indices, executor);
    }
    return std::tie(values, indices);
}
}
