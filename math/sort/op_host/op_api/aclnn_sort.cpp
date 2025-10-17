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

#include "aclnn_sort.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "sort.h"
#include "aclnn_kernels/transpose.h"
#include "math/zero_op/op_host/op_api/zero_op.h"

#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// Sort的self支持的Dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16,  op::DataType::DT_UINT8,
    op::DataType::DT_INT8,    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64};
// Sort的value支持的dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_VALUE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT8,   op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BF16};
// Sort的indices支持的dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INT = {op::DataType::DT_INT64};
// 910D Sort的self/value支持的dtype
static const std::initializer_list<op::DataType> ASCEND910D_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16,  op::DataType::DT_UINT8,
    op::DataType::DT_INT8,    op::DataType::DT_INT16,  op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT16,  op::DataType::DT_UINT32, op::DataType::DT_UINT64};
static const int64_t DATA_LIMIT = 100000;
static const int64_t DIM_MAX = 8;

// parm判断
static inline bool CheckNotNull(const aclTensor* self, const aclTensor* values, const aclTensor* indices)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(values, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

// 检查dType符合预期
static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* values, const aclTensor* indices)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910D_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(values, ASCEND910D_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, DTYPE_SUPPORT_LIST_INT, return false);
        return true;
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(values, DTYPE_SUPPORT_LIST_VALUE, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, DTYPE_SUPPORT_LIST_INT, return false);
    }
    return true;
}

// 获得tensor的维度数
static inline int64_t GetTensorDim(const aclTensor* self)
{
    return static_cast<int64_t>(self->GetViewShape().GetDimNum());
}

// dim应该处于范围 [-N, N-1]中
static inline bool CheckDimValue(const aclTensor* self, const int64_t dim)
{
    int64_t dimSize = GetTensorDim(self);
    int64_t dimMin = std::min(-1 * dimSize, dimSize - 1);
    int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
    if ((dim > dimMax) || (dim < dimMin)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be in range [%ld, %ld].", dimMin, dimMax);
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* values, const aclTensor* indices)
{
    OP_CHECK_SHAPE_NOT_EQUAL(self, values, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, indices, return false);
    return true;
}

// 检查参数情况
static aclnnStatus CheckParams(const aclTensor* self, int64_t dim, aclTensor* values, aclTensor* indices)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, values, indices), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数的数据类型是否符合预期
    CHECK_RET(CheckDtypeValid(self, values, indices), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查dim是否为 [-N, N-1] 范围内
    CHECK_RET(CheckDimValue(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape一致
    CHECK_RET(CheckShape(self, values, indices), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 将dim从负数转换为正数
static inline int64_t wrapDim(int64_t dim, int64_t dimSize)
{
    return (dim < 0) ? dim + dimSize : dim;
}

// 将dim与最后一个dim进行对换
static aclIntArray* GetPermResult(int64_t dim, int64_t dimSize, aclOpExecutor* executor)
{
    std::vector<int64_t> valuePerm(dimSize);
    for (int64_t i = 0; i < dimSize; i++) {
        valuePerm[i] = i;
    }

    std::swap(valuePerm[dim], valuePerm[dimSize - 1]);
    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

// 获得shape的 aclIntArray
static aclIntArray* GetTensorShape(const aclTensor* x, aclOpExecutor* executor)
{
    auto shape = x->GetViewShape();
    auto dimSize = GetTensorDim(x);

    std::vector<int64_t> valuePerm(dimSize);
    for (int64_t i = 0; i < dimSize; i++) {
        valuePerm[i] = shape[i];
    }

    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

// 检查tuple <values, indices>里的元素是否为非null。true表示为非null，false表示为null
static bool CheckTupleNullptr(std::tuple<const aclTensor*, const aclTensor*> tensorTuple)
{
    // tensorTuple must be size 2
    if (std::tuple_size<decltype(tensorTuple)>::value != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of tuple returned by Sort is not 2.");
        return false;
    }
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

// 使x的shape小于等于8维，并且dim维度不受影响
static aclIntArray* reshapeShape(const aclTensor* x, int64_t dim, aclOpExecutor* executor)
{
    auto shape = x->GetViewShape();
    auto dimSize = GetTensorDim(x);

    int64_t leftPart = 1;
    for (int64_t i = 0; i < dim; i++) {
        leftPart *= shape[i];
    }

    // dim正好为最后一个维度时： 只有left 和 sort dim
    if (dim == dimSize - 1) {
        int64_t valuePerm[2] = {leftPart, shape[dim]};
        auto perm = executor->AllocIntArray(valuePerm, 2);
        if (perm == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "reshapeShape (left + sort) perm return a nullptr.");
        }
        return perm;
    }

    // dim 不为最后一个维度
    int64_t rightPart = 1;
    for (int64_t i = dim + 1; i < dimSize; i++) {
        rightPart *= shape[i];
    }

    int64_t valuePerm[3] = {leftPart, shape[dim], rightPart};
    auto perm = executor->AllocIntArray(valuePerm, 3);
    if (perm == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "reshapeShape perm return a nullptr.");
    }

    return perm;
}

const aclTensor* reshapeIfLargeTensor(
    const aclTensor* x, aclOpExecutor* executor, int64_t originalDimSize, aclIntArray* valuePerm = nullptr)
{
    auto dimSize = GetTensorDim(x);
    // 如果没有reshape过
    if (originalDimSize == dimSize && dimSize <= DIM_MAX) {
        return x;
    }

    auto reshapeSelf = l0op::Reshape(x, valuePerm, executor);
    return reshapeSelf;
}

// 处理0维场景：self cast以后viewcopy, indices是为0
static aclnnStatus HandleDimZeroTensor(
    const aclTensor* self, aclTensor* valuesOut, aclTensor* indicesOut, aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // indices是返回0
    auto zeroIndices = l0op::ZerosLike(selfContiguous, executor);
    CHECK_RET(zeroIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valuesCast = l0op::Cast(selfContiguous, valuesOut->GetDataType(), executor);
    CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesCast = l0op::Cast(zeroIndices, DataType::DT_INT64, executor);
    CHECK_RET(indicesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyValues = l0op::ViewCopy(valuesCast, valuesOut, executor);
    CHECK_RET(viewCopyValues != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyIndices = l0op::ViewCopy(indicesCast, indicesOut, executor);
    CHECK_RET(viewCopyIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

// 如果需要perm， 计算perm  如果不需要，返回nullptr
static aclIntArray* updatePerm(int64_t dim, int64_t dimSize, aclOpExecutor* executor)
{
    // 需要transpose，计算perm。
    if (dim != dimSize - 1) {
        auto perm = GetPermResult(dim, dimSize, executor); // 不需改动时的版本
        if (dimSize > DIM_MAX) {
            perm = GetPermResult(1, 3, executor); // 1为sort的维度，3为reshape以后总共有3维
        }
        return perm;
    }
    return nullptr;
}

// 根据perm判断是否进行transpose, 再进行sort的计算
static std::tuple<const aclTensor*, const aclTensor*> SortProcess(
    const aclTensor* reshapeSelf, aclIntArray* perm, bool stable, bool descending, op::DataType indicesType,
    aclOpExecutor* executor)
{
    bool needTranspose = (perm != nullptr);
    auto nullPtr = nullptr;

    if (needTranspose) {
        reshapeSelf = l0op::Transpose(reshapeSelf, perm, executor);
        CHECK_RET(reshapeSelf != nullptr, std::tie(nullPtr, nullPtr));
    }

    bool needCastFp16 = (reshapeSelf->GetDataType() == op::DataType::DT_FLOAT) &&
                        ((GetCurrentPlatformInfo().GetSocVersion()) == SocVersion::ASCEND910);
    if (needCastFp16) {
        reshapeSelf = l0op::Cast(reshapeSelf, op::DataType::DT_FLOAT16, executor);
        CHECK_RET(reshapeSelf != nullptr, std::tie(nullPtr, nullPtr));
    }
    auto sortRes = l0op::Sort(reshapeSelf, -1, descending, stable, indicesType, executor);
    CHECK_RET(CheckTupleNullptr(sortRes), std::tie(nullPtr, nullPtr));
    auto sortValues = std::get<0>(sortRes);
    auto sortIndices = std::get<1>(sortRes);

    if (needTranspose) {
        auto values = l0op::Transpose(sortValues, perm, executor);
        auto indices = l0op::Transpose(sortIndices, perm, executor);
        CHECK_RET(values != nullptr, std::tie(nullPtr, nullPtr));
        CHECK_RET(indices != nullptr, std::tie(nullPtr, nullPtr));
        return std::tie(values, indices);
    }

    return std::tie(sortValues, sortIndices);
}

static std::tuple<const aclTensor*, const aclTensor*> reshapeCastRes(
    std::tuple<const aclTensor*, const aclTensor*> sortRes,
    std::tuple<const aclTensor*, const aclTensor*> expectedCastRes, int64_t dimSize, aclIntArray* selfShapeDetail,
    aclOpExecutor* executor)
{
    auto nullPtr = nullptr;
    auto sortValues = std::get<0>(sortRes);
    auto sortIndices = std::get<1>(sortRes);
    auto opValues = reshapeIfLargeTensor(sortValues, executor, dimSize, selfShapeDetail);
    auto opIndices = reshapeIfLargeTensor(sortIndices, executor, dimSize, selfShapeDetail);

    // 将values更改成out的dtype, 将indices更改成 INT64 的Dtype
    auto expectedValue = std::get<0>(expectedCastRes);
    auto valuesCast = l0op::Cast(opValues, expectedValue->GetDataType(), executor);
    CHECK_RET(valuesCast != nullptr, std::tie(nullPtr, nullPtr));
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return std::tie(valuesCast, opIndices);
    }
    auto indicesCast = l0op::Cast(opIndices, op::DataType::DT_INT64, executor);
    CHECK_RET(indicesCast != nullptr, std::tie(nullPtr, nullPtr));
    return std::tie(valuesCast, indicesCast);
}

aclnnStatus aclnnSortGetWorkspaceSize(
    const aclTensor* self, bool stable, int64_t dim, bool descending, aclTensor* valuesOut, aclTensor* indicesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSort, DFX_IN(self, stable, dim, descending), DFX_OUT(valuesOut, indicesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Ascend310 不支持 stable = True
    if (stable && GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Ascend310 does not support sort(stable=True)");
        CHECK_RET(false, ACLNN_ERR_PARAM_INVALID);
    }

    auto ret = CheckParams(self, dim, valuesOut, indicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // sort支持空tensor
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // sort支持0维tensor(a), dim此时必须得为0 / -1, 已在前面校验过
    if (self->GetViewShape().GetDimNum() == 0) {
        auto res = HandleDimZeroTensor(self, valuesOut, indicesOut, uniqueExecutor.get());
        CHECK_RET(res == ACLNN_SUCCESS, res);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    int64_t dimSize = GetTensorDim(self);
    int64_t dimPositive = wrapDim(dim, dimSize);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // kernel暂不支持bf16输入，转为fp32进行计算
    if (self->GetDataType() == op::DataType::DT_BF16 &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    auto selfShapeDetail = GetTensorShape(selfContiguous, uniqueExecutor.get()); // self最原始的shape

    // 如果大于8维，需要reshape
    if (dimSize > DIM_MAX) {
        auto shapeNew = reshapeShape(selfContiguous, dimPositive, uniqueExecutor.get());
        selfContiguous = reshapeIfLargeTensor(selfContiguous, uniqueExecutor.get(), dimSize, shapeNew);
    }
    auto indicesType = indicesOut->GetDataType();
    auto perm = updatePerm(dimPositive, dimSize, uniqueExecutor.get());
    auto sortRes = SortProcess(selfContiguous, perm, stable, descending, indicesType, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(sortRes), ACLNN_ERR_PARAM_NULLPTR);

    auto expectedCastRes = std::tie(valuesOut, indicesOut);
    auto castRes = reshapeCastRes(sortRes, expectedCastRes, dimSize, selfShapeDetail, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(castRes), ACLNN_ERR_PARAM_NULLPTR);
    auto valuesCast = std::get<0>(castRes);
    auto indicesCast = std::get<1>(castRes);

    auto viewCopyValues = l0op::ViewCopy(valuesCast, valuesOut, uniqueExecutor.get());
    auto viewCopyIndices = l0op::ViewCopy(indicesCast, indicesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyValues != nullptr && viewCopyIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSort(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSort);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif