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

/*!
 * \file aclnn_cat.cpp
 * \brief
 */
#include "aclnn_cat.h"
#include "concat.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "common/op_api_def.h"
#include "common/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,  DataType::DT_FLOAT16,   DataType::DT_INT16,
    DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16,
    DataType::DT_INT16,     DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE,
    DataType::DT_COMPLEX64, DataType::DT_BF16,  DataType::DT_BOOL};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(SocVersion socVersion)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (socVersion == SocVersion::ASCEND310P || socVersion == SocVersion::ASCEND910) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    } else if (
        socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        socVersion == SocVersion::ASCEND310B || socVersion == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* y)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    const auto& dTypeSupportList = GetSupportDtypeList(socVersion);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!CheckType((*tensors)[i]->GetDataType(), dTypeSupportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s, should be in dtype support list %s.", i,
                op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(dTypeSupportList).GetString());
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(y, dTypeSupportList, return false);
    return true;
}

static bool CheckNotNull(const aclTensorList* tensors, const aclTensor* y)
{
    OP_CHECK_NULL(tensors, return false);

    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_NULL((*tensors)[i], return false);
    }
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckPromoteType(const aclTensorList* tensors, const aclTensor* y)
{
    op::DataType promoteType = (*tensors)[0]->GetDataType();
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        if (promoteType == DataType::DT_UNDEFINED) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "tensor %lu dtype %s and dtype %s can not promote dtype.", i,
                op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(promoteType).GetString());
            return false;
        }
        if (promoteType == DataType::DT_COMPLEX128) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "tensor dtype has been promoted to %s, which has not been implemented yet.",
                op::ToString(promoteType).GetString());
            return false;
        }
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, y->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensorList* tensors, const aclTensor* y)
{
    op::Format format = (*tensors)[0]->GetStorageFormat();
    if (op::IsPrivateFormat(format)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        if ((*tensors)[i]->GetStorageFormat() != format) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Format of tensors should be equal, tensor %lu [%s], tensor 0 [%s].", i,
                op::ToString((*tensors)[i]->GetStorageFormat()).GetString(), op::ToString(format).GetString());
            return false;
        }
    }
    if (y->GetStorageFormat() != format) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, tensor 0 [%s] out [%s].",
            op::ToString(y->GetStorageFormat()).GetString(), op::ToString(y->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensorList* tensors, int64_t* realDim)
{
    OP_CHECK_MAX_DIM((*tensors)[0], MAX_SUPPORT_DIMS_NUMS, return false);
    op::Shape shape0 = (*tensors)[0]->GetViewShape();
    auto dimNum = (int64_t)shape0.GetDimNum();
    auto orgDim = *realDim;
    if (*realDim < 0) {
        (*realDim) += dimNum;
    }
    if ((*realDim) < 0 || (*realDim) >= dimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimnum %ld exceed the dim range of the tensor %ld.", orgDim, dimNum);
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        op::Shape shape = (*tensors)[i]->GetViewShape();
        if (dimNum != (int64_t)shape.GetDimNum()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "dimnum of tensor %lu is [%zu], should be equal to tensor 0 [%ld].", i,
                shape.GetDimNum(), dimNum);
            return false;
        }
        for (int64_t j = 0; j < dimNum; j++) {
            if (*realDim == j) {
                continue;
            }
            if (shape0.GetDim(j) != shape.GetDim(j)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "dim %ld of tensor %lu is [%ld], should be equal to tensor 0 [%ld].", j, i,
                    shape.GetDim(j), shape0.GetDim(j));
                return false;
            }
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, int64_t* realDim, const aclTensor* y)
{
    CHECK_RET(CheckNotNull(tensors, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(tensors, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckPromoteType(tensors, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(tensors, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(tensors, realDim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessOneTensor(const aclTensor* in, aclTensor* out, aclOpExecutor* executor)
{
    auto contiguous = l0op::Contiguous(in, executor);
    CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewIn = l0op::Cast(contiguous, out->GetDataType(), executor);
    CHECK_RET(viewIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(viewIn, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus SplitToConcat(const aclTensorList* tensors, int64_t dim, aclTensor* out, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> tensorListA;
    auto promoteType = (*tensors)[0]->GetDataType();
    if (!(*tensors)[0]->IsEmpty()) {
        tensorListA.emplace_back((*tensors)[0]);
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        if (!(*tensors)[i]->IsEmpty()) {
            tensorListA.emplace_back((*tensors)[i]);
        }
    }

    if (tensorListA.size() == 1) {
        return ProcessOneTensor(tensorListA[0], out, executor);
    }

    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    size_t catMaxInputs = (socVersion == op::SocVersion::ASCEND910_95) ? 512 : 32;
    bool firstLoop = true;
    while (tensorListA.size() > 1) {
        op::FVector<const aclTensor*> tensorListOnce;
        op::FVector<const aclTensor*> tensorListB;
        for (auto tensor : tensorListA) {
            if (firstLoop) {
                auto contiguous = l0op::Contiguous(tensor, executor);
                CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto castOut = l0op::Cast(contiguous, promoteType, executor);
                CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListOnce.emplace_back(castOut);
            } else {
                tensorListOnce.emplace_back(tensor);
            }
            if (tensorListOnce.size() == catMaxInputs) {
                auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensor = l0op::ConcatD(tensorList, dim, executor);
                CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListB.emplace_back(concatTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            if (tensorListOnce.size() == 1) {
                tensorListB.emplace_back(tensorListOnce.front());
            } else {
                auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensorTail = l0op::ConcatD(aclTensorListTail, dim, executor);
                CHECK_RET(concatTensorTail != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListB.emplace_back(concatTensorTail);
            }
            tensorListOnce.clear();
        }
        tensorListA = tensorListB;
        firstLoop = false;
    }

    if (tensorListA.empty()) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckShapeAndScalarSame(tensorListA.front(), out), ACLNN_ERR_PARAM_INVALID);
    auto castOut = l0op::Cast(tensorListA.front(), out->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclTensorList* EraseDim1EmptyTensor(const aclTensorList* tensors, aclOpExecutor* executor)
{
    if (tensors == nullptr) {
        return nullptr;
    }
    op::FVector<const aclTensor*> fTensorList;
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        op::Shape shape = (*tensors)[i]->GetViewShape();
        if ((shape.GetDimNum() == 1) && (shape.GetDim(0) == 0)) {
            continue;
        }
        fTensorList.push_back((*tensors)[i]);
    }
    return executor->AllocTensorList(fTensorList.data(), fTensorList.size());
}

aclnnStatus aclnnCatGetWorkspaceSize(
    const aclTensorList* tensors, int64_t dim, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCat, DFX_IN(tensors, dim), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclTensorList* tensorList = EraseDim1EmptyTensor(tensors, uniqueExecutor.get());
    if (tensorList != nullptr && tensorList->Size() == 0) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    int64_t realDim = dim;
    auto ret = CheckParams(tensorList, &realDim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = SplitToConcat(tensorList, realDim, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCat);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
