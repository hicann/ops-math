/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_population_count.cpp
 * @brief ACLNN L2 API implementation for PopulationCount
 */

#include "aclnn_population_count.h"
#include "population_count.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT16, DataType::DT_UINT16
};

static bool IsXDtypeSupported(DataType dtype)
{
    return CheckType(dtype, X_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK(IsXDtypeSupported(x->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "PopulationCount: x dtype not supported: %d (must be INT16 or UINT16).",
                     static_cast<int>(x->GetDataType())),
             return false);
    OP_CHECK(y->GetDataType() == DataType::DT_UINT8,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "PopulationCount: y dtype must be UINT8, got %d.",
                     static_cast<int>(y->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* y)
{
    auto xFmt = x->GetStorageFormat();
    auto yFmt = y->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(xFmt) || IsPrivateFormat(yFmt)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "PopulationCount: private format not supported. x=%d, y=%d",
                     static_cast<int>(xFmt), static_cast<int>(yFmt)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);

    auto xShape = x->GetViewShape();
    auto yShape = y->GetViewShape();
    OP_CHECK(xShape.GetDimNum() == yShape.GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "PopulationCount: x/y dim num mismatch: %zu vs %zu",
                     xShape.GetDimNum(), yShape.GetDimNum()),
             return false);
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        OP_CHECK(xShape.GetDim(i) == yShape.GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "PopulationCount: x/y shape mismatch at dim %zu: %ld vs %ld",
                         i, xShape.GetDim(i), yShape.GetDim(i)),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* y)
{
    CHECK_COND(CheckNotNull(x, y), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: x_dtype=%d, y_dtype=%d",
               static_cast<int>(x->GetDataType()), static_cast<int>(y->GetDataType()));
    CHECK_COND(CheckFormat(x, y), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed");
    CHECK_COND(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed");
    return ACLNN_SUCCESS;
}

/**
 * @brief Two-phase L2 API (Phase 1): compute workspace + build executor.
 */
extern "C" aclnnStatus aclnnPopulationCountGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
{
    L2_DFX_PHASE_1(aclnnPopulationCount, DFX_IN(x), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Empty tensor fast-path
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::PopulationCount(xContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Two-phase L2 API (Phase 2): launch kernel.
 */
extern "C" aclnnStatus aclnnPopulationCount(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
{
    L2_DFX_PHASE_2(aclnnPopulationCount);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
