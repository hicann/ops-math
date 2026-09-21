/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unfold_grad.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UnfoldGrad);

namespace {
constexpr int64_t FP32_TYPESIZE = 4;
constexpr int64_t TRANS_BLOCK = 16;
constexpr int64_t ONCE_TRANSDATATO5HD_SIZE = 512;
constexpr int64_t DOUBLE = 2;
constexpr int64_t TOTAL_SIZE = 5;
constexpr int64_t WIDTH = 8;
constexpr int64_t ARCH22_UB_SIZE = 192 * 1024 - 256;
constexpr int64_t ARCH35_UB_SIZE = 248 * 1024;

int64_t Gcd(int64_t x, int64_t y)
{
    while (y != 0) {
        int64_t tmp = x % y;
        x = y;
        y = tmp;
    }
    return x;
}

int64_t getLowestCommonMultiple(int64_t x, int64_t y)
{
    int64_t greatestCommonDivisor = Gcd(x, y);
    return greatestCommonDivisor == 0 ? 0 : x / greatestCommonDivisor * y;
}

bool IsSecondLastAxisAiCoreSupport(op::DataType dataType, int64_t size, int64_t step)
{
    if ((dataType != DataType::DT_FLOAT && dataType != DataType::DT_FLOAT16 && dataType != DataType::DT_BF16) ||
        size <= 0 || step <= 0) {
        return false;
    }

    const bool isArch35 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    const int64_t ubSize = isArch35 ? ARCH35_UB_SIZE : ARCH22_UB_SIZE;
    int64_t ubSizeT2 = 0;
    if (dataType == DataType::DT_FLOAT16 || dataType == DataType::DT_BF16) {
        ubSizeT2 = ubSize / (ONCE_TRANSDATATO5HD_SIZE * TOTAL_SIZE) * (ONCE_TRANSDATATO5HD_SIZE * DOUBLE);
    } else {
        ubSizeT2 = ubSize / DOUBLE / (ONCE_TRANSDATATO5HD_SIZE * DOUBLE) * (ONCE_TRANSDATATO5HD_SIZE * DOUBLE);
    }

    const int64_t rowAvailableLength = (size + WIDTH - 1) / WIDTH * WIDTH;
    const int64_t lowestCommonMultiple = getLowestCommonMultiple(size, TRANS_BLOCK);
    const int64_t rowT2NeededLength = lowestCommonMultiple / size * rowAvailableLength;
    const int64_t colOnceMaxPerUB = ubSizeT2 / FP32_TYPESIZE / rowT2NeededLength / TRANS_BLOCK * TRANS_BLOCK;
    const int64_t tasksOnceMaxPerCore = colOnceMaxPerUB * lowestCommonMultiple / size;
    return tasksOnceMaxPerCore > 0;
}
} // namespace

static inline bool IsAiCoreSupport(const aclTensor* gradOut, int64_t dim, int64_t size, int64_t step)
{
    int64_t dimNum = gradOut->GetViewShape().GetDimNum() - 1;
    int64_t maxv = size >= step ? size : step;
    op::DataType gradOutDtype = gradOut->GetDataType();
    if ((dim == dimNum - 1) &&
        (((gradOutDtype == DataType::DT_FLOAT) && (maxv <= 49088)) ||
         ((gradOutDtype == DataType::DT_FLOAT16 || gradOutDtype == DataType::DT_BF16) && (maxv <= 32720)))) {
        return true;
    } else if ((dim == dimNum - 2) && IsSecondLastAxisAiCoreSupport(gradOutDtype, size, step)) {
        return true;
    } else {
        return false;
    }
}

// AICPU算子kernel
static const aclTensor* UnfoldGradAiCpu(const aclTensor* gradOut, const aclTensor* inputSizes, int64_t dim,
                                        int64_t size, int64_t step, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(UnfoldGradAiCpu, gradOut, inputSizes, dim, size, step, out);

    static internal::AicpuTaskSpace space("UnfoldGrad");
    ADD_TO_LAUNCHER_LIST_AICPU(UnfoldGrad, OP_ATTR_NAMES({"dim", "size", "step"}), OP_INPUT(gradOut, inputSizes),
                               OP_OUTPUT(out), OP_ATTR(dim, size, step));
    return out;
}

// AICORE算子kernel
static const aclTensor* UnfoldGradAiCore(const aclTensor* gradOut, const aclTensor* inputSizes, int64_t dim,
                                         int64_t size, int64_t step, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(UnfoldGradAiCore, gradOut, inputSizes, dim, size, step, out);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE
    ADD_TO_LAUNCHER_LIST_AICORE(UnfoldGrad, OP_INPUT(gradOut, inputSizes), OP_OUTPUT(out), OP_ATTR(dim, size, step));
    return out;
}

const aclTensor* UnfoldGrad(const aclTensor* gradOut, const aclTensor* inputSizes, int64_t dim, int64_t size,
                            int64_t step, aclOpExecutor* executor)
{
    L0_DFX(UnfoldGrad, gradOut, inputSizes, dim, size, step);
    auto out = executor->AllocTensor(gradOut->GetDataType(), gradOut->GetStorageFormat(), gradOut->GetOriginalFormat());
    auto ret = INFER_SHAPE(UnfoldGrad, OP_INPUT(gradOut, inputSizes), OP_OUTPUT(out), OP_ATTR(dim, size, step));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infershape failed.");
        return nullptr;
    }
    if (IsAiCoreSupport(gradOut, dim, size, step)) {
        return UnfoldGradAiCore(gradOut, inputSizes, dim, size, step, out, executor);
    } else {
        return UnfoldGradAiCpu(gradOut, inputSizes, dim, size, step, out, executor);
    }
}
} // namespace l0op
