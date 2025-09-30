/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "aclnn_pad2d_backward.h"
#include "../../../pad_v3_grad/op_host/op_api/padv3grad.h"
#include "aclnn_kernels/contiguous.h"
#include "../../../squeeze/op_host/op_api/squeeze.h"
#include "../../../unsqueeze/op_host/op_api/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const string REFLECTION_MODE = "reflect";
static const string REPLICATION_MODE = "edge";
static const int64_t INPUT_NUM_2 = 2;
static const int64_t INPUT_NUM_3 = 3;
static const int64_t MAX_PADDING_VALUE = 6;
static const int64_t REFLECT_MODE_MAX_PADDING_VALUE = 7;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16,   op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

inline static bool CheckNotNull(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查gradInput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, dtypeSupportList, return false);

    // gradOutput, self和gradInput数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradInput, self->GetDataType(), return false);
    return true;
}

inline static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    OP_CHECK(
        gradOutput->GetViewFormat() == self->GetViewFormat() &&
            gradOutput->GetViewFormat() == gradInput->GetViewFormat(),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of input and output should be equal, gradOutput [%s], self [%s], gradInput [%s].",
            op::ToString(gradOutput->GetViewFormat()).GetString(), op::ToString(self->GetViewFormat()).GetString(),
            op::ToString(gradInput->GetViewFormat()).GetString()),
        return false);
    return true;
}

static bool CheckPaddingValue(const aclIntArray* padding, [[maybe_unused]] const string& mode)
{
    // padding的每一维度的数值要大于等于0
    if ((*padding)[0] < 0 || (*padding)[1] < 0 || (*padding)[INPUT_NUM_2] < 0 || (*padding)[INPUT_NUM_3] < 0) {
        OP_LOGW("on aicore situation, padding values should be greater than 0 or equal 0.");
        return false;
    }
    if ((*padding)[INPUT_NUM_2] == 0 && (*padding)[INPUT_NUM_3] == 0) {
        OP_LOGW("h dim no pad.");
        return true;
    }
    if ((*padding)[0] > REFLECT_MODE_MAX_PADDING_VALUE || (*padding)[1] > REFLECT_MODE_MAX_PADDING_VALUE ||
        (*padding)[INPUT_NUM_2] > REFLECT_MODE_MAX_PADDING_VALUE ||
        (*padding)[INPUT_NUM_3] > REFLECT_MODE_MAX_PADDING_VALUE) {
        OP_LOGW("on aicore situation, padding values should be less than 7 or equal 7.");
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, const string& mode,
    const aclTensor* gradInput)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self和gradInput的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradInput, return false);

    // 3 and 4 are dims, self只支持3维和4维
    OP_CHECK_MIN_DIM(self, 3, return false);
    OP_CHECK_MAX_DIM(self, 4, return false);

    // gradOutput, self, gradInput维度需要一致
    OP_CHECK(
        gradOutput->GetViewShape().GetDimNum() == selfDimnum && gradInput->GetViewShape().GetDimNum() == selfDimnum,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput, self, gradInput dim should be same."), return false);

    // padding长度为4
    OP_CHECK(
        padding->Size() == 4,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 4, but got %lu.", padding->Size()), return false);

    // reflection时padding的每一维度的数值要小于对应self的dim
    if (mode == "reflect") {
        // 0, 1, 2, 3 are indexes
        OP_CHECK(
            (*padding)[0] < self->GetViewShape().GetDim(selfDimnum - 1) &&
                (*padding)[1] < self->GetViewShape().GetDim(selfDimnum - 1) &&
                (*padding)[2] < self->GetViewShape().GetDim(selfDimnum - 2) &&
                (*padding)[3] < self->GetViewShape().GetDim(selfDimnum - 2),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding size should be less than the corresponding self dimention."),
            return false);
    }

    // bf16仅支持h dim no pad场景或者padding值不大于7的场景
    if (gradOutput->GetDataType() == op::DataType::DT_BF16) {
        auto isBF16Support = CheckPaddingValue(padding, mode);
        OP_CHECK(
            isBF16Support,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "bf16 only supports h dim no pad or padding values less than 7 or equal 7."),
            return false);
    }

    // check the last 2 dim value of out. 0, 1, 2, 3 are indexes
    OP_CHECK(
        gradOutput->GetViewShape().GetDim(selfDimnum - 2) ==
                self->GetViewShape().GetDim(selfDimnum - 2) + (*padding)[2] + (*padding)[3] &&
            gradOutput->GetViewShape().GetDim(selfDimnum - 1) ==
                self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1],
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "wrong gradOutput shape."), return false);
    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, const string& mode,
    const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, padding, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(gradOutput, self, padding, mode, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetPaddingTensor(int64_t dim, const aclIntArray* padding, aclOpExecutor* executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> paddingsVector;
    // 2 is the magnification
    for (size_t i = 2 * dim; i > 0; i -= 2) {
        if (i <= (size_t)padding->Size()) {
            // 2 and 1 indicate the element of padding is put into paddingsVector from the back to the front
            paddingsVector.emplace_back((*padding)[i - 2]);
            paddingsVector.emplace_back((*padding)[i - 1]);
        } else {
            paddingsVector.emplace_back(0);
            paddingsVector.emplace_back(0);
        }
    }
    // 2 is the magnification
    auto newpadding = executor->AllocIntArray(paddingsVector.data(), 2 * dim);
    auto paddingsTensor = executor->ConvertToTensor(newpadding, static_cast<op::DataType>(ACL_INT64));
    return paddingsTensor;
}

static aclnnStatus InputPreprocess(
    const aclTensor*& gradOutput, const aclTensor*& self, int64_t dimCp, aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    gradOutput = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 3 is dim
    if (dimCp == 3) {
        // 0 is index
        const int64_t appendDim[] = {0};
        // 1 is the dim num to be unsqueezed
        aclIntArray* dimArray = executor->AllocIntArray(appendDim, 1);
        self = l0op::UnsqueezeNd(self, dimArray, executor);
        gradOutput = l0op::UnsqueezeNd(gradOutput, dimArray, executor);
        CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus commonPad2dBackward(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, const string& mode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, padding, mode, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (gradOutput->IsEmpty() || self->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0;
        // 3 is dim number
        if (self->GetViewShape().GetDimNum() == 3) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0 || mode == "edge") {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 4 is dim number
        if (self->GetViewShape().GetDimNum() == 4) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0 ||
                // 3 is index
                self->GetViewShape().GetDim(3) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子进行计算
    auto dim = self->GetViewShape().GetDimNum();
    auto dimCp = dim;
    ret = InputPreprocess(gradOutput, self, dimCp, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = GetPaddingTensor(dim, padding, uniqueExecutor.get());
    auto padFlag = CheckPaddingValue(padding, mode);
    const aclTensor* pad2dbackwardResult = nullptr;
    auto originOutDataType = gradOutput->GetDataType();
    // cast to fp32
    if (mode == REPLICATION_MODE && padFlag &&
        (originOutDataType == op::DataType::DT_FLOAT16 || originOutDataType == op::DataType::DT_BF16)) {
        gradOutput = l0op::Cast(gradOutput, op::DataType::DT_FLOAT, uniqueExecutor.get());
        OP_LOGD("[PadV3Grad] FP16 or BF16 Cast to FP32: true");
    }
    pad2dbackwardResult = l0op::PadV3Grad(gradOutput, paddingsTensor, mode, true, padFlag, uniqueExecutor.get());
    CHECK_RET(pad2dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3 is dim
    if (dimCp == 3) {
        // 0 is index
        const int64_t appendDim[] = {0};
        // 1 is the dim num to be squeezed
        aclIntArray* dimArray = (uniqueExecutor.get())->AllocIntArray(appendDim, 1);
        pad2dbackwardResult = l0op::SqueezeNd(pad2dbackwardResult, dimArray, uniqueExecutor.get());
        CHECK_RET(pad2dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // cast to fp16 or bf16
    if (mode == REPLICATION_MODE && padFlag &&
        (originOutDataType == op::DataType::DT_FLOAT16 || originOutDataType == op::DataType::DT_BF16)) {
        pad2dbackwardResult = l0op::Cast(pad2dbackwardResult, originOutDataType, uniqueExecutor.get());
        OP_LOGD("[PadV3Grad] FP32 Cast to FP16 or BF16: true");
    }
    // 如果出参gradInput是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(pad2dbackwardResult, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReflectionPad2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnReflectionPad2dBackward, DFX_IN(gradOutput, self, padding), DFX_OUT(gradInput));
    return commonPad2dBackward(gradOutput, self, padding, REFLECTION_MODE, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnReflectionPad2dBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReflectionPad2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnReplicationPad2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnReplicationPad2dBackward, DFX_IN(gradOutput, self, padding), DFX_OUT(gradInput));
    return commonPad2dBackward(gradOutput, self, padding, REPLICATION_MODE, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnReplicationPad2dBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReplicationPad2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif