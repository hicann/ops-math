/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confusion_matrix_infershape.cpp
 * \brief Infer shape implementation for ConfusionMatrix operator
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

#include <string>

using namespace ge;
namespace ops {
static constexpr size_t kAttrIndexNumClasses = 0U;
static constexpr size_t kInputIndexLabels = 0U;
static constexpr size_t kInputIndexPredictions = 1U;
static constexpr size_t kInputIndexWeights = 2U;
static constexpr size_t kOutputIndexY = 0U;
static constexpr int64_t kNumClassesMin = 1;
static constexpr int64_t kNumClassesMax = 4096;
static constexpr size_t kOutputDimNum = 2U;
static constexpr size_t kOutputDim0 = 0U;
static constexpr size_t kOutputDim1 = 1U;
static constexpr size_t kInputDimNum = 1U;
static constexpr size_t kInputDim0 = 0U;

ge::graphStatus InferShapeForConfusionMatrix(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForConfusionMatrix");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* numClassesPtr = attrs->GetAttrPointer<int64_t>(kAttrIndexNumClasses);
    OP_CHECK_NULL_WITH_CONTEXT(context, numClassesPtr);

    int64_t numClasses = *numClassesPtr;
    OP_CHECK_IF(
        (numClasses < kNumClassesMin || numClasses > kNumClassesMax),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_classes", std::to_string(numClasses).c_str(),
                                              "num_classes must be in range [1, 4096]."),
        return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(kOutputIndexY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Shape* labelsShape = context->GetInputShape(kInputIndexLabels);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelsShape);
    const gert::Shape* predictionsShape = context->GetInputShape(kInputIndexPredictions);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictionsShape);

    // labels/predictions 为 unknown rank(-2) 时无法校验 1D 约束，输出透传 unknown rank；
    // unknown shape(-1，rank 已知)时输出仍由 num_classes 决定，走下方固定 [num_classes, num_classes] 分支
    if (Ops::Base::IsUnknownRank(*labelsShape) || Ops::Base::IsUnknownRank(*predictionsShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGI(context->GetNodeName(), "labels/predictions is unknown rank, set output to unknown rank");
        return ge::GRAPH_SUCCESS;
    }

    // 公开契约 R1/R2：labels/predictions 必须为 1 维且形状一致；
    // 动态维度(-1)无法静态判定相等时放行，交由 tiling/运行时兑底。
    OP_CHECK_IF((labelsShape->GetDimNum() != kInputDimNum || predictionsShape->GetDimNum() != kInputDimNum),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "labels/predictions",
                                             (std::to_string(labelsShape->GetDimNum()) + "D/" +
                                              std::to_string(predictionsShape->GetDimNum()) + "D")
                                                 .c_str(),
                                             "1D"),
                return ge::GRAPH_FAILED);

    const int64_t labelsDim = labelsShape->GetDim(kInputDim0);
    const int64_t predictionsDim = predictionsShape->GetDim(kInputDim0);
    OP_CHECK_IF((labelsDim >= 0 && predictionsDim >= 0 && labelsDim != predictionsDim),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "labels/predictions",
                    (std::to_string(labelsDim) + " and " + std::to_string(predictionsDim)).c_str(),
                    "The shapes of labels and predictions must be the same."),
                return ge::GRAPH_FAILED);

    // 公开契约 R3：weights 为可选输入，非空时必须为 1 维且与 labels 形状一致。
    const gert::Shape* weightsShape = context->GetOptionalInputShape(kInputIndexWeights);
    if (weightsShape != nullptr && !Ops::Base::IsUnknownRank(*weightsShape)) {
        OP_CHECK_IF((weightsShape->GetDimNum() != kInputDimNum),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "weights",
                                                 (std::to_string(weightsShape->GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
        const int64_t weightsDim = weightsShape->GetDim(kInputDim0);
        OP_CHECK_IF((weightsDim > 0 && labelsDim > 0 && weightsDim != labelsDim),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "weights",
                        (std::to_string(weightsDim) + " and " + std::to_string(labelsDim)).c_str(),
                        "The shape of weights must be the same as labels when provided."),
                    return ge::GRAPH_FAILED);
    }

    yShape->SetDimNum(kOutputDimNum);
    yShape->SetDim(kOutputDim0, numClasses);
    yShape->SetDim(kOutputDim1, numClasses);

    OP_LOGI(context->GetNodeName(), "End to do InferShapeForConfusionMatrix");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionMatrix).InferShape(InferShapeForConfusionMatrix);

} // namespace ops
