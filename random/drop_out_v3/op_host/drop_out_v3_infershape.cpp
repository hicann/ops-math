/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file drop_out_v3_infershape.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_util.h"

using namespace ge;
namespace ops {
// ------------------- DropOutV3 Ops START---------------------
static constexpr size_t DropOutV3_IN_X_IDX = 0;
static constexpr size_t DropOutV3_IN_NOISE_IDX = 1;
static constexpr size_t DropOutV3_IN_P_IDX = 2;
static constexpr size_t DropOutV3_IN_SEED_IDX = 3;
static constexpr size_t DropOutV3_IN_OFFSET_IDX = 4;
static constexpr size_t MAX_DIM_NUM = 8;

static constexpr size_t DropOutV3_OUT_Y_IDX = 0;
static constexpr size_t DropOutV3_OUT_MASK_IDX = 1;
static constexpr size_t RATIO_PARAM = 8;

static constexpr int64_t MASK_ALIGN_SIZE = 128;
static constexpr int64_t MASK_BIT_TO_UINT8 = 8;

ge::graphStatus InfershapeForDropOutV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do DropOutV3Infershape.");

    // 获取输入值shape
    const gert::Shape* xInputShape = context->GetRequiredInputShape(DropOutV3_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputShape);

    const gert::Shape* noiseInputShape = context->GetOptionalInputShape(DropOutV3_IN_NOISE_IDX);
    if (noiseInputShape != nullptr) {
        if (noiseInputShape->GetDimNum() > MAX_DIM_NUM) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape* pInputShape = context->GetRequiredInputShape(DropOutV3_IN_P_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pInputShape);

    const gert::Shape* seedInputShape = context->GetRequiredInputShape(DropOutV3_IN_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seedInputShape);

    const gert::Shape* offsetInputShape = context->GetRequiredInputShape(DropOutV3_IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetInputShape);

    // 获取输出值shape
    gert::Shape* yOutputShape = context->GetOutputShape(DropOutV3_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutputShape);

    gert::Shape* maskOutputShape = context->GetOutputShape(DropOutV3_OUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskOutputShape);

    // 设置out输出
    int64_t xDimNum = static_cast<int64_t>(xInputShape->GetDimNum());
    yOutputShape->SetDimNum(xDimNum);
    for (int64_t i = 0; i < xDimNum; i++) {
        yOutputShape->SetDim(i, xInputShape->GetDim(i));
    }

    // set mask shape
    int64_t xShapeSize = xInputShape->GetShapeSize();
    int64_t maskSize = (xShapeSize + MASK_ALIGN_SIZE - 1) / MASK_ALIGN_SIZE * MASK_ALIGN_SIZE / MASK_BIT_TO_UINT8;
    maskOutputShape->SetDimNum(1);
    maskOutputShape->SetDim(0, maskSize);

    // 输入输出校验
    OP_CHECK_IF(
        xInputShape->GetDimNum() != yOutputShape->GetDimNum(),
        OP_LOGE(
            context->GetNodeName(), "%s",
            ConcatString(
                "Shape of input [", xInputShape->GetDimNum(), "] should be be match with output [",
                yOutputShape->GetDimNum(), "]").c_str()),
        return GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "xInputShape = %s.", Ops::Base::ToString(*xInputShape).c_str());
    OP_LOGI(context->GetNodeName(), "yOutputShape = %s.", Ops::Base::ToString(*yOutputShape).c_str());
    OP_LOGI(context->GetNodeName(), "maskOutputShape = %s", Ops::Base::ToString(*maskOutputShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do DropOutV3Infershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DropOutV3).InferShape(InfershapeForDropOutV3);
// -------------------DropOutV3 Ops END---------------------
} // namespace ops