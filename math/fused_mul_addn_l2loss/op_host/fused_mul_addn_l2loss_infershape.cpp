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
 * \file fused_mul_addn_l2loss_infershape.cpp
 * \brief y1 shape = x1 shape；y2 = 单元素标量（shape (1,)）
 *        与 canndev built-in FusedMulAddNL2lossInferShape（elewise_calculation_ops.cc）
 *        功能完全一致，gert infershape 2.0 写法
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus FusedMulAddNL2lossInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    const gert::Shape* x3Shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x3Shape);
    gert::Shape* y1Shape = context->GetOutputShape(0);
    gert::Shape* y2Shape = context->GetOutputShape(1);

    // ND 格式合法 rank 范围为 0..8
    constexpr size_t kMaxNdrRank = 8;
    size_t dimNum = x1Shape->GetDimNum();
    OP_CHECK_IF(dimNum > kMaxNdrRank,
                OP_LOGE(context->GetNodeName(), "invalid x1 rank %zu: ND format supports rank 0..8", dimNum),
                return ge::GRAPH_FAILED);

    // README 约束：x2 与 x1 shape 必须一致（逐维相等；未知维 -1/-2 视为通配）
    OP_CHECK_IF(x2Shape->GetDimNum() != dimNum,
                OP_LOGE(context->GetNodeName(), "x2 rank %zu must equal x1 rank %zu", x2Shape->GetDimNum(), dimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; i++) {
        const int64_t d1 = x1Shape->GetDim(i);
        const int64_t d2 = x2Shape->GetDim(i);
        OP_CHECK_IF(d1 >= 0 && d2 >= 0 && d1 != d2,
                    OP_LOGE(context->GetNodeName(), "x2 dim[%zu]=%ld must equal x1 dim[%zu]=%ld", i, d2, i, d1),
                    return ge::GRAPH_FAILED);
    }

    // README 约束：x3 为单元素张量（kernel 按 x3[0] 标量广播）；未知维跳过元素数校验
    size_t x3DimNum = x3Shape->GetDimNum();
    int64_t x3Elements = 1;
    bool x3Known = true;
    for (size_t i = 0; i < x3DimNum; i++) {
        const int64_t d = x3Shape->GetDim(i);
        if (d < 0) {
            x3Known = false;
            break;
        }
        x3Elements *= d;
    }
    OP_CHECK_IF(x3Known && x3Elements != 1,
                OP_LOGE(context->GetNodeName(), "x3 must be a single-element tensor, got %ld elements", x3Elements),
                return ge::GRAPH_FAILED);

    // y1 shape = x1 shape
    y1Shape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        y1Shape->SetDim(i, x1Shape->GetDim(i));
    }

    // y2 为单元素标量（shape (1,)）：与 golden（reshape(1)）、仓库 ST CSV 一致；
    // 0 维空 shape 会触发 CANN 静态编译 add_op_param_to_workspace 对空 shape reduce() 崩溃
    y2Shape->SetDimNum(1);
    y2Shape->SetDim(0, 1);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedMulAddNL2loss).InferShape(FusedMulAddNL2lossInferShape);
} // namespace ops
