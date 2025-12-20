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
 * \file clip_by_value_v2_infershape.cpp
 * \brief
 */

#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_NUM_THREE = 3;

static bool BroadcastDim(int64_t& dim1, const int64_t dim2)
{
    /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
    dim -1  0  1  d2
    -1  -1  0 -1  d2
    0    0  0  0  E
    1   -1  0  1  d2
    d1  d1  E  d1 E
    */

    // no need to broadcast
    if (dim1 == dim2) {
        return true;
    }

    if ((dim1 != 1) && (dim2 != 1)) {
        // dynamic shape infershape
        if ((dim1 == -1) || (dim2 == -1)) {
            dim1 = (dim1 == -1) ? dim2 : dim1;
            return true;
        }

        OP_LOGE("BroadcastDim", "%ld and %ld cannot broadcast!", dim1, dim2);
        return false;
    }

    // static shape infershape
    dim1 = (dim1 == 1) ? dim2 : dim1;

    return true;
}

/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shapeOutput: gert::Shape*, output shape
 * @return succeed or not
 */
static bool BroadcastShapeToOutShape(const gert::Shape* shape, gert::Shape* shapeOutput)
{
    OP_LOGD(
        "BroadcastShapeToOutShape", "start broadcast %s to %s!", Ops::Base::ToString(*shape).c_str(),
        Ops::Base::ToString(*shapeOutput).c_str());

    if (Ops::Base::IsUnknownRank(*shape) || Ops::Base::IsUnknownRank(*shapeOutput)) {
        OP_LOGD("BroadcastShapeToOutShape", "the input shape is [-2], set output shape is [-2]!");
        Ops::Base::SetUnknownRank(*shapeOutput);
        return true;
    }

    size_t shapeLen = shape->GetDimNum();
    size_t shapeYLen = shapeOutput->GetDimNum();
    if (shapeLen > shapeYLen) {
        shapeOutput->SetDimNum(shapeLen);
        size_t lenSub = shapeLen - shapeYLen;
        for (size_t i = shapeYLen; i > 0; i--) {
            int64_t dim1 = shape->GetDim(lenSub + i - 1);
            int64_t dim2 = shapeOutput->GetDim(i - 1);
            OP_CHECK_IF(
                !BroadcastDim(dim1, dim2),
                OP_LOGE("BroadcastShapeToOutShape", "%ld and %ld cannot broadcast!", dim1, dim2), return false);
            shapeOutput->SetDim(lenSub + i - 1, dim1);
        }
        for (size_t i = 0; i < lenSub; i++) {
            shapeOutput->SetDim(i, shape->GetDim(i));
        }
    } else {
        auto lenSub = shapeYLen - shapeLen;
        for (size_t i = 0; i < shapeLen; i++) {
            int64_t dim1 = shapeOutput->GetDim(lenSub + i);
            int64_t dim2 = shape->GetDim(i);
            OP_CHECK_IF(
                !BroadcastDim(dim1, dim2),
                OP_LOGE("BroadcastShapeToOutShape", "%ld and %ld cannot broadcast!", dim1, dim2), return false);
            shapeOutput->SetDim(lenSub + i, dim1);
        }
    }
    return true;
}

static bool BroadcastShape(const gert::Shape** in_shapes, size_t size, gert::Shape* out_shape)
{
    OP_CHECK_IF(size == 0, OP_LOGE("BroadcastShape", "in_shapes is empty!"), return false);
    *out_shape = *in_shapes[0];

    for (size_t i = 1; i < size; i++) {
        OP_CHECK_IF(
            !BroadcastShapeToOutShape(in_shapes[i], out_shape),
            OP_LOGE(
                "BroadcastShape", "shape %s and %s cannot broadcast!", Ops::Base::ToString(*in_shapes[i]).c_str(),
                Ops::Base::ToString(*out_shape).c_str()),
            return false);
    }

    return true;
}

static ge::graphStatus InferDataType4ClipByValueV2(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataType4ClipByValueV2.");
    const ge::DataType x1DataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1DataType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ClipByValueV2(gert::InferShapeContext* context)
{
    OP_LOGI("Begin InferShape4ClipByValueV2.");
    std::array<const gert::Shape*, INPUT_NUM_THREE> to_broadcast_shapes;
    for (size_t i = 0; i < INPUT_NUM_THREE; i++) {
        auto in_shape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
        to_broadcast_shapes[i] = in_shape;
    }
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    OP_CHECK_IF(
        !BroadcastShape(to_broadcast_shapes.data(), INPUT_NUM_THREE, out_shape),
        OP_LOGE(context->GetNodeName(), "BroadcastShape failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ClipByValueV2).InferShape(InferShape4ClipByValueV2).InferDataType(InferDataType4ClipByValueV2);

} // namespace ops
