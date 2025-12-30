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
 * \file masked_scale_infershape.cpp
 * \brief
 */

#include "op_host/infershape_elewise_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4MaskedScale(gert::InferShapeContext* context) {
  auto in_shape1 = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape1);
  auto in_shape2 = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape2);

  OP_CHECK_IF(in_shape1->GetShapeSize() != in_shape2->GetShapeSize(),
           OP_LOGE(
               context->GetNodeName(),
               "shapesize of x %s not match mask %s", ToString(*in_shape1).c_str(), ToString(*in_shape2).c_str()),
           return ge::GRAPH_FAILED);
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  OP_CHECK_IF(!BroadcastShape(in_shape1, in_shape2, out_shape),
           OP_LOGE(context->GetNodeName(), "shape %s and %s cannot broadcast!",
                   ToString(*in_shape2).c_str(), ToString(*in_shape1).c_str()),
           return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedScale).InferShape(InferShape4MaskedScale);

} // namespace ops