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
 * \file dropout_gen_mask_infershape.cpp
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus DropOutGenMaskInferDataType(gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }
  OP_LOGD(context->GetNodeName(), "Begin to do infer data type.");
  context->SetOutputDataType(0, ge::DT_UINT8);
  OP_LOGD(context->GetNodeName(), "End to do infer data type.");
  return GRAPH_SUCCESS;
}


IMPL_OP(StatelessDropOutGenMask).InferDataType(DropOutGenMaskInferDataType);

}  // namespace ops