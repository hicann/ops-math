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
 * \file fill_v2_proto.h
 * \brief
 */
#ifndef FILL_V2_PROTO_H_
#define FILL_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fill the value to a tensor has the specified shape.

* @par Inputs:
* One inputs, including:
* @li dims: An Tensor, specify the shape that the value to fill.

* @par Attributes:
* @li value: An optional float value. Defaults to 0.0.

* @par Outputs:
* @li y: A Tensor. Has the shape specify by attr shape, and full of the value specify by attr value.

* @par Third-party framework compatibility
* Compatible with the ONNX operator ConstantOfShape.
*/
#ifndef OPS_PROTO_DEF_FILLV2
#define OPS_PROTO_DEF_FILLV2
REG_OP(FillV2)
    .INPUT(dims, TensorType({DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .ATTR(value, Float, 0)
    .OP_END_FACTORY_REG(FillV2)
#endif
} // namespace ge
#endif // OP_PROTO_FILL_V2_H_
