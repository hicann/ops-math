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
 * \file pad_v3_proto.h
 * \brief
 */

#ifndef PAD_V3_PROTO_H_
#define PAD_V3_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Pads a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, bfloat16(only support on constant mode),
*     float32, double, int32, uint8, int16, int8, complex64, int64,
*     qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, bool.
* @li paddings: A Tensor of type int32 or int64.
* @li constant_values: A optional Tensor, dtype same as "x"

* @par Attributes:
* @li mode: An optional string, Defaults to "constant", indicates paddings mode,
*     support "constant", "reflect", "edge"
* @li paddings_contiguous: An optional bool value, Defaults to true.
*     If true, paddings is arranged as [[begin0, end0], [begin1, end1], ...]
*     If false, paddings is arranged as [[begin0, begin1], ..., [end0, end1], ...]

* @par Outputs:
* y: A Tensor of the same type as "x".

* @par Third-party framework compatibility:
* Compatible with ONNX operator Pad.
*/
REG_OP(PadV3)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(constant_values, TensorType::BasicType())
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(mode, String, "constant")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3)
} // namespace ge

#endif