/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atan_proto.h
 * \brief
 */
#ifndef ATAN_PROTO_H_
#define ATAN_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Computes the trignometric inverse tangent of x element-wise.
* The atan operation returns the inverse of tan, such that if y = tan(x) then, x = atan(y).

*
*@par Inputs:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* bfloat16, float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
* The output of atan will lie within the invertible range of tan, i.e (-pi/2, pi/2).
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Atan.
*
*/
REG_OP(Atan)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Atan)
} // namespace ge
#endif // ATAN_PROTO_H_