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
 * \file variable_proto.h
 * \brief Definition of the Variable operator.
 */
#ifndef VARIABLE_PROTO_H_
#define VARIABLE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
 * @brief Creates a variable tensor . \n

 * @par Inputs:
 * x: A tensor, used to assign a value to the variable tensor internally.
 The caller does not need to pass the value of the variable tensor . \n

 * @par Attributes:
 * @li index: An integer. Index of the input tensor.
 * @li value: A tensor, used to pass and record the value of the variable tensor.
 * @li container: A string. The container of the variable tensor.
 * @li shared_name: A string. The shared name of the variable tensor . \n

 * @par Outputs:
 * y: The created variable tensor . \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator Variable.
 */
#ifndef OPS_PROTO_DEF_VARIABLE
#define OPS_PROTO_DEF_VARIABLE
REG_OP(Variable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(Variable)
#endif
} // namespace ge
#endif // VARIABLE_PROTO_H_
