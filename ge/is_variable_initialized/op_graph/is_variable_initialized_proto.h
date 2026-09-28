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
 * \file is_variable_initialized_proto.h
 * \brief Definition of the IsVariableInitialized operator.
 */
#ifndef IS_VARIABLE_INITIALIZED_PROTO_H_
#define IS_VARIABLE_INITIALIZED_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Checks whether a tensor has been initialized. Outputs boolean scalar indicating whether the tensor has been
 initialized . \n

 * @par Inputs:
 * x: A Tensor of type float16, float32, double, bool, int8, uint8, uint16, int16, int32, uint32, uint64, int64.

 * @par Outputs:
 * y: A tensor, indicating whether "x" has been initialized . \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator IsVariableInitialized.
 */
#ifndef OPS_PROTO_DEF_ISVARIABLEINITIALIZED
#define OPS_PROTO_DEF_ISVARIABLEINITIALIZED
REG_OP(IsVariableInitialized)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsVariableInitialized)
#endif
} // namespace ge
#endif // IS_VARIABLE_INITIALIZED_PROTO_H_
