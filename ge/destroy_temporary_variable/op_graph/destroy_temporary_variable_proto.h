/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file destroy_temporary_variable_proto.h
 * \brief Definition of the DestroyTemporaryVariable operator.
 */
#ifndef DESTROY_TEMPORARY_VARIABLE_PROTO_H_
#define DESTROY_TEMPORARY_VARIABLE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Destroys the temporary variable and returns its final value.
 * All other uses of the temporary variable must have been executed before this op . \n

 * @par Inputs:
 * x: A reference to the temporary variable tensor . \n

 * @par Attributes:
 * var_name: A required string. Name of the temporary variable.
 * Must be the same as the "var_name" attribute of the reference to the temporary variable tensor . \n

 * @par Outputs:
 * y: Final value of the reference to the temporary variable tensor . \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator DestroyTemporaryVariable.

 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_DESTROYTEMPORARYVARIABLE
#define OPS_PROTO_DEF_DESTROYTEMPORARYVARIABLE
REG_OP(DestroyTemporaryVariable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(var_name, String, "")
    .OP_END_FACTORY_REG(DestroyTemporaryVariable)
#endif
} // namespace ge
#endif // DESTROY_TEMPORARY_VARIABLE_PROTO_H_
