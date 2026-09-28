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
 * \file temporary_variable_proto.h
 * \brief Definition of the TemporaryVariable operator.
 */
#ifndef TEMPORARY_VARIABLE_PROTO_H_
#define TEMPORARY_VARIABLE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Returns a temporary variable tensor. After the use of TemporaryVariable,
 * pass the reference to the variable tensor to the matching DestroyTemporaryVariable op for destruction . \n

 * @par Attributes:
 * @li shape: A required list of int32 or int64. The shape of the variable tensor.
 * @li dtype: Required. The type of elements in the variable tensor.
 * @li var_name: An optional string. The name of the variable to be created . \n

 * @par Outputs:
 * y: The created variable tensor . \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator TemporaryVariable.

 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_TEMPORARYVARIABLE
#define OPS_PROTO_DEF_TEMPORARYVARIABLE
REG_OP(TemporaryVariable)
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(var_name, String, "")
    .OP_END_FACTORY_REG(TemporaryVariable)
#endif
} // namespace ge
#endif // TEMPORARY_VARIABLE_PROTO_H_
