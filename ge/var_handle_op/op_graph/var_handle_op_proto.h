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
 * \file var_handle_op_proto.h
 * \brief Definition of the VarHandleOp operator.
 */
#ifndef VAR_HANDLE_OP_PROTO_H_
#define VAR_HANDLE_OP_PROTO_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Creates a handle to a Variable resource. \n

* @par Outputs:
* y:A Tensor of type resource. \n

* @par Attributes:
* @li container: optional, string. the container this
variable is placed in.
* @li shared_name: optional, string.the name by which
 this variable is referred to.
* @li dtype: required, type. the output of type.
* @li shape: optional, ListInt. the output of shape. \n

* @see VarHandleOp.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

#ifndef OPS_PROTO_DEF_VARHANDLEOP
#define OPS_PROTO_DEF_VARHANDLEOP
REG_OP(VarHandleOp)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(shape, ListInt, ge::UNKNOWN_SHAPE)
    .OUTPUT(y, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(VarHandleOp)
#endif
} // namespace ge
#endif // VAR_HANDLE_OP_PROTO_H_
