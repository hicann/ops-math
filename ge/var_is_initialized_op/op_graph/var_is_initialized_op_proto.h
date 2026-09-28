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
 * \file var_is_initialized_op_proto.h
 * \brief Definition of the VarIsInitializedOp operator.
 */
#ifndef VAR_IS_INITIALIZED_OP_PROTO_H_
#define VAR_IS_INITIALIZED_OP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Checks whether a tensor has been initialized. Outputs boolean scalar indicating whether the tensor has been
initialized . \n

* @par Inputs:
* x: A tensor . \n

* @par Outputs:
* y: A tensor, indicating whether "x" has been initialized, and the data type is boolean . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator VarIsInitializedOp.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
#ifndef OPS_PROTO_DEF_VARISINITIALIZEDOP
#define OPS_PROTO_DEF_VARISINITIALIZEDOP
REG_OP(VarIsInitializedOp)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(VarIsInitializedOp)
#endif
} // namespace ge
#endif // VAR_IS_INITIALIZED_OP_PROTO_H_
