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
 * \file const_proto.h
 * \brief Definition of the Const operator.
 */
#ifndef CONST_PROTO_H_
#define CONST_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Creates a constant tensor from a tensor-like object. This operator is used for inference.
 Operator Const has the same definition as operator Constant. \n

 *@par Attributes:
 *value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

 *@par Outputs:
 *y: A constant tensor. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator Const.
 */
#ifndef OPS_PROTO_DEF_CONST
#define OPS_PROTO_DEF_CONST
REG_OP(Const).OUTPUT(y, TensorType::ALL()).ATTR(value, Tensor, Tensor()).OP_END_FACTORY_REG(Const)
#endif
} // namespace ge
#endif // CONST_PROTO_H_
