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
 * \file placeholder_withdefault_proto.h
 * \brief Definition of the PlaceholderWithDefault operator.
 */
#ifndef PLACEHOLDER_WITHDEFAULT_PROTO_H_
#define PLACEHOLDER_WITHDEFAULT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Inserts a placeholder with default value for a tensor. \n

 *@par Inputs:
 *x: A tensor. \n

 *@par Attributes:
 *shape: tensor shape. \n

 *@par Outputs:
 *y: The created placeholder tensor. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator PlaceholderWithDefault.
 */
#ifndef OPS_PROTO_DEF_PLACEHOLDERWITHDEFAULT
#define OPS_PROTO_DEF_PLACEHOLDERWITHDEFAULT
REG_OP(PlaceholderWithDefault)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(PlaceholderWithDefault)
#endif
} // namespace ge
#endif // PLACEHOLDER_WITHDEFAULT_PROTO_H_
