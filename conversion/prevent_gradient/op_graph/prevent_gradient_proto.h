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
 * \file prevent_gradient_proto.h
 * \brief Definition of the PreventGradient operator.
 */
#ifndef PREVENT_GRADIENT_PROTO_H_
#define PREVENT_GRADIENT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Forwards its input tensor and reports an error if a gradient is requested.
 *
 * @par Inputs:
 * x: A tensor.
 *
 * @par Attributes:
 * message: Message printed when a gradient is requested.
 *
 * @par Outputs:
 * y: The input tensor.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator PreventGradient.
 */
#ifndef OPS_PROTO_DEF_PREVENTGRADIENT
#define OPS_PROTO_DEF_PREVENTGRADIENT
REG_OP(PreventGradient)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(message, String, "")
    .OP_END_FACTORY_REG(PreventGradient)
#endif
} // namespace ge
#endif // PREVENT_GRADIENT_PROTO_H_
