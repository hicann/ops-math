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
 * \file guarantee_const_proto.h
 * \brief Definition of the GuaranteeConst operator.
 */
#ifndef GUARANTEE_CONST_PROTO_H_
#define GUARANTEE_CONST_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Gives a guarantee to the runtime that the input tensor is a constant. \n

 *@par Inputs:
 *x: A tensor. \n

 *@par Outputs:
 *y: The input tensor. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator GuaranteeConst.

 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_GUARANTEECONST
#define OPS_PROTO_DEF_GUARANTEECONST
REG_OP(GuaranteeConst).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).OP_END_FACTORY_REG(GuaranteeConst)
#endif
} // namespace ge
#endif // GUARANTEE_CONST_PROTO_H_
