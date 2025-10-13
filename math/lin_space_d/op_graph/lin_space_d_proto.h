/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_d_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LIN_SPACE_OPS_H_
#define OPS_OP_PROTO_INC_LIN_SPACE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Generates values in an interval .

*@par Inputs:
* Three ND inputs, including:
*@li start: A 1D Tensor, for the first entry in the range.
* Type must be one of the following types:
* float32, float16, bfloat16, int32, int16, int8, uint8.
*@li end: A 1D Tensor, for the last entry in the range.
* Type must be one of the following types:
* float32, float16, bfloat16, int32, int16, int8, uint8.
*@li num: A 1D Tensor of type int64, for the common difference of the entries . \n

*@par Outputs:
*z: A 1D Tensor, Type is float32. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator lin_space_d.
*/
REG_OP(LinSpaceD)
    .INPUT(start, TensorType({DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .INPUT(end, TensorType({DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .INPUT(num, TensorType({DT_INT64}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LinSpaceD)

} // namespace ge

#endif // OPS_OP_PROTO_INC_LIN_SPACE_OPS_H_