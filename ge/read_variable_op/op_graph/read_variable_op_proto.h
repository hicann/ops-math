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
 * \file read_variable_op_proto.h
 * \brief Definition of the ReadVariableOp operator.
 */
#ifndef READ_VARIABLE_OP_PROTO_H_
#define READ_VARIABLE_OP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Reads and returns the value of the input variable tensor. \n

 *@par Inputs:
 *x: A tensor must have numeric type. \n

 *@par Attributes:
 *dtype: Same as the input data type. The output data type. Defaults to int32. \n

 *@par Outputs:
 *y: A tensor must have numeric type. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator ReadVariableOp.
 */
#ifndef OPS_PROTO_DEF_READVARIABLEOP
#define OPS_PROTO_DEF_READVARIABLEOP
REG_OP(ReadVariableOp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ReadVariableOp)
#endif
} // namespace ge
#endif // READ_VARIABLE_OP_PROTO_H_
