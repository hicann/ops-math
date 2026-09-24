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
 * \file parallel_concat_proto.h
 * \brief ParallelConcat 算子 GE IR 原型定义（图模式）。
 */

#ifndef OPS_OP_PROTO_INC_PARALLEL_CONCAT_H_
#define OPS_OP_PROTO_INC_PARALLEL_CONCAT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Concatenates a list of N tensors along the first dimension.
 *@par Inputs:
 *@li values: A list of Tensors. Must be one of the following types: float32,
 * float16, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
 * bool. Tensors to be concatenated. All must have size 1 in the first dimension
 * and same shape. It's a dynamic input.
 *@par Attributes:
 *@li shape: A required list of ints. Explicit output shape [N, d_1..d_k]; must
 * be fully defined and non-negative, with shape[0] == N and shape[1:] equal to
 * every input's shape[1:].
 *@li N: A required int. The number of dynamic_input "values" instances.
 *@par Outputs:
 *@li output_data: The concatenated tensor with same type as "values"; row i is
 * a bitwise copy of input i.
 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator ParallelConcat.
 */
#ifndef OPS_PROTO_DEF_PARALLELCONCAT
#define OPS_PROTO_DEF_PARALLELCONCAT
REG_OP(ParallelConcat)
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                       DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .OUTPUT(output_data, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                     DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ParallelConcat)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_PARALLEL_CONCAT_H_
