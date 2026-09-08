/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_DENSE_BINCOUNT_H_
#define OPS_OP_PROTO_INC_DENSE_BINCOUNT_H_
#include "graph/operator_reg.h"
#include "graph/types.h"
namespace ge {
/**
 * @brief Counts occurrences of integer bins for a dense 1D or 2D input.
 * @par Inputs: input and size are matching int32/int64 tensors; weights is int32, int64, float32 or double.
 * @par Outputs: output has one bin dimension of length size and follows the weights type.
 * @par Attributes: binary_output is an optional bool and defaults to false.
 * @par Third-party framework compatibility: TensorFlow DenseBincount.
 */
#ifndef OPS_PROTO_DEF_DENSEBINCOUNT
#define OPS_PROTO_DEF_DENSEBINCOUNT
REG_OP(DenseBincount)
    .INPUT(input, TensorType({DT_INT32, DT_INT64}))
    .INPUT(size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weights, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .ATTR(binary_output, Bool, false)
    .OP_END_FACTORY_REG(DenseBincount)
#endif
} // namespace ge
#endif
