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
 * \file ref_data_proto.h
 * \brief
 */
#ifndef REF_DATA_PROTO_H_
#define REF_DATA_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Input data for other operators.
        It could be overwritten by ref ops, acting like a variable. \n

* @par Inputs:
* x: A tensor. \n

* @par Attributes:
* index: Index of the input tensor.The data type must be int32 or int64.
  Assume that net has two data nodes and one ref_data node, previous two data index set as (0, 1),
  and the left ref_data should be set 2. \n

* @par Outputs:
* x: A tensor. Same with input name, which means ref with input. \n
*/
#ifndef OPS_PROTO_DEF_REFDATA
#define OPS_PROTO_DEF_REFDATA
REG_OP(RefData)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .ATTR(index, Int, 0)
    .DATATYPE(T, TensorType::ALL())
    .OP_END_FACTORY_REG(RefData)
#endif
} // namespace ge
#endif // REF_DATA_PROTO_H_
