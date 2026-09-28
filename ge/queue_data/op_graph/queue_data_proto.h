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
 * \file queue_data_proto.h
 * \brief
 */
#ifndef QUEUE_DATA_PROTO_H_
#define QUEUE_DATA_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Queue data for other operators. \n
* @par Attributes:
* @li index: Index of the input tensor.The data type must be int32 or int64.
Assume that net has three data nodes, one should be set 0, another should
be set 1, and the left should be set 2.
* @li queue_name: An optional string that indicates the queue name. Defaults to "".
* @li output_types: An optional type list that indicates the data types of outputs data.
* @li output_shapes: An optional int list list that indicates the list shapes of outputs data.
* @par Outputs:
* y: A DT_UINT8 tensor. \n
*/
#ifndef OPS_PROTO_DEF_QUEUEDATA
#define OPS_PROTO_DEF_QUEUEDATA
REG_OP(QueueData)
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(index, Int, 0)
    .ATTR(queue_name, String, "")
    .ATTR(output_types, ListType, {})
    .ATTR(output_shapes, ListListInt, {{}, {}})
    .OP_END_FACTORY_REG(QueueData)
#endif
} // namespace ge
#endif // QUEUE_DATA_PROTO_H_
