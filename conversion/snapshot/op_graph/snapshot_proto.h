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
 * \file snapshot_proto.h
 * \brief Definition of the Snapshot operator.
 */
#ifndef SNAPSHOT_PROTO_H_
#define SNAPSHOT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Returns a copy of the input tensor.
 *
 * @par Inputs:
 * x: A tensor.
 *
 * @par Outputs:
 * y: A copy of input tensor x.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator Snapshot.
 */
#ifndef OPS_PROTO_DEF_SNAPSHOT
#define OPS_PROTO_DEF_SNAPSHOT
REG_OP(Snapshot)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OP_END_FACTORY_REG(Snapshot)
#endif
} // namespace ge
#endif // SNAPSHOT_PROTO_H_
