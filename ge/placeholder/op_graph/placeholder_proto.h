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
 * \file placeholder_proto.h
 * \brief Definition of the PlaceHolder operator.
 */
#ifndef PLACEHOLDER_PROTO_H_
#define PLACEHOLDER_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Inserts a placeholder for a tensor that will be always fed. \n

 *@par Inputs:
 *x: A tensor. \n

 *@par Attributes:
 *@li peerIndex: An integer type. The index of the corresponding "end" node connected to.
 *@li parentId: A string, used to check if the nodes are from the saved parent node.
 *@li parentOpType: A string. Op type of the original node.
 *@li anchorIndex: An integer, used to check if the node is from the saved anchor. \n

 *@par Outputs:
 *y: The created placeholder tensor. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator PlaceHolder.
 */
#ifndef OPS_PROTO_DEF_PLACEHOLDER
#define OPS_PROTO_DEF_PLACEHOLDER
REG_OP(PlaceHolder)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0)
    .ATTR(parentId, String, "")
    .ATTR(parentOpType, String, "")
    .ATTR(anchorIndex, Int, 0)
    .OP_END_FACTORY_REG(PlaceHolder)
#endif
} // namespace ge
#endif // PLACEHOLDER_PROTO_H_
