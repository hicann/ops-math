/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confusion_matrix_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CONFUSION_MATRIX_H_
#define OPS_OP_PROTO_INC_CONFUSION_MATRIX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the confusion matrix from predictions and labels .

*@par Inputs:
*Three inputs, including:
*@li labels: A Tensor. Must be one of the following types: float16, float32,
*int32, int8, uint8. 1D. Has format ND.
*@li predictions: A Tensor. Must be one of the following types: float16,
*float32, int32, int8, uint8. 1D. Has format ND.
*@li weights: A optional Tensor. Must be one of the following types: float16, float32,
*int32, int8, uint8. 1D. Has format ND. \n

*@par Attributes:
*@li num_classes: An integer for the shape of the output matrix.
*@li dtype: Data type of the confusion matrix. \n

*@par Outputs:
*y: A Tensor. 1D. Has format ND. Has the same type and format as input "labels" . \n

*@attention Constraints:
*@li "weights", "labels", and "predictions" are 1D tensors.
*@li The output is with shape (num_classes, num_classes),
*where, 1 <= num_classes <= 4096 . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ConfusionMatrix.
*/
#ifndef OPS_PROTO_DEF_CONFUSIONMATRIX
#define OPS_PROTO_DEF_CONFUSIONMATRIX
REG_OP(ConfusionMatrix)
    .INPUT(labels, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .INPUT(predictions, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(weights, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(num_classes, Int)
    .REQUIRED_ATTR(dtype, String)
    .OP_END_FACTORY_REG(ConfusionMatrix)
#endif // OPS_PROTO_DEF_CONFUSIONMATRIX

} // namespace ge

#endif // OPS_OP_PROTO_INC_CONFUSION_MATRIX_H_
