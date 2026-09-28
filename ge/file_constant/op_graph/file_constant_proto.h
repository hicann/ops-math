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
 * \file file_constant_proto.h
 * \brief Definition of the FileConstant operator.
 */
#ifndef FILECONSTANT_PROTO_H_
#define FILECONSTANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Creates a file constant tensor, The operator is used to process the very large weight which is store in file.
 \n

 *@par Attributes:
 *file_path: A string, used to record file path. \n
 *file_id: A string, used to record file id. \n
 *shape: data shape. \n
 *dtype: data type. \n

 *@par Outputs:
 *y: The FileConstant tensor. \n
 */
#ifndef OPS_PROTO_DEF_FILECONSTANT
#define OPS_PROTO_DEF_FILECONSTANT
REG_OP(FileConstant)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(file_path, String, "")
    .ATTR(file_id, String, "")
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(FileConstant)
#endif
} // namespace ge
#endif // FILECONSTANT_PROTO_H_
