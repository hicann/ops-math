/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file view_copy_proto.h
 * \brief
 */
#ifndef OP_PROTO_VIEW_COPY_PROTO_H_
#define OP_PROTO_VIEW_COPY_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Copy the src tensor to the dst tensor according the special parameter. \n

*@par Inputs:
*Eight inputs, including:
*@li dst: A tensor. Must be one of the following types:
*float32, float16, bfloat16, int8, uint8, int16, uint16,
*int32, uint32, int64, uint64, bool, hifloat8,
*float8_e5m2, float8_e4m3fn.
*@li dst_size: A tensor. Must be one of the following types: int32, int64.
*@li dst_stride: A tensor. Must be one of the following types: int32, int64.
*@li dst_storage_offset: A tensor. Must be one of the following types: int32, int64.
*@li src: A tensor. Must be one of the following types:
*float32, float16, bfloat16, int8, uint8, int16, uint16,
*int32, uint32, int64, uint64, bool, hifloat8, float8_e5m2, float8_e4m3fn.
*@li src_size: A tensor. Must be one of the following types: int32, int64.
*@li src_stride: A tensor. Must be one of the following types: int32, int64.
*@li src_storage_offset: The storage_offset of src tensor . Must be one of the following types:
*int32, int64.
*@par Outputs:
*dst: An ref tensor. Must be one of the following types:
*float32, float16, bfloat16, int8, uint8, int16, uint16,
*int32, uint32, int64, uint64, bool.
*/

REG_OP(ViewCopy)
    .INPUT(dst, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(dst_size, TensorType::IndexNumberType())
    .INPUT(dst_stride, TensorType::IndexNumberType())
    .INPUT(dst_storage_offset, TensorType::IndexNumberType())
    .INPUT(src, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(src_size, TensorType::IndexNumberType())
    .INPUT(src_stride, TensorType::IndexNumberType())
    .INPUT(src_storage_offset, TensorType::IndexNumberType())
    .OUTPUT(dst, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(ViewCopy)

} // namespace ge
#endif // OP_PROTO_VIEW_COPY_PROTO_H_