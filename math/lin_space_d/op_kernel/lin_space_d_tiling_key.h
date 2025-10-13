/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_d_tiling_key.h
 * \brief LinSpaceD operator tiling template parameter declaration
 */
#ifndef LIN_SPACE_D_TILING_KEY_H
#define LIN_SPACE_D_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define LIN_SPACE_D_TPL_BF16    27     // bfloat16
#define LIN_SPACE_D_TPL_UINT8   4      // uint8
#define LIN_SPACE_D_TPL_FP16    1      // float16
#define LIN_SPACE_D_TPL_FP32    0      // float32
#define LIN_SPACE_D_TPL_INT8    2      // int8
#define LIN_SPACE_D_TPL_INT16   6      // int16
#define LIN_SPACE_D_TPL_INT32   3      // int32

ASCENDC_TPL_ARGS_DECL(
    KernelLinSpaceD, 
    ASCENDC_TPL_DTYPE_DECL(
        DTYPE_START,  
        LIN_SPACE_D_TPL_BF16,
        LIN_SPACE_D_TPL_UINT8,
        LIN_SPACE_D_TPL_FP16,
        LIN_SPACE_D_TPL_FP32,
        LIN_SPACE_D_TPL_INT8,
        LIN_SPACE_D_TPL_INT16,
        LIN_SPACE_D_TPL_INT32
    ),
    ASCENDC_TPL_DTYPE_DECL(
        DTYPE_END,
        LIN_SPACE_D_TPL_BF16,
        LIN_SPACE_D_TPL_UINT8,
        LIN_SPACE_D_TPL_FP16,
        LIN_SPACE_D_TPL_FP32,
        LIN_SPACE_D_TPL_INT8,
        LIN_SPACE_D_TPL_INT16,
        LIN_SPACE_D_TPL_INT32
    )
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(
            DTYPE_START,
            LIN_SPACE_D_TPL_BF16,
            LIN_SPACE_D_TPL_UINT8,
            LIN_SPACE_D_TPL_FP16,
            LIN_SPACE_D_TPL_FP32,
            LIN_SPACE_D_TPL_INT8,
            LIN_SPACE_D_TPL_INT16,
            LIN_SPACE_D_TPL_INT32
        ),
        ASCENDC_TPL_DTYPE_SEL(
            DTYPE_END,
            LIN_SPACE_D_TPL_BF16,
            LIN_SPACE_D_TPL_UINT8,
            LIN_SPACE_D_TPL_FP16,
            LIN_SPACE_D_TPL_FP32,
            LIN_SPACE_D_TPL_INT8,
            LIN_SPACE_D_TPL_INT16,
            LIN_SPACE_D_TPL_INT32
        )
    )
);

#endif // LIN_SPACE_D_TILING_KEY_H