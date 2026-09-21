/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill_v2_tiling_key.h
 * \brief fill_v2 tiling key
 */

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef _FILL_V2_TILING_KEY_H_
#define _FILL_V2_TILING_KEY_H_

#define FILL_V2_TPL_FP16 1
#define FILL_V2_TPL_FP32 2
#define FILL_V2_TPL_DOUBLE 3
#define FILL_V2_TPL_INT8 4
#define FILL_V2_TPL_INT16 5
#define FILL_V2_TPL_INT32 6
#define FILL_V2_TPL_INT64 7

// ElementwiseSch16B 的 schMode 模板参数当前为固定值（框架内未按 shape 区分调度模式），不作为 tiling key 维度
#define FILL_V2_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(FillV2, ASCENDC_TPL_DTYPE_DECL(dType, FILL_V2_TPL_FP16, FILL_V2_TPL_FP32, FILL_V2_TPL_DOUBLE,
                                                     FILL_V2_TPL_INT8, FILL_V2_TPL_INT16, FILL_V2_TPL_INT32,
                                                     FILL_V2_TPL_INT64));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_FP16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_FP32)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_DOUBLE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_INT8)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_INT16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_INT32)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(dType, FILL_V2_TPL_INT64)), );
#endif
