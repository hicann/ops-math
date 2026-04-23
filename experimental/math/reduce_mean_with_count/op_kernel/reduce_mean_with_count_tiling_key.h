/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file reduce_mean_with_count_tiling_key.h
 * @brief ReduceMeanWithCount TilingKey definitions
 *
 * Template parameters:
 *   - D_T_X: data type (C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16)
 *   - REDUCE_MODE: 0=AR full-load, 1=AR col-split, 2=ARA full-load
 */

#ifndef __REDUCE_MEAN_WITH_COUNT_TILING_KEY_H__
#define __REDUCE_MEAN_WITH_COUNT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// REDUCE_MODE constants
#define REDUCE_MODE_AR_FULLLOAD  0
#define REDUCE_MODE_AR_COLSPLIT  1
#define REDUCE_MODE_ARA_FULLLOAD 2

ASCENDC_TPL_ARGS_DECL(ReduceMeanWithCount,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(REDUCE_MODE, 8, ASCENDC_TPL_UI_LIST,
                          REDUCE_MODE_AR_FULLLOAD,
                          REDUCE_MODE_AR_COLSPLIT,
                          REDUCE_MODE_ARA_FULLLOAD)
);

ASCENDC_TPL_SEL(
    // FP32
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(REDUCE_MODE, ASCENDC_TPL_UI_LIST,
                             REDUCE_MODE_AR_FULLLOAD, REDUCE_MODE_AR_COLSPLIT, REDUCE_MODE_ARA_FULLLOAD)
    ),
    // FP16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(REDUCE_MODE, ASCENDC_TPL_UI_LIST,
                             REDUCE_MODE_AR_FULLLOAD, REDUCE_MODE_AR_COLSPLIT, REDUCE_MODE_ARA_FULLLOAD)
    ),
    // BF16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(REDUCE_MODE, ASCENDC_TPL_UI_LIST,
                             REDUCE_MODE_AR_FULLLOAD, REDUCE_MODE_AR_COLSPLIT, REDUCE_MODE_ARA_FULLLOAD)
    ),
);

#endif // __REDUCE_MEAN_WITH_COUNT_TILING_KEY_H__
