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
 * \file floor_mod_tiling_key.h
 * \brief floor_mod tiling key declare
 */
#ifndef FLOOR_MOD_TILING_KEY_H
#define FLOOR_MOD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace FloorModNs {

#define FLOOR_MOD_TPL_BF16 10
#define FLOOR_MOD_TPL_FP16 20
#define FLOOR_MOD_TPL_FP32 30
#define FLOOR_MOD_TPL_INT32 40
#define FLOOR_MOD_TPL_INT64 50
#define FLOOR_MOD_TPL_DOUBLE 60

#define FLOOR_MOD_TPL_PATH_GENERAL 0
#define FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH 1
#define FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH 2
#define FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST 3
#define FLOOR_MOD_TPL_PATH_SMALL_DENSE 4
#define FLOOR_MOD_TPL_PATH_FP64_STORAGE 5

ASCENDC_TPL_ARGS_DECL(FloorMod,
                      ASCENDC_TPL_DTYPE_DECL(D_T_X1, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT64, FLOOR_MOD_TPL_FP16,
                                             FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_DOUBLE),
                      ASCENDC_TPL_DTYPE_DECL(D_T_X2, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT64, FLOOR_MOD_TPL_FP16,
                                             FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_DOUBLE),
                      ASCENDC_TPL_DTYPE_DECL(D_T_Y, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT64, FLOOR_MOD_TPL_FP16,
                                             FLOOR_MOD_TPL_BF16, FLOOR_MOD_TPL_FP32, FLOOR_MOD_TPL_DOUBLE),
                      ASCENDC_TPL_UINT_DECL(KERNEL_PATH, 6, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_GENERAL,
                                            FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH, FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH,
                                            FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST, FLOOR_MOD_TPL_PATH_SMALL_DENSE,
                                            FLOOR_MOD_TPL_PATH_FP64_STORAGE), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X1, FLOOR_MOD_TPL_INT32),
                         ASCENDC_TPL_DTYPE_SEL(D_T_X2, FLOOR_MOD_TPL_INT32),
                         ASCENDC_TPL_DTYPE_SEL(D_T_Y, FLOOR_MOD_TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_GENERAL,
                                              FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH, FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH,
                                              FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST, FLOOR_MOD_TPL_PATH_SMALL_DENSE), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X1, FLOOR_MOD_TPL_INT64),
                         ASCENDC_TPL_DTYPE_SEL(D_T_X2, FLOOR_MOD_TPL_INT64),
                         ASCENDC_TPL_DTYPE_SEL(D_T_Y, FLOOR_MOD_TPL_INT64),
                         ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_GENERAL,
                                              FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST, FLOOR_MOD_TPL_PATH_SMALL_DENSE), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X1, FLOOR_MOD_TPL_FP16),
                         ASCENDC_TPL_DTYPE_SEL(D_T_X2, FLOOR_MOD_TPL_FP16),
                         ASCENDC_TPL_DTYPE_SEL(D_T_Y, FLOOR_MOD_TPL_FP16),
                         ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_GENERAL,
                                              FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH, FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH,
                                              FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST, FLOOR_MOD_TPL_PATH_SMALL_DENSE), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X1, FLOOR_MOD_TPL_FP32),
                         ASCENDC_TPL_DTYPE_SEL(D_T_X2, FLOOR_MOD_TPL_FP32),
                         ASCENDC_TPL_DTYPE_SEL(D_T_Y, FLOOR_MOD_TPL_FP32),
                         ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_GENERAL,
                                              FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH, FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH,
                                              FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST, FLOOR_MOD_TPL_PATH_SMALL_DENSE), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X1, FLOOR_MOD_TPL_BF16),
                         ASCENDC_TPL_DTYPE_SEL(D_T_X2, FLOOR_MOD_TPL_BF16),
                         ASCENDC_TPL_DTYPE_SEL(D_T_Y, FLOOR_MOD_TPL_BF16),
                         ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_GENERAL,
                                              FLOOR_MOD_TPL_PATH_DENSE_TAIL_BATCH, FLOOR_MOD_TPL_PATH_COMPACT_ROW_BATCH,
                                              FLOOR_MOD_TPL_PATH_SCALAR_BROADCAST, FLOOR_MOD_TPL_PATH_SMALL_DENSE), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X1, FLOOR_MOD_TPL_DOUBLE),
                         ASCENDC_TPL_DTYPE_SEL(D_T_X2, FLOOR_MOD_TPL_DOUBLE),
                         ASCENDC_TPL_DTYPE_SEL(D_T_Y, FLOOR_MOD_TPL_DOUBLE),
                         ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, FLOOR_MOD_TPL_PATH_FP64_STORAGE), ));

} // namespace FloorModNs

#endif // FLOOR_MOD_TILING_KEY_H
