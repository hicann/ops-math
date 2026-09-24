/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARALLEL_CONCAT_STRUCT_H
#define PARALLEL_CONCAT_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

#define PARALLELCONCAT_COPY_MODE_SIMT 0
#define PARALLELCONCAT_COPY_MODE_SIMD 1

ASCENDC_TPL_ARGS_DECL(ParallelConcat,
                      ASCENDC_TPL_UINT_DECL(COPY_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                            PARALLELCONCAT_COPY_MODE_SIMT, PARALLELCONCAT_COPY_MODE_SIMD));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(COPY_MODE, ASCENDC_TPL_UI_LIST, PARALLELCONCAT_COPY_MODE_SIMT)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(COPY_MODE, ASCENDC_TPL_UI_LIST, PARALLELCONCAT_COPY_MODE_SIMD)));

#endif
