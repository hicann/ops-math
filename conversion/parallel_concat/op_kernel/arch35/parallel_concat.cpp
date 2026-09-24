/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h" // dynamic input address table (ListTensorDesc)
#include "parallel_concat_tiling_struct.h"    // shared TilingData POD + routing threshold
#include "parallel_concat_struct.h"           // ASCENDC_TPL template parameter declarations
#include "parallel_concat_kernel_simt.h"      // tilingKey=0 engine
#include "parallel_concat_kernel_simd.h"      // tilingKey=1 engine

template <int COPY_MODE>
__global__ __aicore__ void parallel_concat(GM_ADDR values, GM_ADDR output_data, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TilingData is not TPL-templated: both branches share the 8-field POD.
    REGISTER_TILING_DEFAULT(ParallelConcatTilingData);
    GET_TILING_DATA_WITH_STRUCT(ParallelConcatTilingData, tilingData, tiling);

    // Empty-tensor early return (any tail dim d_m=0 -> totalBytes=0): output
    // [N, 0] has no byte to write; takes priority over the template Process.
    if (tilingData.totalBytes == 0) {
        return;
    }

    if constexpr (COPY_MODE == PARALLELCONCAT_COPY_MODE_SIMT) {
        AscendC::InitSocState();
        ParallelConcatKernelSimt kernel;
        kernel.Init(values, output_data, &tilingData); // Init holds the core guard
        kernel.Process();
    } else {
        AscendC::TPipe pipe;
        ParallelConcatKernelSimd kernel;
        kernel.Init(values, output_data, &tilingData, &pipe); // Init holds the core guard
        kernel.Process();
    }
}
