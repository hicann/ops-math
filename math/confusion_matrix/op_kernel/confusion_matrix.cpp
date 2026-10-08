/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confusion_matrix.cpp
 * \brief The kernel function of confusion_matrix and the entrance for using confusion_matrix.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/confusion_matrix_tiling_data.h"
#include "arch35/confusion_matrix_tiling_key.h"
#include "arch35/confusion_matrix_simt_base.h"
#include "arch35/confusion_matrix_simt_full_load.h"
#include "arch35/confusion_matrix_simt_not_full_load_ub.h"
#include "arch35/confusion_matrix_simt_not_full_load_gm.h"
#include "arch35/confusion_matrix_determine.h"

using namespace AscendC;

#define TPL_SCH_ID_FULL_LOAD 0     // full load to UB
#define TPL_SCH_ID_BATCH_LOAD 1    // batch load to UB
#define TPL_SCH_ID_NOT_FULL_LOAD 2 // not load to UB
#define TPL_SCH_ID_DETERMINE 3     // determine

#define TPL_OUTPUT_DTYPE_FLOAT 0
#define TPL_OUTPUT_DTYPE_INT32 1
#define TPL_OUTPUT_DTYPE_FLOAT16 2
#define TPL_OUTPUT_DTYPE_INT8 3
#define TPL_OUTPUT_DTYPE_UINT8 4

#define TPL_EMPTY_WEIGHT 0
#define TPL_HAS_WEIGHT 1

template <uint64_t schId, uint64_t outputDtype, uint64_t isWeight>
__global__ __aicore__ void confusion_matrix(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y,
                                            GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    using LABEL_TYPE = DTYPE_LABELS;

    AscendC::TPipe tPipe;

    REGISTER_TILING_DEFAULT(ConfusionMatrixTilingData);
    GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingDataIn, tiling);
    const ConfusionMatrixTilingData* __restrict tilingData = &tilingDataIn;

    bool isWeightEmpty = isWeight == TPL_EMPTY_WEIGHT ? true : false;

    // For int8/uint8 output: only DETERMINE mode is supported.
    // asc_atomic_add does not support int8_t/uint8_t, so FULL_LOAD,
    // BATCH_LOAD, and NOT_FULL_LOAD (which rely on asc_atomic_add) cannot
    // be used. Tiling forces DETERMINE (schId=3) for these types.
    // DETERMINE mode uses direct += on disjoint output ranges per core,
    // so no atomic operations are needed and int8_t/uint8_t work correctly.
    if constexpr (outputDtype == TPL_OUTPUT_DTYPE_INT8 || outputDtype == TPL_OUTPUT_DTYPE_UINT8) {
        if (schId == TPL_SCH_ID_DETERMINE) {
            if (outputDtype == TPL_OUTPUT_DTYPE_INT8) {
                ConfusionMatrixSimt::ConfusionMatrixDetermine<LABEL_TYPE, int8_t> op;
                op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
                op.Process();
            } else {
                ConfusionMatrixSimt::ConfusionMatrixDetermine<LABEL_TYPE, uint8_t> op;
                op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
                op.Process();
            }
        }
    } else {
        // FULL_LOAD
        if (schId == TPL_SCH_ID_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
            ConfusionMatrixSimt::ConfusionMatrixSimtFullLoad<LABEL_TYPE, float> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_INT32) {
            ConfusionMatrixSimt::ConfusionMatrixSimtFullLoad<LABEL_TYPE, int32_t> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT16) {
            ConfusionMatrixSimt::ConfusionMatrixSimtFullLoad<LABEL_TYPE, half> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }

        // BATCH_LOAD
        if (schId == TPL_SCH_ID_BATCH_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
            ConfusionMatrixSimt::ConfusionMatrixSimtBatchLoad<LABEL_TYPE, float> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_BATCH_LOAD && outputDtype == TPL_OUTPUT_DTYPE_INT32) {
            ConfusionMatrixSimt::ConfusionMatrixSimtBatchLoad<LABEL_TYPE, int32_t> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_BATCH_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT16) {
            ConfusionMatrixSimt::ConfusionMatrixSimtBatchLoad<LABEL_TYPE, half> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }

        // NOT_FULL_LOAD (GM)
        if (schId == TPL_SCH_ID_NOT_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
            ConfusionMatrixSimt::ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, float> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_NOT_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_INT32) {
            ConfusionMatrixSimt::ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, int32_t> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_NOT_FULL_LOAD && outputDtype == TPL_OUTPUT_DTYPE_FLOAT16) {
            ConfusionMatrixSimt::ConfusionMatrixSimtNotFullLoadGm<LABEL_TYPE, half> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }

        // DETERMINE
        if (schId == TPL_SCH_ID_DETERMINE && outputDtype == TPL_OUTPUT_DTYPE_FLOAT) {
            ConfusionMatrixSimt::ConfusionMatrixDetermine<LABEL_TYPE, float> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_DETERMINE && outputDtype == TPL_OUTPUT_DTYPE_INT32) {
            ConfusionMatrixSimt::ConfusionMatrixDetermine<LABEL_TYPE, int32_t> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
        if (schId == TPL_SCH_ID_DETERMINE && outputDtype == TPL_OUTPUT_DTYPE_FLOAT16) {
            ConfusionMatrixSimt::ConfusionMatrixDetermine<LABEL_TYPE, half> op;
            op.Init(labels, predictions, weights, y, tilingData, &tPipe, isWeightEmpty);
            op.Process();
        }
    }
}
