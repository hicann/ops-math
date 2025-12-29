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
 * \file one_hot_tiling.cpp
 * \brief
 */
#include "one_hot_tiling.h"
#include "one_hot_tiling_arch35.h"
#include "op_host/util/const_util.h"

using namespace ge;

namespace {
// Input, Output and Attr indices
const int64_t INDEX_INPUT_X = 0;
const int64_t INDEX_INPUT_DEPTH = 1;
const int64_t INDEX_INPUT_ON_VALUE = 2;
const int64_t INDEX_INPUT_OFF_VALUE = 3;
const int64_t INDEX_OUTPUT_Y = 0;
const int64_t INDEX_ATTR_AXIS = 0;
const int64_t AXIS_DEFAULT_VALUE = -1;
constexpr int64_t DEPTH_SIZE = 1;
// Calcuted acording to the min size of radio of ub and tensor
int64_t RADIO_UB_SIZE_FIRST = 0;
// Calcuted acording to the max size of radio of ub and tensor
int64_t RADIO_UB_SIZE_SCEOND = 0;
constexpr int64_t UB_PART_COUNT = 3;
constexpr int64_t ONE_PART_UB = 1;
constexpr int64_t TWO_PART_UB = 2;
// default tiling mode
const int64_t TILING_MODE_DEFAULT = 0;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t INT64_BYTES = 8;
constexpr int64_t ALIGN_16 = 16;

// last axis && enough ub size for both input and output
const int64_t TILING_MODE_LAST_AXIS_ENOUGH_SPACE = 1;
// last axis && only enough ub size for input and partial output
const int64_t TILING_MODE_LAST_AXIS_PARTIAL_SPACE = 2;
// last axis && only enough ub size for input
const int64_t TILING_MODE_LAST_AXIS_ENOUGH_INPUT = 3;
// last axis && only enough ub size for partial input and partial output
const int64_t TILING_MODE_LAST_AXIS_PARTIAL_INPUT = 4;
// last axis && no enough ub size for input and output
const int64_t TILING_MODE_LAST_AXIS_NO_SPACE = 5;

// first axis && enough ub size for both input and output
const int64_t TILING_MODE_FIRST_AXIS_ENOUGH_SPACE = 6;
// first axis && only enough ub size for input and partial output
const int64_t TILING_MODE_FIRST_AXIS_PARTIAL_SPACE = 7;
// first axis && only enough ub size for input
const int64_t TILING_MODE_FIRST_AXIS_ENOUGH_INPUT = 8;

// middle axis && enough ub size for both input and output
const int64_t TILING_MODE_MIDDLE_AXIS_ENOUGH_SPACE = 9;
// middle axis && only enough ub size for input and partial output
const int64_t TILING_MODE_MIDDLE_AXIS_PARTIAL_SPACE = 10;
// middle axis && only enough ub size for input
const int64_t TILING_MODE_MIDDLE_AXIS_ENOUGH_INPUT = 11;

bool MergeAxis(int64_t axis, int64_t depth, const gert::Shape& x_shape, std::vector<int64_t>& merged_x_shape)
{
    int64_t x_shape_size = x_shape.GetDimNum();
    int64_t first_dim_size = 1;
    int64_t last_dim_size = 1;
    int64_t i = 0;
    int64_t axis_size = depth;
    if (axis == 0) {
        while (i < x_shape_size) {
            last_dim_size *= x_shape.GetDim(i);
            ++i;
        }
        merged_x_shape.push_back(axis_size);
        merged_x_shape.push_back(last_dim_size);
        return true;
    } else if (axis > 0 && axis < x_shape_size) {
        while (i < axis) {
            first_dim_size *= x_shape.GetDim(i);
            ++i;
        }
        i = axis;
        merged_x_shape.push_back(first_dim_size);
        merged_x_shape.push_back(axis_size);
        while (i < x_shape_size) {
            last_dim_size *= x_shape.GetDim(i);
            ++i;
        }
        merged_x_shape.push_back(last_dim_size);
        return true;
    } else if (axis == -1 || axis == x_shape_size) {
        while (i < x_shape_size) {
            first_dim_size *= x_shape.GetDim(i);
            ++i;
        }
        merged_x_shape.push_back(first_dim_size);
        merged_x_shape.push_back(axis_size);
        return true;
    }
    return false;
}

int64_t CeilDiv(int64_t first_value, int64_t second_value)
{
    OP_CHECK_IF(second_value == 0, OP_LOGE("one_hot", "second_value = 0 is not support"), return -1);
    int64_t result = 1;
    result = (first_value + second_value - 1) / second_value;
    return result;
}

int64_t CalTensorNumel(std::vector<int64_t> tensor_shape)
{
    int64_t numel = std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<int64_t>());
    return numel;
}

int64_t get_core_num(int64_t numel, const gert::Shape& x_shape, int64_t depth, int64_t axis, int64_t core_num)
{
    OP_CHECK_IF(core_num == 0, OP_LOGE("one_hot", "core_num = 0 is not support"), return -1);
    auto ele_per_core = (numel - 1) / core_num + 1;
    auto core_used = (numel - 1) / ele_per_core + 1;
    auto numel_x = x_shape.GetShapeSize();
    std::vector<int64_t> merged_x_shape;
    MergeAxis(axis, depth, x_shape, merged_x_shape);
    int64_t block = 16;
    int64_t x_shape_size = static_cast<int64_t>(x_shape.GetDimNum());

    if (axis == 0) {
        auto per_core_index = CeilDiv(depth, core_used);
        while ((per_core_index * numel_x < block) && (core_used > 1)) {
            core_used -= 1;
            per_core_index = CeilDiv(depth, core_used);
        }
        core_used = CeilDiv(depth, per_core_index);
    } else if (axis == x_shape_size || axis == -1) {
        auto per_core_numel = CeilDiv(numel_x, core_used);
        while (per_core_numel * depth < block && core_used > 1) {
            core_used -= 1;
            per_core_numel = CeilDiv(numel_x, core_used);
        }
        core_used = CeilDiv(numel_x, per_core_numel);
    } else {
        auto first_dim_x = merged_x_shape[0];
        auto last_dim_x = merged_x_shape[2];
        auto per_core_numel = CeilDiv(first_dim_x, core_used);
        while (per_core_numel * last_dim_x * depth < block && core_used > 1) {
            core_used -= 1;
            per_core_numel = CeilDiv(first_dim_x, core_used);
        }
        core_used = CeilDiv(first_dim_x, per_core_numel);
    }
    return core_used;
}

int64_t CalTilingMode(const gert::Shape& x_shape, int64_t depth, int64_t axis, int64_t core_num)
{
    int64_t x_shape_size = static_cast<int64_t>(x_shape.GetDimNum());
    int64_t x_numel = static_cast<int64_t>(x_shape.GetShapeSize());
    int64_t tiling_mode = TILING_MODE_DEFAULT;

    if (axis == -1 || axis == x_shape_size) {
        auto core_used = get_core_num(x_numel, x_shape, depth, axis, core_num);
        auto per_core_numel = CeilDiv(x_numel, core_used);
        if (per_core_numel <= RADIO_UB_SIZE_FIRST && per_core_numel * depth <= RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_LAST_AXIS_ENOUGH_SPACE;
        } else if (
            per_core_numel <= RADIO_UB_SIZE_FIRST && per_core_numel * depth > RADIO_UB_SIZE_SCEOND &&
            depth >= DEPTH_SIZE && depth <= RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_LAST_AXIS_PARTIAL_SPACE;
        } else if (per_core_numel <= RADIO_UB_SIZE_FIRST && depth > RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_LAST_AXIS_ENOUGH_INPUT;
        } else if (per_core_numel > RADIO_UB_SIZE_FIRST && depth >= DEPTH_SIZE && depth <= RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_LAST_AXIS_PARTIAL_INPUT;
        } else {
            tiling_mode = TILING_MODE_LAST_AXIS_NO_SPACE;
        }
    } else if (axis == 0) {
        if (x_numel <= RADIO_UB_SIZE_FIRST && x_numel * depth <= RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_FIRST_AXIS_ENOUGH_SPACE;
        } else if (x_numel <= RADIO_UB_SIZE_FIRST && x_numel * depth > RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_FIRST_AXIS_PARTIAL_SPACE;
        } else if (x_numel > RADIO_UB_SIZE_FIRST && x_numel * depth > RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_FIRST_AXIS_ENOUGH_INPUT;
        }
    } else if (axis < x_shape_size && axis > 0) {
        std::vector<int64_t> merged_x_shape;
        MergeAxis(axis, depth, x_shape, merged_x_shape);
        auto first_dim_x = merged_x_shape[0];
        auto core_used = get_core_num(first_dim_x, x_shape, depth, axis, core_num);
        auto per_core_numel = CeilDiv(first_dim_x, core_used);
        auto last_dim_x = merged_x_shape[2];
        if (per_core_numel * last_dim_x <= RADIO_UB_SIZE_FIRST &&
            per_core_numel * last_dim_x * depth <= RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_MIDDLE_AXIS_ENOUGH_SPACE;
        } else if (
            per_core_numel * last_dim_x <= RADIO_UB_SIZE_FIRST &&
            per_core_numel * last_dim_x * depth > RADIO_UB_SIZE_SCEOND && last_dim_x >= 1 &&
            last_dim_x <= RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_MIDDLE_AXIS_PARTIAL_SPACE;
        } else if (
            per_core_numel * last_dim_x > RADIO_UB_SIZE_FIRST &&
            per_core_numel * last_dim_x * depth > RADIO_UB_SIZE_SCEOND) {
            tiling_mode = TILING_MODE_MIDDLE_AXIS_ENOUGH_INPUT;
        }
    }

    return tiling_mode;
}

void CalCoreInfo(
    optiling::OneHotTilingParams* tiling_params, int64_t core_num, int64_t depth, int64_t axis,
    const gert::Shape& x_shape)
{
    std::vector<int64_t> merged_x_shape;
    MergeAxis(axis, depth, x_shape, merged_x_shape);

    int64_t per_core_index = 0;
    int64_t last_core_index = 0;
    int64_t core_used = 0;
    int64_t per_core_numel = 0;
    int64_t last_core_numel = 0;
    int64_t x_shape_size = static_cast<int64_t>(x_shape.GetDimNum());
    int64_t numel_x = static_cast<int64_t>(x_shape.GetShapeSize());

    if (axis == 0) {
        core_used = get_core_num(depth, x_shape, depth, axis, core_num);
        per_core_index = CeilDiv(depth, core_used);
        last_core_index = depth - (core_used - 1) * per_core_index;
    } else if (axis == -1 || axis == x_shape_size) {
        core_used = get_core_num(numel_x, x_shape, depth, axis, core_num);
        per_core_numel = CeilDiv(numel_x, core_used);
        last_core_numel = numel_x - (core_used - 1) * per_core_numel;
    } else {
        auto first_dim_x = merged_x_shape[0];
        core_used = get_core_num(first_dim_x, x_shape, depth, axis, core_num);
        per_core_numel = CeilDiv(first_dim_x, core_used);
        last_core_numel = first_dim_x - (core_used - 1) * per_core_numel;
    }

    tiling_params->last_core_index = last_core_index;
    tiling_params->not_last_core_index = per_core_index;
    tiling_params->not_last_core_numel = per_core_numel;
    tiling_params->core_used = core_used;
    tiling_params->last_core_numel = last_core_numel;
}

void CalRunningInfo(
    optiling::OneHotTilingParams* tiling_params, int64_t core_num, int64_t depth, int64_t axis,
    const gert::Shape& x_shape)
{
    std::vector<int64_t> merged_x_shape;
    MergeAxis(axis, depth, x_shape, merged_x_shape);

    int64_t first_dim_x = 1;
    int64_t last_dim_x = 1;
    int64_t numel_x = 1;
    int64_t numel_merged_x = 1;
    int64_t x_shape_size = static_cast<int64_t>(x_shape.GetDimNum());

    numel_x = static_cast<int64_t>(x_shape.GetShapeSize());
    numel_merged_x = CalTensorNumel(merged_x_shape);
    tiling_params->numel_shape_x = numel_x;
    if (axis > 0 && axis < x_shape_size) {
        first_dim_x = merged_x_shape[0];
        last_dim_x = merged_x_shape[2]; // 2 means the last one dimension.
    } else if (axis == x_shape_size || axis == -1) {
        first_dim_x = merged_x_shape[0];
    } else {
        last_dim_x = merged_x_shape[1];
    }
    tiling_params->first_dim_x = first_dim_x;
    tiling_params->last_dim_x = last_dim_x;
    tiling_params->numel_shape_off_value_tensor = numel_merged_x;
    tiling_params->tiling_core_num = core_num;
    tiling_params->mode_of_cal_with_axis = CalTilingMode(x_shape, depth, axis, core_num);
    CalCoreInfo(tiling_params, core_num, depth, axis, x_shape);
}

void PrintTilingParams(const gert::TilingContext* context, const optiling::OneHotTilingParams* tiling_params)
{
    OP_LOGD(context->GetNodeName(), "PrintTilingParams is running");
    OP_LOGD(context->GetNodeName(), "is_zero_off_value=%ld.", tiling_params->is_zero_off_value);
    OP_LOGD(context->GetNodeName(), "not_last_core_numel=%ld.", tiling_params->not_last_core_numel);
    OP_LOGD(context->GetNodeName(), "mode_of_cal_with_axis=%ld.", tiling_params->mode_of_cal_with_axis);
    OP_LOGD(context->GetNodeName(), "core_used=%ld.", tiling_params->core_used);
    OP_LOGD(context->GetNodeName(), "numel_shape_x=%ld.", tiling_params->numel_shape_x);
    OP_LOGD(context->GetNodeName(), "first_dim_x=%ld.", tiling_params->first_dim_x);
    OP_LOGD(context->GetNodeName(), "last_dim_x=%ld.", tiling_params->last_dim_x);
    OP_LOGD(context->GetNodeName(), "numel_shape_off_value_tensor=%ld.", tiling_params->numel_shape_off_value_tensor);
    OP_LOGD(context->GetNodeName(), "last_core_numel=%ld.", tiling_params->last_core_numel);
    OP_LOGD(context->GetNodeName(), "not_last_core_index=%ld.", tiling_params->not_last_core_index);
    OP_LOGD(context->GetNodeName(), "last_core_index=%ld.", tiling_params->last_core_index);
    OP_LOGD(context->GetNodeName(), "tiling_core_num=%ld.", tiling_params->tiling_core_num);
}
} // namespace

namespace optiling {
static ge::graphStatus OneHotTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "OneHotTiling is running");
    const OneHotCompileInfo* compile_info = reinterpret_cast<const OneHotCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return OneHotTilingForAscendC(context);
}

static ge::graphStatus TilingPrepare4OneHotForAscendc(gert::TilingParseContext* context, OneHotCompileInfo* compileInfo)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepare4OneHotForAscendc.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->core_num <= 0), OP_LOGE(context->GetNodeName(), "core num invalid."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ub_size <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4OneHot(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForOneHot running.");
    auto compile_info = context->GetCompiledInfo<OneHotCompileInfo>();
    return TilingPrepare4OneHotForAscendc(context, compile_info);
}

IMPL_OP_OPTILING(OneHot)
    .Tiling(OneHotTiling)
    .TilingInputsDataDependency({ONEHOT_INPUT_DEPENDENCY_IDX})
    .TilingParse<OneHotCompileInfo>(TilingPrepare4OneHot);
} // namespace optiling
