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
 * \file concat_infershape.cpp
 * \brief
 */
#include "concat_infershape.h"
#include "log/log.h"
#include "op_api/op_util.h"

using namespace ge;
namespace ops {

ge::graphStatus ConcatInferShapeCommon(gert::InferShapeContext* context, const int64_t dynamic_input_idx,
                                       int64_t num_concat, int64_t axis)
{
    constexpr int64_t kUnknownDim = -1;
    auto in_shape = context->GetDynamicInputShape(dynamic_input_idx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    OP_LOGD(context->GetNodeName(), "input_shape 0:%s", Ops::Base::ToString(*in_shape).c_str());
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;

    if (num_concat == 1) {
        // dynamic case or the input only one will use dynamic infer func
        return ge::GRAPH_SUCCESS;
    }

    bool has_unknown_rank_input = false;
    const gert::Shape* base_shape = nullptr;
    int64_t base_index = 0;
    for (int64_t relative_index = 0; relative_index < num_concat; relative_index++) {
        const gert::Shape* shape_i = context->GetDynamicInputShape(dynamic_input_idx, relative_index);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape_i);
        if (Ops::Base::IsUnknownRank(*shape_i)) {
            has_unknown_rank_input = true;
        } else if (base_shape == nullptr) {
            base_shape = shape_i;
            base_index = relative_index;
        }
    }
    if (base_shape == nullptr) {
        Ops::Base::SetUnknownRank(*out_shape);
        OP_LOGD(context->GetNodeName(), "output_shape:%s", Ops::Base::ToString(*out_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }
    *out_shape = *base_shape;

    if (out_shape->IsScalar()) {
        // scalar to shape [1]
        out_shape->AppendDim(1);
    }

    size_t output_dim = out_shape->GetDimNum();
    OP_CHECK_IF(!IsDimValid(output_dim, axis),
                OP_LOGE(context->GetNodeName(), "%s", GenInvalidDimMsg("concat_dim", output_dim, axis).c_str()),
                return ge::GRAPH_FAILED);
    if (axis < 0) {
        axis += output_dim;
    }

    bool concat_dim_unknown = has_unknown_rank_input || (out_shape->GetDim(axis) == kUnknownDim);
    int64_t concat_dim_size = 0;
    if (!concat_dim_unknown) {
        concat_dim_size = out_shape->GetDim(axis);
    }
    for (int64_t relative_index = 1; relative_index < num_concat; relative_index++) {
        const gert::Shape* input_i_shape = context->GetDynamicInputShape(dynamic_input_idx, relative_index);
        OP_CHECK_NULL_WITH_CONTEXT(context, input_i_shape);
        OP_LOGD(context->GetNodeName(), "input_shape %ld:%s", relative_index,
                Ops::Base::ToString(*input_i_shape).c_str());
        if (relative_index == base_index || Ops::Base::IsUnknownRank(*input_i_shape)) {
            continue;
        }
        if (input_i_shape->IsScalar() && output_dim == 1) {
            concat_dim_size += 1;
            continue;
        }
        if (input_i_shape->GetDimNum() != output_dim) {
            // input shape size is not equal output
            OP_LOGE(context->GetNodeName(), "shape[%" PRId64 "].GetDimNum %zu, must be equal to %zu!", relative_index,
                    input_i_shape->GetDimNum(), output_dim);
            return ge::GRAPH_FAILED;
        }
        // check whether the non concat dim is equal, unknown dim is treated as wildcard
        for (int64_t check_dim = 0; check_dim < static_cast<int64_t>(output_dim); check_dim++) {
            if (check_dim == axis) {
                continue;
            }
            const int64_t input_dim = input_i_shape->GetDim(check_dim);
            const int64_t output_dim_value = out_shape->GetDim(check_dim);
            if (input_dim != kUnknownDim && output_dim_value != kUnknownDim && input_dim != output_dim_value) {
                OP_LOGE(context->GetNodeName(),
                        "shape[%" PRId64 "][%" PRId64 "] is %" PRId64 ", must be equal to %" PRId64 "!", relative_index,
                        check_dim, input_dim, output_dim_value);
                return ge::GRAPH_FAILED;
            }
            if (output_dim_value == kUnknownDim && input_dim != kUnknownDim) {
                out_shape->SetDim(check_dim, input_dim);
            }
        }
        const int64_t input_axis_dim = input_i_shape->GetDim(axis);
        if (input_axis_dim == kUnknownDim) {
            concat_dim_unknown = true;
        } else {
            concat_dim_size += input_axis_dim;
        }
    }
    out_shape->SetDim(axis, concat_dim_unknown ? kUnknownDim : concat_dim_size);
    OP_LOGD(context->GetNodeName(), "output_shape:%s", Ops::Base::ToString(*out_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static bool GetConcatDim(gert::InferShapeContext* context, int64_t dimIdx, int64_t& concatDim)
{
    // 注意: 此处必须按 IR anchor 索引取 tensor(GetRequiredInputTensor), 不能用扁平索引(GetInputTensor),
    // 因为 x 为动态输入, concat_dim 的扁平位置随实例数变化
    auto concatDimTensor = context->GetRequiredInputTensor(dimIdx);
    if (concatDimTensor == nullptr) {
        return false;
    }
    const T* concatDimValPtr = concatDimTensor->GetData<T>();
    if (concatDimValPtr == nullptr) {
        // 编译期轴值不可得(Data-feed): 由调用方兜底, 不在此处报错
        return false;
    }
    concatDim = static_cast<int64_t>(concatDimValPtr[0]);
    return true;
}

ge::graphStatus InferShapeForConcatAndConcatV2(gert::InferShapeContext* context, int64_t inputIdx, int64_t dimIdx)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    int64_t inputNum = static_cast<int64_t>(anchorInstanceInfo->GetInstanceNum());
    auto concatDimPtr = context->GetRequiredInputDesc(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimPtr);
    ge::DataType concatDimType = concatDimPtr->GetDataType();
    int64_t concatDim = 0;
    bool dimAvailable = false;
    if (concatDimType == ge::DT_INT32) {
        dimAvailable = GetConcatDim<int32_t>(context, dimIdx, concatDim);
    } else {
        dimAvailable = GetConcatDim<int64_t>(context, dimIdx, concatDim);
    }
    if (!dimAvailable) {
        // 轴值编译期不可得(Data-feed): 秩一致性为轴无关约束, 编译期即校验;
        // 秩已知时输出保秩全-1, 否则输出未知秩, 交由运行期携带真实数据重推
        OP_LOGW(context->GetNodeName(), "concat_dim value unavailable at compile time, set output to unknown shape.");
        auto outShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        int64_t knownRank = -1;
        for (int64_t relativeIndex = 0; relativeIndex < inputNum; relativeIndex++) {
            const gert::Shape* inputShape = context->GetDynamicInputShape(inputIdx, relativeIndex);
            OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
            if (Ops::Base::IsUnknownRank(*inputShape)) {
                continue;
            }
            const int64_t dimNum = static_cast<int64_t>(inputShape->GetDimNum());
            if (knownRank < 0) {
                knownRank = dimNum;
            } else if (dimNum != knownRank) {
                OP_LOGE(context->GetNodeName(), "shape[%" PRId64 "] rank is %" PRId64 ", must be equal to %" PRId64 "!",
                        relativeIndex, dimNum, knownRank);
                return ge::GRAPH_FAILED;
            }
        }
        if (knownRank < 0) {
            // 所有输入均为未知秩, 输出只能为未知秩
            Ops::Base::SetUnknownRank(*outShape);
        } else {
            Ops::Base::SetUnknownShape(knownRank, *outShape);
        }
        return ge::GRAPH_SUCCESS;
    }

    return ConcatInferShapeCommon(context, inputIdx, inputNum, concatDim);
}

static ge::graphStatus InferShape4Concat(gert::InferShapeContext* context)
{
    return InferShapeForConcatAndConcatV2(context, INPUT_IDX, INDEX_CONCAT_DIM);
}

IMPL_OP_INFERSHAPE(Concat).InferShape(InferShape4Concat).InputsDataDependency({INDEX_CONCAT_DIM});
} // namespace ops
