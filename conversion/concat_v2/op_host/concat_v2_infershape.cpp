/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_CONCAT_DIM_FOR_CONCAT_V2 = 1;
constexpr size_t INPUT_IDX_FOR_CONCAT_V2 = 0;

template <typename T>
static inline bool GetAxisValue(gert::InferShapeContext* context, int64_t dimIdx, int64_t& axisValue)
{
    // 注意: 此处必须按 IR anchor 索引取 tensor(GetRequiredInputTensor), 不能用扁平索引(GetInputTensor),
    // 因为 x 为动态输入, concat_dim 的扁平位置随实例数变化
    auto tensor = context->GetRequiredInputTensor(dimIdx);
    if (tensor == nullptr) {
        return false;
    }
    auto ptr = tensor->GetData<T>();
    if (ptr == nullptr) {
        // 编译期轴值不可得(Data-feed): 由调用方兜底, 不在此处报错
        return false;
    }
    axisValue = static_cast<int64_t>(ptr[0]);
    return true;
}

static ge::graphStatus InferShape4ConcatV2(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4ConcatV2 Func");
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);

    auto xInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_IDX_FOR_CONCAT_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInfo);

    int64_t numInputs = static_cast<int64_t>(xInfo->GetInstanceNum());
    OP_LOGW(context, "ConcatV2 Dynamic input count = %ld", numInputs);
    if (numInputs <= 1) {
        OP_LOGE(context, "ConcatV2 requires at least 2 inputs");
        return ge::GRAPH_FAILED;
    }

    auto concatDimPtr = context->GetRequiredInputDesc(INDEX_CONCAT_DIM_FOR_CONCAT_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimPtr);

    auto dtype = concatDimPtr->GetDataType();
    int64_t axis = 0;
    bool axisAvailable = false;
    if (dtype == ge::DT_INT32) {
        axisAvailable = GetAxisValue<int32_t>(context, INDEX_CONCAT_DIM_FOR_CONCAT_V2, axis);
    } else if (dtype == ge::DT_INT64) {
        axisAvailable = GetAxisValue<int64_t>(context, INDEX_CONCAT_DIM_FOR_CONCAT_V2, axis);
    } else {
        OP_LOGE(context, "ConcatV2: unsupported concat_dim dtype %s", Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }
    if (!axisAvailable) {
        // 轴值编译期不可得(Data-feed): 与V1一致, 秩已知时输出保秩全-1, 否则输出未知秩, 交由运行期携带真实数据重推
        OP_LOGW(context, "concat_dim value unavailable at compile time, set output to unknown shape.");
        auto outShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        int64_t knownRank = -1;
        for (int64_t i = 0; i < numInputs; ++i) {
            auto shapeI = context->GetDynamicInputShape(INPUT_IDX_FOR_CONCAT_V2, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, shapeI);
            if (!Ops::Base::IsUnknownRank(*shapeI)) {
                const int64_t dimNum = static_cast<int64_t>(shapeI->GetDimNum());
                if (dimNum > knownRank) {
                    knownRank = dimNum;
                }
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

    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto firstShape = context->GetDynamicInputShape(INPUT_IDX_FOR_CONCAT_V2, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, firstShape);

    *outShape = *firstShape;

    size_t rank = outShape->GetDimNum();
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        OP_LOGE(context, "Invalid concat_dim=%ld, rank=%zu.", axis, rank);
        return ge::GRAPH_FAILED;
    }

    constexpr int64_t kUnknownDim = -1;
    bool dim_unknown = outShape->GetDim(axis) == kUnknownDim;
    int64_t newDim = 0;
    if (!dim_unknown) {
        newDim = outShape->GetDim(axis);
    }
    for (int64_t i = 1; i < numInputs; ++i) {
        auto shape_i = context->GetDynamicInputShape(INPUT_IDX_FOR_CONCAT_V2, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape_i);
        const int64_t dimValue = shape_i->GetDim(axis);
        if (dimValue == kUnknownDim) {
            dim_unknown = true;
        } else {
            newDim += dimValue;
        }
    }

    outShape->SetDim(axis, dim_unknown ? kUnknownDim : newDim);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ConcatV2).InferShape(InferShape4ConcatV2).InputsDataDependency({INDEX_CONCAT_DIM_FOR_CONCAT_V2});
} // namespace ops
