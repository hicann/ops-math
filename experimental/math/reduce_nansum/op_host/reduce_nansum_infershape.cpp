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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file reduce_nansum_infershape.cpp
 * \brief ReduceNansum 算子形状推导实现
 *
 * 支持全量归约和任意 axis 归约，axes 通过 Tensor 输入传入。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ReduceNansum(gert::InferShapeContext* context)
{
    // 获取输入形状
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输出形状指针
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取属性
    const auto* attrs = context->GetAttrs();

    bool keepdim = false;
    bool isFullReduce = true;
    std::vector<int64_t> dimList;

    if (attrs != nullptr) {
        // keep_dims 属性（index=0，唯一属性）
        const bool* keepdimPtr = attrs->GetBool(0);
        if (keepdimPtr != nullptr) {
            keepdim = *keepdimPtr;
        }
    }

    // axes 输入（index=1）
    auto axesTensor = context->GetInputTensor(1);
    if (axesTensor != nullptr && axesTensor->GetShapeSize() > 0) {
        isFullReduce = false;
        auto axesDtype = context->GetInputDesc(1)->GetDataType();
        int64_t axesSize = axesTensor->GetShapeSize();
        if (axesDtype == ge::DT_INT64) {
            const int64_t* data = axesTensor->GetData<int64_t>();
            for (int64_t i = 0; i < axesSize; i++) {
                dimList.push_back(data[i]);
            }
        } else {
            const int32_t* data = axesTensor->GetData<int32_t>();
            for (int64_t i = 0; i < axesSize; i++) {
                dimList.push_back(static_cast<int64_t>(data[i]));
            }
        }
    }

    int64_t inputRank = static_cast<int64_t>(inputShape->GetDimNum());

    if (isFullReduce) {
        // dim=None，全量归约
        if (keepdim) {
            // 输出 shape 为全 1，与输入同维数
            outputShape->SetDimNum(inputRank);
            for (int64_t i = 0; i < inputRank; i++) {
                outputShape->SetDim(i, 1);
            }
        } else {
            // 输出为标量 shape={}
            // 标量在框架中表示为 1 维、大小为 1 的 tensor
            outputShape->SetDimNum(1);
            outputShape->SetDim(0, 1);
        }
    } else {
        // 任意轴归约（迭代二完善）
        // 规范化负索引
        std::vector<bool> reduceDims(inputRank, false);
        for (auto d : dimList) {
            if (d < 0) {
                d += inputRank;
            }
            if (d >= 0 && d < inputRank) {
                reduceDims[d] = true;
            }
        }

        if (keepdim) {
            outputShape->SetDimNum(inputRank);
            for (int64_t i = 0; i < inputRank; i++) {
                if (reduceDims[i]) {
                    outputShape->SetDim(i, 1);
                } else {
                    outputShape->SetDim(i, inputShape->GetDim(i));
                }
            }
        } else {
            int64_t outRank = 0;
            for (int64_t i = 0; i < inputRank; i++) {
                if (!reduceDims[i]) {
                    outRank++;
                }
            }
            if (outRank == 0) {
                // 所有轴都被归约，输出标量
                outputShape->SetDimNum(1);
                outputShape->SetDim(0, 1);
            } else {
                outputShape->SetDimNum(outRank);
                int64_t idx = 0;
                for (int64_t i = 0; i < inputRank; i++) {
                    if (!reduceDims[i]) {
                        outputShape->SetDim(idx, inputShape->GetDim(i));
                        idx++;
                    }
                }
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceNansum)
    .InferShape(InferShape4ReduceNansum)
    .InputsDataDependency({1});

} // namespace ops
