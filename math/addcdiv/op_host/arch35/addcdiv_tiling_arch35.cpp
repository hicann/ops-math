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
 * \file addcdiv_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/addcdiv/op_kernel/arch35/addcdiv_dag.h"
#include "math/addcdiv/op_kernel/arch35/addcdiv_struct.h"
#include "addcdiv_tiling_arch35.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t ADDCDIV_COMMON_TILING_PRIORITY = 0;

ge::graphStatus AddcdivTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool AddcdivTiling::IsCapable()
{
    return true;
}

bool AddcdivTiling::IsMixedDtype(const ge::DataType& inputDataDtype, const ge::DataType& inputValDtype) const
{
    return (inputDataDtype == ge::DT_FLOAT16 && inputValDtype == ge::DT_FLOAT) ||
           (inputDataDtype == ge::DT_BF16 && inputValDtype == ge::DT_FLOAT);
}

bool AddcdivTiling::CheckDtype(
    const ge::DataType& inputDataDtype, const ge::DataType& inputX1Dtype, const ge::DataType& inputX2Dtype,
    const ge::DataType& inputValDtype, const ge::DataType& outputDtype, const bool isMixedDtype) const
{
    auto nodeName = context_->GetNodeName();
    if (!isMixedDtype && (inputDataDtype != inputX1Dtype || inputDataDtype != inputX2Dtype ||
                          inputDataDtype != inputValDtype || inputDataDtype != outputDtype)) {
        OP_LOGE(
            nodeName, "Dtype of inputdata[%s] should be equal to dtype of inputx1[%s] and inputx2[%s] and output[%s]",
            ge::TypeUtils::DataTypeToSerialString(inputDataDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputX1Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputX2Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str());
        return false;
    }
    if (isMixedDtype && inputValDtype != ge::DT_FLOAT) {
        OP_LOGE(
            nodeName, "Dtype of inputvalueDtype[%s] should be fp32 when input dtypes is mixed.",
            ge::TypeUtils::DataTypeToSerialString(inputValDtype).c_str());
        return false;
    }
    return true;
}

ge::graphStatus AddcdivTiling::DoOpTiling()
{
    // 获取输入
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    auto input2Desc = context_->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input2Desc);
    ge::DataType input2Dtype = input2Desc->GetDataType();
    auto input3Desc = context_->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input3Desc);
    ge::DataType input3Dtype = input3Desc->GetDataType();
    // 获取输出
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    // 数据类型校验
    auto isMixedDtype = IsMixedDtype(input0Dtype, input3Dtype);
    if (!CheckDtype(input0Dtype, input1Dtype, input2Dtype, input3Dtype, outputDtype, isMixedDtype)) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (isMixedDtype) {
        Ops::Base::BroadcastBaseTiling<AddcdivOp::AddcdivOp<Ops::Base::half, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT16 || input0Dtype == ge::DT_BF16) {
        Ops::Base::BroadcastBaseTiling<AddcdivOp::AddcdivOp<Ops::Base::half, Ops::Base::half>::OpDag> brcBaseTiling(
            context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT) {
        Ops::Base::BroadcastBaseTiling<AddcdivOp::AddcdivOp<float, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(
            context_->GetNodeName(),
            "Input dtype is only support fp16, bf16, fp32, "
            "while got %s!",
            ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus AddcdivTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AddcdivTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus AddcdivTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddcdivTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddcdivTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAddcdiv(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter TilingForAddcdiv");
    if (context == nullptr) {
        OP_LOGE(nodeName, "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AddcdivCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(nodeName, "Enter ascendc AddcdivTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForAddcdiv(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Addcdiv).Tiling(TilingForAddcdiv).TilingParse<AddcdivCompileInfo>(TilingPrepareForAddcdiv);

REGISTER_OPS_TILING_TEMPLATE(Addcdiv, AddcdivTiling, ADDCDIV_COMMON_TILING_PRIORITY);
} // namespace optiling
