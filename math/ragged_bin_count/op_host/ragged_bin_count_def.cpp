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
 * \file ragged_bin_count_def.cpp
 * \brief Operator definition for RaggedBinCount on Ascend 950.
 */

#include "register/op_def_registry.h"
#include "graph/operator.h"

namespace {
using namespace ge;

constexpr uint32_t INPUT_SPLITS = 0U;
constexpr uint32_t INPUT_VALUES = 1U;
constexpr uint32_t INPUT_SIZE = 2U;
constexpr uint32_t INPUT_WEIGHTS = 3U;
constexpr uint32_t OUTPUT_RESULT = 0U;

graphStatus CheckIfAICoreSupported(const Operator& op, AscendString& result)
{
    const DataType splitsDtype = op.GetInputDesc(INPUT_SPLITS).GetDataType();
    const DataType valuesDtype = op.GetInputDesc(INPUT_VALUES).GetDataType();
    const DataType sizeDtype = op.GetInputDesc(INPUT_SIZE).GetDataType();
    const DataType weightsDtype = op.GetInputDesc(INPUT_WEIGHTS).GetDataType();
    const DataType outputDtype = op.GetOutputDesc(OUTPUT_RESULT).GetDataType();

    const bool isNativeCombination = splitsDtype == DT_INT64 && (valuesDtype == DT_INT32 || valuesDtype == DT_INT64) &&
                                     sizeDtype == valuesDtype && weightsDtype == DT_FLOAT && outputDtype == DT_FLOAT;
    if (!isNativeCombination) {
        result = AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "RaggedBinCount Ascend 950 supports only splits=int64, values/size=int32 or int64 with matching dtype, and weights/output=float32."})");
        return GRAPH_FAILED;
    }

    result = AscendString(
        R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "RaggedBinCount native dtype contract passed."})");
    return GRAPH_SUCCESS;
}
} // namespace

namespace ops {
class RaggedBinCount : public OpDef {
public:
    explicit RaggedBinCount(const char* name) : OpDef(name)
    {
        // The public IR keeps the historical broad dtype set. The native Ascend 950
        // kernel intentionally registers only the two previously verified FP32 combinations.
        this->Input("splits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("binary_output").AttrType(OPTIONAL).Bool(false);

        this->AICore().SetCheckSupport(CheckIfAICoreSupported);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            // Aligned with canndev: its RaggedBinCount op-info leaves precision_reduce unset, so
            // pinning ours to false was the only field where the 950 op-info diverged from the A2/A3
            // baseline, and the support surface must never be narrower than canndev's. Setting true
            // cannot widen us either -- the op-info still registers fp32 weights/output only, so the
            // mixed-precision pass has no lower-precision variant to switch to.
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "ragged_bin_count");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(RaggedBinCount);
// ragged_bin_count_def.cpp is also compiled into the tiling object so that the
// pre-selection CheckSupport callback is available to FE before compatibility casts.
static int g_RaggedBinCount_register_check_support = [](const char* name) {
    RaggedBinCount opDef(name);
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, name, opDef.AICore().GetCheckSupport());
    return 0;
}("RaggedBinCount");
} // namespace ops
