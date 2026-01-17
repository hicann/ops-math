/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file assign_add_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <graph/utils/type_utils.h>

#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../op_kernel/arch35/assign_add_dag.h"
#include "../../op_kernel/arch35/assign_add_struct.h"
#include "assign_add_tiling_arch35.h"

using namespace ge;
using namespace AssignAddOp;
using namespace AssignAddDag;

namespace optiling {
using namespace Ops::Math::OpTiling;
const size_t SYS_WORKSPACE_SIZE = 16777216;

class AssignAddTiling {
public:
    explicit AssignAddTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape() const;

private:
    ge::DataType outputDtype;
    ge::DataType valueDtype;
    gert::TilingContext* tilingContext;
};

ge::graphStatus AssignAddTiling::CheckDtype()
{
    auto refDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, refDesc);
    auto valueDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, valueDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);

    auto refDtype = refDesc->GetDataType();
    this->outputDtype = outputDesc->GetDataType();
    this->valueDtype = valueDesc->GetDataType();

    OP_CHECK_IF(
        refDtype != this->outputDtype, OP_LOGE(tilingContext->GetNodeName(), "dtype of ref and output are not same"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (refDtype != this->valueDtype) && (refDtype != ge::DT_FLOAT),
        OP_LOGE(tilingContext->GetNodeName(), "if ref dtype not equal to value dtype, ref dtype only support float32"),
        return ge::GRAPH_FAILED);

    std::vector<ge::DataType> valueMixDtype = {ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(
        (refDtype != this->valueDtype) &&
            (std::find(valueMixDtype.begin(), valueMixDtype.end(), this->valueDtype) == valueMixDtype.end()),
        OP_LOGE(
            tilingContext->GetNodeName(),
            "if ref dtype not equal to value dtype, value dtype only support float16, bfloat16"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssignAddTiling::CheckShape() const
{
    auto refStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, refStorageShape);
    auto valueStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, valueStorageShape);
    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& refShape = EnsureNotScalar(refStorageShape->GetStorageShape());
    const gert::Shape& valueShape = EnsureNotScalar(valueStorageShape->GetStorageShape());
    const gert::Shape& outputShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        refShape != valueShape || refShape != outputShape,
        OP_LOGE(tilingContext->GetNodeName(), "shape of ref、value and output are not same"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssignAddTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "AssignAddTiling RunTiling enter.");
    ElewiseBaseTiling eleBaseTiling(tilingContext);
    OP_CHECK_IF(
        CheckDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get AssignAddTiling from GE context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        ret = eleBaseTiling.DoTiling<AssignAddDAG<bfloat16_t, bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        ret = eleBaseTiling.DoTiling<AssignAddDAG<int8_t, int8_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        ret = eleBaseTiling.DoTiling<AssignAddDAG<int32_t, int32_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        ret = eleBaseTiling.DoTiling<AssignAddDAG<int64_t, int64_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        ret = eleBaseTiling.DoTiling<AssignAddDAG<uint8_t, uint8_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (this->valueDtype == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<AssignAddDAG<float, float>::OpDag>(*tiling);
        } else if (this->valueDtype == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<AssignAddDAG<float, half>::OpDag>(*tiling);
        } else if (this->valueDtype == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<AssignAddDAG<float, bfloat16_t>::OpDag>(*tiling);
        } else {
            OP_LOGE(tilingContext->GetNodeName(), "if ref dtype is fp32, value dtype only support fp16, bf16, fp32.");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(
            tilingContext->GetNodeName(), "output dtype is only support fp16, bf16, fp32, int8、int32、int64、uint8.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE_SIZE;

    uint64_t dType = TPL_DTYPE_BASE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->scheMode), dType);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AssignAdd(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4AssignAdd rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const AssignAddCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    AssignAddTiling assignAddTiling(context);
    return assignAddTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForAssignAdd(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AssignAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AssignAdd).Tiling(Tiling4AssignAdd).TilingParse<AssignAddCompileInfo>(TilingPrepareForAssignAdd);
} // namespace optiling