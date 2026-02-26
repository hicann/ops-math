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
 * \file transdata_tiling_arch35.cpp
 * \brief calc tiling data for transdata AscendC kernel
 */

#include "transdata_tiling_arch35.h"

#include <algorithm>

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"

using namespace Ops::Math::OpTiling;
using namespace Ops::Base;

namespace optiling {
namespace transdata_asc {

ge::graphStatus TransDataTilingAscendC::GetHardwareInfo()
{
    auto compileInfo = reinterpret_cast<const TransDataCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = static_cast<uint32_t>(compileInfo->coreNum);
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(
        (coreNum_ <= 0U || ubSize_ <= 0L),
        OP_LOGE(context_->GetNodeName(), "TransData GetHardwareInfo failed, core num: %u, ub size: %ld", coreNum_,
                ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void TransDataTilingAscendC::ReshapeInShape()
{
    auto dimCnt = inShape.GetDimNum();
    if (dimCnt > 1) {
        return;
    }
    auto bakDim = inShape.GetDim(0);
    inShape.SetDim(0, 1);
    inShape.AppendDim(1);
    inShape.AppendDim(bakDim);
}

bool TransDataTilingAscendC::GetShapeInfo()
{
    auto xStorage = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    inShape = EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorage);
    outShape = EnsureNotScalar(yStorage->GetStorageShape());

    OP_CHECK_IF(
        inShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
        OP_LOGE(context_->GetNodeName(), "The input or output shape is empty!"),
        return false);

    ReshapeInShape();
    return true;
}

bool TransDataTilingAscendC::GetTransFormatAndDType()
{
    auto srcTd = context_->GetInputDesc(0);
    auto dstTd = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, srcTd);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstTd);
    auto srcFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(srcTd->GetStorageFormat()));
    dstFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(dstTd->GetStorageFormat()));
    srcDtype_ = srcTd->GetDataType();
    dtypeSize = ge::GetSizeByDataType(srcDtype_);

    std::vector<ge::Format> supportSrcFormat = {ge::FORMAT_ND, ge::FORMAT_NCL, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
    std::vector<ge::Format> supportDstFormat = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_16,
                                                ge::FORMAT_FRACTAL_NZ_C0_32};
    OP_CHECK_IF(
        (std::find(supportSrcFormat.begin(), supportSrcFormat.end(), srcFormat) == supportSrcFormat.end() ||
         std::find(supportDstFormat.begin(), supportDstFormat.end(), dstFormat) == supportDstFormat.end()),
        OP_LOGE(context_->GetNodeName(), "The input or output format is invalid!"),
        return false);
    return true;
}

bool TransDataTilingAscendC::CalcC0Size()
{
    auto dimCnt = outShape.GetDimNum();
    auto tmpC0 = outShape.GetDim(dimCnt - 1);

    OP_CHECK_IF(
        dstFormat == ge::FORMAT_FRACTAL_NZ_C0_16 && tmpC0 != C0_16,
        OP_LOGE(context_->GetNodeName(), "The c0 should be 16 when dst format is FRACTAL_NZ_C0_16!"),
        return false);
    OP_CHECK_IF(
        dstFormat == ge::FORMAT_FRACTAL_NZ_C0_32 && tmpC0 != C0_32,
        OP_LOGE(context_->GetNodeName(), "The c0 should be 32 when dst format is FRACTAL_NZ_C0_32!"),
        return false);

    int64_t expectC0 = tmpC0;
    if (dtypeSize == 1U) {
        expectC0 = C0_32;
    } else if (dtypeSize == nTwo) {
        expectC0 = C0_16;
    }
    OP_CHECK_IF(
        dstFormat == ge::FORMAT_FRACTAL_NZ && dtypeSize == nTwo * nTwo && C0_8 != expectC0 && C0_16 != expectC0,
        OP_LOGE(context_->GetNodeName(), "The c0 should be 8 or 16 when dst format is FRACTAL_NZ and dtype size is %zu!",
                dtypeSize),
        return false);
    OP_CHECK_IF(
        dstFormat == ge::FORMAT_FRACTAL_NZ && tmpC0 != expectC0,
        OP_LOGE(context_->GetNodeName(), "The c0 should be %ld when dst format is FRACTAL_NZ and dtype size is %zu!",
                expectC0, dtypeSize),
        return false);

    c0_ = tmpC0;
    if (srcDtype_ == ge::DT_FLOAT4_E2M1) {
        // 当作u8处理，c0缩小一半
        c0_ = c0_ >> 1;
    }
    return true;
}

void TransDataTilingAscendC::CalcHSize()
{
    int64_t res = 1;
    auto dimCnt = inShape.GetDimNum();
    if (dimCnt <= nTwo) {
        h_ = res;
        return;
    }

    for (size_t i = 0; i < dimCnt - nTwo; i++) {
        res *= inShape.GetDim(i);
    }
    h_ = res;
}

void TransDataTilingAscendC::CalcNCSize()
{
    auto dimCnt = inShape.GetDimNum();
    n_ = inShape.GetDim(dimCnt - nTwo);
    c_ = inShape.GetDim(dimCnt - 1);
    if (srcDtype_ == ge::DT_FLOAT4_E2M1) {
        // 当作u8处理，c缩小一半
        c_ = c_ >> 1;
    }
}

void TransDataTilingAscendC::CalcTilingKey()
{
    int64_t ni = 16;
    int64_t shapeSize = h_ * CeilAlign(n_, ni) * CeilAlign(c_, c0_);
    tilingKey_ = (shapeSize > MAX_INT32_SIZE) ? TILING_MODE_SIMT_LARGE_SHAPE : TILING_MODE_SIMT;
}

void TransDataTilingAscendC::CalcBlockAndThreadNum()
{
    bNum_ = coreNum_;
    tNum_ = (tilingKey_ == TILING_MODE_SIMT) ? tNum512 : tNum256;
}

ge::graphStatus TransDataTilingAscendC::CalcTilingData()
{
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "Failed to get shape info!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !GetTransFormatAndDType(), OP_LOGE(context_->GetNodeName(), "Failed to get format and dtype info!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CalcC0Size(), OP_LOGE(context_->GetNodeName(), "Failed to get c0 size!"), return ge::GRAPH_FAILED);
    CalcHSize();
    CalcNCSize();
    CalcTilingKey();
    CalcBlockAndThreadNum();

    return ge::GRAPH_SUCCESS;
}

void TransDataTilingAscendC::WriteTilingData()
{
    context_->SetBlockDim(bNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetLocalMemorySize(ubSize_ - SIMT_RSV_SIZE);

    tilingData_.set_c0(c0_);
    tilingData_.set_h(h_);
    tilingData_.set_n(n_);
    tilingData_.set_c(c_);
    tilingData_.set_tNum(tNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

std::string TransDataTilingAscendC::PrintTilingData()
{
    std::string tilingStr;
    tilingStr += std::to_string(c0_) + ",";
    tilingStr += std::to_string(h_) + ",";
    tilingStr += std::to_string(n_) + ",";
    tilingStr += std::to_string(c_) + ",";
    tilingStr += std::to_string(tNum_);
    return tilingStr;
}

ge::graphStatus TransDataTilingAscendC::DoTiling()
{
    OP_CHECK_IF(
        (CalcTilingData() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "TransDataTilingAscendC failed to calc tiling data."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= SIMT_RSV_SIZE, OP_LOGE(context_->GetNodeName(), "UB size too small for SIMT reserved size."),
                return ge::GRAPH_FAILED);
    WriteTilingData();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = kSyncWorkSpaceSize;
    OP_LOGI(context_->GetNodeName(), "TransData tiling data: %s", PrintTilingData().c_str());
    return ge::GRAPH_SUCCESS;
}

}  // namespace transdata_asc

ge::graphStatus Tiling4TransDataAscendC(gert::TilingContext* context)
{
    transdata_asc::TransDataTilingAscendC tdTiling(context);
    OP_CHECK_IF(
        (tdTiling.GetHardwareInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "TransDataTilingAscendC failed to get hardware info."),
        return ge::GRAPH_FAILED);
    return tdTiling.DoTiling();
}

static ge::graphStatus TilingPrepare4TransData(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<TransDataCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invalid."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransData).Tiling(Tiling4TransDataAscendC).TilingParse<TransDataCompileInfo>(TilingPrepare4TransData);

}  // namespace optiling
