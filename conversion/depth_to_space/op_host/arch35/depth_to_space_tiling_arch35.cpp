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
 * \file depth_to_space_tiling_arch35.cpp
 * \brief tiling for DepthToSpace
 */
#include "depth_to_space_tiling_arch35.h"

namespace optiling {

namespace DepthToSpace {

static ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareTransposeForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (ci->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Transpose Op get coreNum:%ld, ubSize:%ld.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DepthToSpaceTiling::ParametersVerifying()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DepthToSpaceTiling ParametersVerifying.");

    auto xStorageShape = tilingContext_->GetInputShape(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xStorageShape);
    auto xShape= xStorageShape->GetStorageShape();
    auto xDimNum = xShape.GetDimNum();
    paramInfo_.xShape = xShape;

    auto yStorageShape = tilingContext_->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yStorageShape);
    auto yShape = yStorageShape->GetStorageShape();
    auto yDimNum = yShape.GetDimNum();

    auto xDesc = tilingContext_->GetInputDesc(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xDtype = xDesc->GetDataType();
    paramInfo_.xDtype = xDtype;

    auto yDesc = tilingContext_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yFormat = yDesc->GetFormat().GetStorageFormat();

    // lmit input output dim is 4D
    OP_CHECK_IF((xDimNum != DIM_NUM || yDimNum != DIM_NUM), OP_LOGE(tilingContext_->GetNodeName(), "Invalid x or y shape dim, they should be the 4."), return ge::GRAPH_FAILED);

    // limit input output format is same and limit NCHW and NHWC
    OP_CHECK_IF((xFormat != yFormat || (xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NHWC)), OP_LOGE(tilingContext_->GetNodeName(), "Invalid x or y format, they should be the same and only support NCHW or NHWC."), return ge::GRAPH_FAILED);

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, blockSizePtr);
    paramInfo_.blockSize = *blockSizePtr;

    const char* modePtr = attrs->GetAttrPointer<char>(ATTR_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, modePtr);
    paramInfo_.modePtr = modePtr;

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(ATTR_DEPTH_DATA_FORMAT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, dataFormatPtr);
    paramInfo_.dataFormatPtr = dataFormatPtr;

    // limit input format is same as attr format
    ge::Format geFormat = ge::FORMAT_NCHW;
    if (strcmp(dataFormatPtr, "NCHW") == 0) {
        geFormat = ge::FORMAT_NCHW;
    } else if (strcmp(dataFormatPtr, "NHWC") == 0) {
        geFormat = ge::FORMAT_NHWC;
    } else {
        OP_LOGE(tilingContext_->GetNodeName(), "Invalid attr format, it only supports NCHW or NHWC.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((xFormat != geFormat), OP_LOGE(tilingContext_->GetNodeName(), "Invalid attr format, it should be the same as x format."), return ge::GRAPH_FAILED);

    // limit blockSize positive integer greater than or equal to 2
    OP_CHECK_IF((*blockSizePtr < 2), OP_LOGE(tilingContext_->GetNodeName(), "Invalid attr block size, it must be a positive integer greater than or equal to 2."), return ge::GRAPH_FAILED);

    // depth % (block_size**2) == 0
    auto depth = xShape.GetDim(xFormat == ge::FORMAT_NCHW ? 1 : 3);
    OP_CHECK_IF((depth % (*blockSizePtr * *blockSizePtr) != 0), OP_LOGE(tilingContext_->GetNodeName(), "Invalid attr block size, depth size must be divisible by the square of the block size."), return ge::GRAPH_FAILED);

    // limit mode "DCR" or "CRD"
    OP_CHECK_IF((strcmp(modePtr, "DCR") != 0 && strcmp(modePtr, "CRD") != 0), OP_LOGE(tilingContext_->GetNodeName(), "Invalid attr mode, it should be the DCR or CRD."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void DepthToSpaceTiling::ProcessShapeInfo(ShapeInfo& shapeInfo) {
    OP_LOGD(tilingContext_->GetNodeName(), "Start DepthToSpaceTiling ProcessShapeInfo.");
    shapeInfo.permSize = DIM_SIX;
    shapeInfo.eleLenInBytes = ge::GetSizeByDataType(paramInfo_.xDtype);
    shapeInfo.inShapeSize = DIM_SIX;
    shapeInfo.outShapeSize = DIM_SIX;
    shapeInfo.dim = DIM_SIX;
    shapeInfo.origDim = DIM_SIX;
    if (strcmp(paramInfo_.dataFormatPtr, "NHWC") == 0 && strcmp(paramInfo_.modePtr, "DCR") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE];
        shapeInfo.inShape[DIM_TWO] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_THREE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        for (int64_t i = 0; i < DIM_SIX; i++) {
            shapeInfo.perm[i] = nhwcDcrPerm_[i];
            shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
        }
    } else if (strcmp(paramInfo_.dataFormatPtr, "NCHW") == 0 && strcmp(paramInfo_.modePtr, "DCR") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_TWO] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_THREE] = paramInfo_.xShape[DIM_ONE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE];
        for (int64_t i = 0; i < DIM_SIX; i++) {
            shapeInfo.perm[i] = nchwDcrPerm_[i];
            shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
        }
    } else if (strcmp(paramInfo_.dataFormatPtr, "NHWC") == 0 && strcmp(paramInfo_.modePtr, "CRD") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE];
        shapeInfo.inShape[DIM_TWO] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_THREE] = paramInfo_.xShape[DIM_THREE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.blockSize;
        for (int64_t i = 0; i < DIM_SIX; i++) {
            shapeInfo.perm[i] = crdPerm_[i];
            shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
        }
    } else if (strcmp(paramInfo_.dataFormatPtr, "NCHW") == 0 && strcmp(paramInfo_.modePtr, "CRD") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        shapeInfo.inShape[DIM_TWO] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_THREE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE];
        for (int64_t i = 0; i < DIM_SIX; i++) {
            shapeInfo.perm[i] = crdPerm_[i];
            shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
        }
    } else {
         OP_LOGE(tilingContext_->GetNodeName(), "DepthToSpaceTiling not support the data format and mode!");
    }
}

ge::graphStatus DepthToSpaceTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo)
{   
    OP_LOGD(context->GetNodeName(), "Start DepthToSpaceTilingForAscendC.");
    DepthToSpaceTiling tilingObject(context);

    OP_CHECK_IF(tilingObject.ParametersVerifying() != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(),
        "DepthToSpaceTiling failed to verify params!"),
        return ge::GRAPH_FAILED);

    // construct an equivalent Transpose inputShapeInfo
    ShapeInfo inputShapeInfo;
    tilingObject.ProcessShapeInfo(inputShapeInfo);

    DepthToSpaceTilingData tilingData;
    DepthToSpaceCompileInfo compileInfo;
    compileInfo.transposeCompilerInfo.coreNum = transposeCompileInfo->coreNum;
    compileInfo.transposeCompilerInfo.ubSize = transposeCompileInfo->ubSize;

    TransposeNddmaTiling transposeTilingObject(context);
    OP_CHECK_IF(
        (transposeTilingObject.TilingForReleatedTranspose(context, &tilingData.transposeOpTiling,
                                                          &compileInfo.transposeCompilerInfo,
                                                          inputShapeInfo) == ge::GRAPH_FAILED),
         OP_LOGE(context->GetNodeName(), "Transpose Tiling failed"),
         return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                             context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context->GetNodeName(), "DepthToSpaceTilingForAscendC success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForDepthToSpace(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingForDepthToSpace");
  auto compile_info = reinterpret_cast<const TransposeCompilerInfo*>(context->GetCompileInfo());
  return DepthToSpaceTilingForAscendC(context, compile_info);
}

static ge::graphStatus TilingPrepareForRelatedToTranspose(gert::TilingParseContext* context) 
{
  OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForRelatedToTranspose.");
  TilingPrepareTransposeForAscendC(context);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DepthToSpace).Tiling(TilingForDepthToSpace)
                              .TilingParse<TransposeCompilerInfo>(TilingPrepareForRelatedToTranspose);

}  // namespace DepthToSpace
}  // namespace optiling