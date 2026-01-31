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
 * \file tile_tiling_arch35.cpp
 * \brief calc tiling for tile
 */
#include "tile_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "util/platform_util.h"

namespace optiling {
constexpr size_t TILE_MAX_DIM_NUM = 0x8;

template <typename T>
static std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus CheckTileRule(const gert::TilingContext* context, const gert::Shape& inShape,
    const gert::Shape& outShape)
{
    auto outDimNum = outShape.GetDimNum();
    OP_CHECK_IF(
        inShape.GetDimNum() != outDimNum,
        OP_LOGE(context->GetNodeName(), "The input shape dims are different with output's!"),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < outDimNum; i++) {
        if (outShape[i] < inShape[i] || outShape[i] % inShape[i] != 0) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static void ConvertTileAxis2BroadcastToAxis(gert::Shape& inShape, gert::Shape& outShape)
{
    gert::Shape newInShape;
    gert::Shape newOutShape;
    auto DimNum = inShape.GetDimNum();
    for (size_t i = 0; i < DimNum; i++) {
        newInShape.AppendDim(1);
        newInShape.AppendDim(inShape[i]);
        newOutShape.AppendDim(outShape[i] / inShape[i]);
        newOutShape.AppendDim(inShape[i]);
    }
    inShape = newInShape;
    outShape = newOutShape;
}

static ge::graphStatus GetShapeInfo(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    inShape = Ops::Math::OpTiling::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    outShape = Ops::Math::OpTiling::EnsureNotScalar(yStorage->GetStorageShape());

    auto outDimNum = outShape.GetDimNum();
    OP_CHECK_IF(inShape.GetDimNum() > outDimNum,
                    OP_LOGE(context->GetNodeName(),
                                                    "The input shape has more dimensions than output shape!"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outDimNum > TILE_MAX_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Not support the dim num: %lu yet!", outDimNum),
        return ge::GRAPH_FAILED);
    brcto::AdjustShapesToSameDimNum(inShape, outDimNum);
    OP_CHECK_IF(inShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
                    OP_LOGE(context->GetNodeName(), "The input or output shape is empty!"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTileRule(context, inShape, outShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(),
                                                    "The input and output shapes mismatch the broadcast rule!"),
                    return ge::GRAPH_FAILED);
    ConvertTileAxis2BroadcastToAxis(inShape, outShape);

    OP_LOGI(context->GetNodeName(), "The broadcastTo input and output after convertion is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    OP_CHECK_IF(brcto::DeleteOneSizeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to delete one size axes!"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "The reshaped input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    OP_CHECK_IF(brcto::MergeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to merge axes!"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "The merged input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4TileAscendC(gert::TilingContext* context, const gert::Shape* inShapePtr,
                                          const gert::Shape* outShapePtr)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    brcto::BroadcastToTilingAscendC brcToTiling(context, inShapePtr, outShapePtr);

    OP_CHECK_IF((brcToTiling.GetHardwareInfo<TileCompileInfo>() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context->GetNodeName(),
                                                    "BroadcastToTilingAscendC failed to get hardware info."),
                    return ge::GRAPH_FAILED);

    return brcToTiling.DoTiling();
}

static ge::graphStatus Tiling4Tile(gert::TilingContext* context) {
  auto compile_info = reinterpret_cast<const TileCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  gert::Shape inShape;
  gert::Shape outShape;

  // convert tile input shape to broadcastTo input, then use broadcastTo template
  OP_CHECK_IF(GetShapeInfo(context, inShape, outShape) != ge::GRAPH_SUCCESS,
          OP_LOGE(context->GetNodeName(), "Get input or output shape was failed!"),
          return ge::GRAPH_FAILED);
  return Tiling4TileAscendC(context, &inShape, &outShape);
  
}

static ge::graphStatus TilingPrepare4Tile(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4Tile.");

  auto compileInfo = context->GetCompiledInfo<TileCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

  compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
  OP_CHECK_IF((compileInfo->coreNum <= 0),
                  OP_LOGE(context->GetNodeName(), "The core num is negative."),
                  return ge::GRAPH_FAILED);

  uint64_t ubSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compileInfo->ubSize = static_cast<int64_t>(ubSize);
  OP_CHECK_IF((compileInfo->ubSize <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                  return ge::GRAPH_FAILED);

  compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
  OP_CHECK_IF((compileInfo->clSize <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to get cache line size."),
                  return ge::GRAPH_FAILED);

  compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
  OP_CHECK_IF((compileInfo->blockSize <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to get block size."),
                  return ge::GRAPH_FAILED);

  compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
  OP_CHECK_IF((compileInfo->vRegSize <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to get vReg size."),
                  return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4Tile.");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Tile op.
IMPL_OP_OPTILING(Tile).Tiling(Tiling4Tile).TilingParse<TileCompileInfo>(TilingPrepare4Tile);
}  // namespace optiling
