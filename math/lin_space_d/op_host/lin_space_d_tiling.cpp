/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_d_tiling.cpp
 * \brief LinSpaceD operator tiling implementation
 */
#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/lin_space_d_tiling_data.h"
#include "../op_kernel/lin_space_d_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;
static const size_t INPUT_IDX_START = 0;
static const size_t INPUT_IDX_STOP = 1;
static const size_t INPUT_IDX_NUM = 2;
static const uint32_t BLOCK_SIZE = 32;
static const uint32_t BUFFER_NUM = 2;
static const uint32_t WS_SYS_SIZE = 0; 

struct LinSpaceDCustomCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取shape和数据类型信息
static ge::graphStatus GetShapeAndTypeInfo(gert::TilingContext* context, uint32_t& totalLength, 
                                         int& start_type, int& end_type)
{
    // 获取num输入张量
    auto tensorN = context->GetInputTensor(INPUT_IDX_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorN);
    auto dataPtr = tensorN->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, dataPtr);
    totalLength = static_cast<uint32_t>(dataPtr[0]);

    // 获取数据类型
    auto inputStartDesc = context->GetInputDesc(INPUT_IDX_START);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStartDesc);
    start_type = inputStartDesc->GetDataType();

    auto inputEndDesc = context->GetInputDesc(INPUT_IDX_STOP);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputEndDesc);
    end_type = inputEndDesc->GetDataType();

    return ge::GRAPH_SUCCESS;
}

// 计算Tiling数据
static ge::graphStatus CalculateTilingData(LinSpaceDTilingData* tiling,
                                         uint64_t ubSize, int64_t coreNum, uint32_t totalLength,
                                         uint32_t dataTypeSize)
{
    uint32_t ubDataNumber = 20;
    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
    uint32_t ubBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint32_t totalLengthAligned = (totalLength % alignNum == 0U)
        ? totalLength
        : ((totalLength + alignNum - 1) / alignNum) * alignNum;
  
    uint32_t totalBlockCount = totalLengthAligned / alignNum;  
    uint32_t formerNum = totalBlockCount % coreNum;                          
    uint32_t formerLength = 0, tailLength = 0;
    uint32_t fTileNum = 0, fLastTileLen = 0;
    uint32_t tTileNum = 0, tLastTileLen = 0;
    uint32_t maxPerCoreElem = alignNum * ubBlockNum;

    if (formerNum != 0) {
        formerLength = (totalBlockCount + coreNum - 1) / coreNum * alignNum;
        fTileNum = formerLength / maxPerCoreElem; 
        
        if ((formerLength % maxPerCoreElem == 0U) || (fTileNum == 0U)) {
            if (fTileNum == 0U) {
                fTileNum = (formerLength + alignNum - 1) / alignNum;
            }                                                                      
            if (formerLength < maxPerCoreElem) {
                uint32_t blockCount = (formerLength + alignNum - 1) / alignNum;  
                blockCount = (blockCount + BUFFER_NUM - 1) / BUFFER_NUM * BUFFER_NUM;                             
                fLastTileLen = (fTileNum == 1) ? formerLength : (formerLength % alignNum == 0 ? alignNum : formerLength % alignNum);
            } 
            else {
                fLastTileLen = formerLength - (fTileNum - 1) * maxPerCoreElem;
            }
        } 
        else {
            fTileNum++;
            fLastTileLen = formerLength - (fTileNum - 1) * maxPerCoreElem;
        }
    }

    tailLength = totalBlockCount / coreNum * alignNum;
    tTileNum = tailLength / maxPerCoreElem;  
    if ((tailLength % maxPerCoreElem == 0U) || (tTileNum == 0U)) {
        if (tTileNum == 0U) {
            tTileNum = (tailLength + alignNum - 1) / alignNum;
        }                                                                      
        if (tailLength < maxPerCoreElem) {
            uint32_t blockCount = (tailLength + alignNum - 1) / alignNum;  
            blockCount = (blockCount + BUFFER_NUM - 1) / BUFFER_NUM * BUFFER_NUM;                             
            tLastTileLen = (tTileNum == 1) ? tailLength : (tailLength % alignNum == 0 ? alignNum : tailLength % alignNum);
        } 
        else {
            tLastTileLen = tailLength - (tTileNum - 1) * maxPerCoreElem;
        }
    } 
    else {
        tTileNum++;
        tLastTileLen = tailLength - (tTileNum - 1) * maxPerCoreElem;
    }

    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->formerTileNum = fTileNum;
    tiling->formerLastTileLength = fLastTileLen;
    tiling->tailLength = tailLength;
    tiling->tailTileNum = tTileNum;
    tiling->tailLastTileLength = tLastTileLen;  
    tiling->totalLength = totalLength;
    
    return ge::GRAPH_SUCCESS;
}

// 获取WorkspaceSize信息
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus LinSpaceDTilingFunc(gert::TilingContext* context)
{
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, 
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    
    // 获取shape和类型信息
    uint32_t totalLength;
    int start_type, end_type;
    OP_CHECK_IF(
        GetShapeAndTypeInfo(context, totalLength, start_type, end_type) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndTypeInfo error"), 
        return ge::GRAPH_FAILED);

    // 获取WorkspaceSize信息    
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, 
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 设置tiling信息
    uint32_t dataTypeSize = 4; 
    LinSpaceDTilingData* tiling = context->GetTilingData<LinSpaceDTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LinSpaceDTilingData), 0, sizeof(LinSpaceDTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), 
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalculateTilingData(tiling, ubSize, coreNum, totalLength, dataTypeSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateTilingData error"), 
        return ge::GRAPH_FAILED);

    context->SetBlockDim(coreNum);

    uint64_t tilingKey = 0;
    if (
        (start_type == ge::DT_UINT8 || start_type == ge::DT_BF16 ||
         start_type == ge::DT_FLOAT16 || start_type == ge::DT_FLOAT || 
         start_type == ge::DT_INT8 || start_type == ge::DT_INT16 || start_type == ge::DT_INT32) &&
        (end_type == ge::DT_UINT8  || end_type == ge::DT_BF16 ||
         end_type == ge::DT_FLOAT16 || end_type == ge::DT_FLOAT || 
         end_type == ge::DT_INT8 || end_type == ge::DT_INT16 || end_type == ge::DT_INT32)
    ) {
        tilingKey = GET_TPL_TILING_KEY(start_type, end_type);
        OP_CHECK_IF(
            context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS, 
            OP_LOGE(context, "SetTilingKey failed"), 
            return ge::GRAPH_FAILED);
    }
    else {
        OP_LOGE(context, "Unsupported data type combination");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLinSpaceD([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(LinSpaceD).Tiling(LinSpaceDTilingFunc).TilingParse<LinSpaceDCustomCompileInfo>(TilingParseForLinSpaceD);

} // namespace optiling
