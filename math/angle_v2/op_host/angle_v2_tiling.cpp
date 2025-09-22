/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file angle_v2_tiling.cpp
 * \brief
 */

#include <sstream>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "log/log.h"
#include "angle_v2_tiling.h"

namespace optiling {
constexpr uint64_t COMPLEX64_MODE = 1UL;
constexpr uint64_t FP32_MODE = 2UL;
constexpr uint64_t FP16_MODE = 3UL;
constexpr uint64_t BOOL_MODE = 4UL;
constexpr uint64_t UINT8_MODE = 5UL;
constexpr uint64_t INT8_MODE = 6UL;
constexpr uint64_t INT16_MODE = 7UL;
constexpr uint64_t INT32_MODE = 8UL;
constexpr uint64_t INT64_MODE = 9UL;
constexpr uint32_t SIZE_OF_B8 = 1;
constexpr uint32_t SIZE_OF_B16 = 2;
constexpr uint32_t SIZE_OF_B32 = 4;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;                 // The amount of data that can be processed by a repeat.
constexpr uint32_t SELECT_MODE_GE_ZERO_TMP_UB = 8000; // select mode 2 need 8000B
class AngleV2Tiling
{
public:
    explicit AngleV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    void SetTilingKeyMode(ge::DataType dType);
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatform, ge::DataType dType);
    void GetUsedBytesPerDataInKernel(ge::DataType dType);
    void CalTilingAligned(ge::DataType dType);
    void GetAlignNum(ge::DataType dType);
    AngleV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t coreNum = 0;
    uint32_t dataPerRepeat = 0;
    uint32_t tileLength = 0;         // align to 256B
    uint32_t totalLength = 1;        // the length of input
    uint32_t formerNum = 0;          // deal more data core num
    uint32_t tailNum = 0;            // deal less data core num
    uint32_t formerLength = 0;       // deal more data length
    uint32_t tailLength = 0;         // deal less data length
    uint32_t alignNum = 0;           // data count per block
    uint32_t totalLengthAligned = 0; // length to align 32B
    uint64_t ubSizePlatForm = 0UL;
    uint32_t bytesPerData = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

void AngleV2Tiling::GetUsedBytesPerDataInKernel(ge::DataType dType)
{
    // Calculate the bytes of buffer for one element used
    uint32_t bytesI8 = 1;
    uint32_t bytesI16 = 2;
    uint32_t bytesI32 = 4;
    uint32_t bytesI64 = 8;
    uint32_t coefficentTwo = 2;
    uint32_t coefficentThree = 3;
    uint32_t coefficentTen = 10;
    switch (dType) {
        case ge::DT_COMPLEX64:
            // double buffer for input(complex64) and output(float32)
            bytesPerData = bytesI64 * coefficentTwo + bytesI32 * coefficentTwo;
            // two masks(uint8) and ten localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 * coefficentTwo + bytesI32 * coefficentTen;
            break;
        case ge::DT_FLOAT:
            // double buffer for input(float32) and output(float32)
            bytesPerData = bytesI32 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and three localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentThree;
            break;
        case ge::DT_INT8:
            // double buffer for input(int8) and output(float32)
            bytesPerData = bytesI8 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) ,two localTensor(float32) and one castTensor(float16) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo + bytesI16;
            break;
        case ge::DT_INT16:
            // double buffer for input(int16) and output(float32)
            bytesPerData = bytesI16 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and two localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo;
            break;
        case ge::DT_INT32:
            // double buffer for input(int32) and output(float32)
            bytesPerData = bytesI32 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and two localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo;
            break;
        case ge::DT_INT64:
            // double buffer for input(int64) and output(float32)
            bytesPerData = bytesI64 * coefficentTwo + bytesI32 * coefficentTwo;
            // one masks(uint8) and two localTensor(float32) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI32 * coefficentTwo;
            break;
        default:
            // double buffer for input(float16) and output(float16)
            bytesPerData = bytesI16 * coefficentTwo + bytesI16 * coefficentTwo;
            // one masks(uint8) and three localTensor(float16) are used for calculate the ouput
            bytesPerData += bytesI8 + bytesI16 * coefficentThree;
            break;
    }
}

void AngleV2Tiling::SetTilingKeyMode(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_COMPLEX64:
            tilingContext->SetTilingKey(COMPLEX64_MODE);
            break;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(FP16_MODE);
            break;
        case ge::DT_BOOL:
            tilingContext->SetTilingKey(BOOL_MODE);
            break;
        case ge::DT_UINT8:
            tilingContext->SetTilingKey(UINT8_MODE);
            break;
        case ge::DT_INT8:
            tilingContext->SetTilingKey(INT8_MODE);
            break;
        case ge::DT_INT16:
            tilingContext->SetTilingKey(INT16_MODE);
            break;
        case ge::DT_INT32:
            tilingContext->SetTilingKey(INT32_MODE);
            break;
        case ge::DT_INT64:
            tilingContext->SetTilingKey(INT64_MODE);
            break;
        default:
            tilingContext->SetTilingKey(FP32_MODE);
            break;
    }
}

uint32_t AngleV2Tiling::GetNeedCoreNum(const uint32_t coreNumPlatform, ge::DataType dType)
{
    if (dType == ge::DT_FLOAT16) {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_B16;
    } else {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_B32;
    }
    uint32_t tempCoreNum = (totalLength + dataPerRepeat - 1) / dataPerRepeat;
    if (tempCoreNum == 0) {
        tempCoreNum = 1;
    }
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void AngleV2Tiling::GetAlignNum(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_FLOAT16:
            alignNum = BYTE_BLOCK / SIZE_OF_B16;
            break;
        case ge::DT_INT8:
            alignNum = BYTE_BLOCK / SIZE_OF_B8;
            break;
        case ge::DT_INT16:
            alignNum = BYTE_BLOCK / SIZE_OF_B16;
            break;
        default:
            alignNum = BYTE_BLOCK / SIZE_OF_B32;
            break;
    }
}
void AngleV2Tiling::CalTilingAligned(ge::DataType dType)
{
    GetAlignNum(dType);
    totalLengthAligned = ((totalLength + alignNum - 1) / alignNum) * alignNum;
    // Divide blocks evenly into each core
    auto blockNum = totalLengthAligned / alignNum;
    formerNum = blockNum % coreNum;
    tailNum = coreNum - formerNum;
    formerLength = ((blockNum + coreNum - 1) / coreNum) * alignNum;
    tailLength = (blockNum / coreNum) * alignNum;

    if (socVersion == platform_ascendc::SocVersion::ASCEND910) {
        tileLength = ubSizePlatForm / bytesPerData / dataPerRepeat * dataPerRepeat;
    } else {
        tileLength = (ubSizePlatForm - SELECT_MODE_GE_ZERO_TMP_UB) / bytesPerData / dataPerRepeat * dataPerRepeat;
    }
}

ge::graphStatus AngleV2Tiling::Init()
{
    OP_LOGD(tilingContext, "Tiling initing.");
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    auto inputShape0 = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputShape0);
    auto xShape = inputShape0->GetStorageShape();
    totalLength = xShape.GetShapeSize();
    OP_LOGD(tilingContext, "totalLength %u.", totalLength);

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion = ascendcPlatform.GetSocVersion();
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(tilingContext, "coreNum %u.", coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_LOGD(tilingContext, "ub_size_platform is %lu.", ubSizePlatForm);
    auto inputDesc0 = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc0);
    auto dType = inputDesc0->GetDataType();
    coreNum = GetNeedCoreNum(coreNum, dType);

    SetTilingKeyMode(dType);
    GetUsedBytesPerDataInKernel(dType);
    CalTilingAligned(dType);
    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AngleV2Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");
    tilingContext->SetBlockDim(coreNum);
    tilingData.set_totalLength(totalLength);
    tilingData.set_formerNum(formerNum);
    tilingData.set_tailNum(tailNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_tailLength(tailLength);
    tilingData.set_alignNum(alignNum);
    tilingData.set_totalLengthAligned(totalLengthAligned);
    tilingData.set_tileLength(tileLength);
    tilingData.set_dataPerRepeat(dataPerRepeat);
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void AngleV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %u.", coreNum);
    OP_LOGD(tilingContext, "totalLength: %u.", totalLength);
    OP_LOGD(tilingContext, "formerNum: %u.", formerNum);
    OP_LOGD(tilingContext, "tailNum: %u.", tailNum);
    OP_LOGD(tilingContext, "formerLength: %u.", formerLength);
    OP_LOGD(tilingContext, "tailLength: %u.", tailLength);
    OP_LOGD(tilingContext, "alignNum: %u.", alignNum);
    OP_LOGD(tilingContext, "totalLengthAligned: %u.", totalLengthAligned);
    OP_LOGD(tilingContext, "tileLength: %u.", tileLength);
    OP_LOGD(tilingContext, "dataPerRepeat: %u.", dataPerRepeat);
}

static ge::graphStatus TilingAngleV2(gert::TilingContext* context)
{
    AngleV2Tiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForAngleV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForAngleV2 start.");
    auto compileInfo = context->GetCompiledInfo<AngleV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForAngleV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AngleV2).Tiling(TilingAngleV2).TilingParse<AngleV2CompileInfo>(TilingPrepareForAngleV2);
} // namespace optiling