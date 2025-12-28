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
 * \file drop_out_v3_tiling_arch35.cpp
 * \brief
 */

#include <vector>
#include <cmath>
#include "drop_out_v3_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace optiling {

static const int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
static const int64_t CORE_MINEST_NUM = 256;
static const int64_t RESERVED_UB_SIZE = 2048;
static const int64_t DOUBLE_UB_SIZE = 2;

static const int32_t UB_CALCU_RADIO = 5;
static const int32_t UINT32_TO_UINT8_RADIO = 8;
static const int32_t MAX_DIM_NUM = 8;

static const int32_t INDEX_INPUT_X = 0;
static const int32_t INDEX_INPUT_NOISE = 1;
static const int32_t INDEX_INPUT_P = 2;
static const int32_t INDEX_INPUT_SEED = 3;
static const int32_t INDEX_INPUT_OFFSET = 4;
static const int32_t INDEX_OUTPUT_Y = 0;
static const int32_t INDEX_OUTPUT_MASK = 1;

static constexpr uint32_t RIGHT_SHIFT_NUM = 32;
static constexpr uint64_t COUNTER_IDX_0 = 0;
static constexpr uint64_t COUNTER_IDX_1 = 1;
static constexpr uint64_t COUNTER_IDX_2 = 2;
static constexpr uint64_t COUNTER_IDX_3 = 3;

static const int32_t TILING_KEY_FP32 = 1001;
static const int32_t TILING_KEY_FP16 = 1002;
static const int32_t TILING_KEY_BF16 = 1003;

uint32_t key_[ALG_KEY_SIZE] = {0, 0};
uint32_t counter_[ALG_COUNTER_SIZE] = {0, 0, 0, 0};

static inline int64_t Align256CeilSize(int64_t value)
{
    return static_cast<int64_t>((value + CORE_MINEST_NUM - 1) / CORE_MINEST_NUM * CORE_MINEST_NUM);
}

static inline int64_t Align256FloorSize(int64_t value)
{
    return static_cast<int64_t>((value / CORE_MINEST_NUM) * CORE_MINEST_NUM);
}

template <typename T>
ge::graphStatus GetIntValue(
    const gert::TilingContext* context, const gert::Tensor* constTensor, gert::Shape& constShape)
{
    const T* constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, constValue);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetIntToShape(const gert::TilingContext* context, const int64_t constIdx, gert::Shape& constShape)
{
    auto constTensor = context->GetRequiredInputTensor(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, constTensor);

    auto inputDescPtr = context->GetRequiredInputDesc(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    auto constDtype = inputDescPtr->GetDataType();

    auto ret = ge::GRAPH_FAILED;
    switch (constDtype) {
        case ge::DT_INT32:
            ret = GetIntValue<int32_t>(context, constTensor, constShape);
            break;
        case ge::DT_INT64:
            ret = GetIntValue<int64_t>(context, constTensor, constShape);
            break;
        default:
            OP_LOGD(
                context->GetNodeName(), "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
                Ops::Base::ToString(constDtype).c_str());
            return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "get const value failed, please check."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "current const value is %s", Ops::Base::ToString(constShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus DropOutV3SetTilingData(gert::TilingContext* context, DropOutV3TilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static inline bool IsSameShape(const gert::Shape& shape1, const gert::Shape& shape2)
{
    size_t inputShapeSize = shape1.GetDimNum();
    if (shape2.GetDimNum() != inputShapeSize) {
        return false;
    }
    for (size_t i = 0; i < inputShapeSize; ++i) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end)
{
    int64_t size = 1;
    for (size_t i = begin; i < end; i++) {
        size *= shape[i];
    }
    return size;
}

static inline bool IsMatchShape(const gert::Shape& shape1, const gert::Shape& shape2)
{
    int64_t inputDataNum = GetPartShapeSize(shape1, 0, shape1.GetDimNum());
    int64_t maskDataNum = GetPartShapeSize(shape2, 0, shape2.GetDimNum());
    if ((inputDataNum / UINT32_TO_UINT8_RADIO) != maskDataNum) {
        return false;
    } else {
        return true;
    }
}

void GetKeyFromMem(const int64_t key)
{
    key_[0] = static_cast<int32_t>(key);
    key_[1] = static_cast<int32_t>(key >> RIGHT_SHIFT_NUM);
}

void GetCounterFromMem(const std::vector<int64_t>& counter)
{
    counter_[COUNTER_IDX_0] = static_cast<int32_t>(counter[0]);
    counter_[COUNTER_IDX_1] = static_cast<int32_t>(counter[0] >> RIGHT_SHIFT_NUM);
    counter_[COUNTER_IDX_2] = static_cast<int32_t>(counter[1]);
    counter_[COUNTER_IDX_3] = static_cast<int32_t>(counter[1] >> RIGHT_SHIFT_NUM);
}

static ge::graphStatus CheckParamsShape(const gert::TilingContext* context)
{
    // 校验shape信息
    auto xInputShapePtr = context->GetRequiredInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputShapePtr);
    auto xInputShape = xInputShapePtr->GetStorageShape();
    auto xDimNum = xInputShape.GetDimNum();

    auto pInputShapePtr = context->GetRequiredInputShape(INDEX_INPUT_P);
    OP_CHECK_NULL_WITH_CONTEXT(context, pInputShapePtr);
    auto pShapeSize = pInputShapePtr->GetStorageShape().GetShapeSize();

    auto seedInputShapePtr = context->GetRequiredInputShape(INDEX_INPUT_SEED);
    OP_CHECK_NULL_WITH_CONTEXT(context, seedInputShapePtr);
    auto seedShapeSize = seedInputShapePtr->GetStorageShape().GetShapeSize();

    auto offsetInputShapePtr = context->GetRequiredInputShape(INDEX_INPUT_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetInputShapePtr);
    auto offsetShapeSize = offsetInputShapePtr->GetStorageShape().GetShapeSize();

    auto yOutputShapePtr = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutputShapePtr);
    auto yOutputShape = yOutputShapePtr->GetStorageShape();

    OP_CHECK_IF(
        (xDimNum > MAX_DIM_NUM),
        OP_LOGE(
            context->GetNodeName(), "%s",
            ops::ConcatString("The dim of input x should be 0~8, but got", xDimNum, "please check.").c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (seedShapeSize != 1) || (offsetShapeSize != 2),
        OP_LOGE(
            context->GetNodeName(), "%s",
            ops::ConcatString(
                "The seed=", seedShapeSize, "should be 1 and offset=", offsetShapeSize, "should be 2, please check.")
                .c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !IsSameShape(xInputShape, yOutputShape),
        OP_LOGE(context->GetNodeName(), "The shape of input x_size and y_size should have same shape, please check."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "%s", ops::ConcatString("The dim of p=", pShapeSize).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetParamsData(const gert::TilingContext* context)
{
    gert::Shape inputSeed_;
    gert::Shape inputOffset_;
    OP_CHECK_IF(
        GetIntToShape(context, INDEX_INPUT_SEED, inputSeed_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get const shape of seed failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetIntToShape(context, INDEX_INPUT_OFFSET, inputOffset_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get const shape of offset failed"), return ge::GRAPH_FAILED);
    int64_t key = static_cast<int64_t>(inputSeed_[0]);
    std::vector<int64_t> counter = {inputOffset_[0], inputOffset_[1]};
    GetKeyFromMem(key);
    GetCounterFromMem(counter);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParamsIsValid(const gert::TilingContext* context)
{
    auto xInputPtr = context->GetRequiredInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputPtr);
    auto xInputType = xInputPtr->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(xInputType);

    auto pInputPtr = context->GetRequiredInputDesc(INDEX_INPUT_P);
    OP_CHECK_NULL_WITH_CONTEXT(context, pInputPtr);
    auto pInputType = pInputPtr->GetDataType();

    auto seedInputPtr = context->GetRequiredInputDesc(INDEX_INPUT_SEED);
    OP_CHECK_NULL_WITH_CONTEXT(context, seedInputPtr);
    auto seedInputType = seedInputPtr->GetDataType();

    auto offsetInputPtr = context->GetRequiredInputDesc(INDEX_INPUT_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetInputPtr);
    auto offsetInputType = offsetInputPtr->GetDataType();

    OP_CHECK_IF(
        (pInputType != ge::DT_FLOAT) && (pInputType != ge::DT_FLOAT16) && (pInputType != ge::DT_BF16),
        OP_LOGE(
            context->GetNodeName(), "The type of p should be FP32/FP16/BF16, but got %s, please check.",
            Ops::Base::ToString(pInputType).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (seedInputType != ge::DT_INT64) && (seedInputType != ge::DT_INT32),
        OP_LOGE(
            context->GetNodeName(), "The type of seed should be INT32/INT64, but got %s, please check.",
            Ops::Base::ToString(seedInputType).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        offsetInputType != ge::DT_INT64,
        OP_LOGE(
            context->GetNodeName(), "The type of offset should be INT64, but got %s, please check.",
            Ops::Base::ToString(offsetInputType).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (xInputType != ge::DT_FLOAT) && (xInputType != ge::DT_FLOAT16) && (xInputType != ge::DT_BF16),
        OP_LOGE(
            context->GetNodeName(), "The type of input should be FP32/FP16/BF16, but got %s, please check.",
            Ops::Base::ToString(xInputType).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (dtypeSize != 2) && (dtypeSize != 4),
        OP_LOGE(context->GetNodeName(), "The dtypeSize of input should be 2 or 4, please check."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckParamsShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The param of shape is invalid, please check."), return ge::GRAPH_FAILED);

    auto pSizeTensor = context->GetRequiredInputTensor(INDEX_INPUT_P);
    OP_CHECK_NULL_WITH_CONTEXT(context, pSizeTensor);
    const float* pSizeVal = pSizeTensor->GetData<float>();
    OP_CHECK_NULL_WITH_CONTEXT(context, pSizeVal);
    OP_CHECK_IF(
        (pSizeVal[0] <= 0) || (pSizeVal[0] >= 1),
        OP_LOGE(context->GetNodeName(), "Input of p [%f] should between 0~1, please check.", pSizeVal[0]),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetParamsData(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get param data failed, please check."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcTilingKey(const DataType dataType, DropOutV3TilingData& tilingData)
{
    if (dataType == DT_FLOAT) {
        tilingData.set_tilingKey(TILING_KEY_FP32);
    } else if (dataType == DT_FLOAT16) {
        tilingData.set_tilingKey(TILING_KEY_FP16);
    } else if (dataType == DT_BF16) {
        tilingData.set_tilingKey(TILING_KEY_BF16);
    }
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcShareTmpBuffer(DropOutV3TilingData& tilingData)
{
    uint32_t typeSize = 2;
    std::vector<int64_t> srcDims = {1, RESERVED_UB_SIZE, RESERVED_UB_SIZE};
    ge::Shape shape(srcDims);

    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    AscendC::GetDropOutMaxMinTmpSize(shape, typeSize, false, maxSize, minSize);
    if (maxSize == 0) {
        OP_LOGD("[DropOutV3]", "Tiling GetDropOutMaxMinTmpSize fail.");
        maxSize = RESERVED_UB_SIZE;
    }
    uint32_t shareTmpSize = (maxSize > RESERVED_UB_SIZE) ? RESERVED_UB_SIZE : maxSize;
    int64_t shareTmpSizeInt64 = static_cast<int64_t>(shareTmpSize);
    tilingData.set_shareTmpUbSize(shareTmpSizeInt64);

    return ge::GRAPH_SUCCESS;
}

static inline void TilingDataToLogging(DropOutV3TilingData& tilingData)
{
    OP_LOGI(
        "[DropOutV3]",
        "[usedCoreNum]: %ld, [totalCoreNum]: %ld, [hardWareUbSize]: %ld, [numOfPerCore]: %ld,  \
[numOfTailCore]: %ld, [ubOneLoopNum]: %ld, [loopOfPerCore]: %ld, [perOfPerCore]: %ld, [tailOfPerCore]: %ld,    \
[loopOfTailCore]: %ld, [perOfTailCore]: %ld, [tailOfTailCore]: %ld, [workspaceSize]: %ld, [tilingKey]: %ld, [key]: %s, [counter]: %s",
        tilingData.get_usedCoreNum(), tilingData.get_totalCoreNum(), tilingData.get_hardWareUbSize(),
        tilingData.get_numOfPerCore(), tilingData.get_numOfTailCore(), tilingData.get_ubOneLoopNum(),
        tilingData.get_loopOfPerCore(), tilingData.get_perOfPerCore(), tilingData.get_tailOfPerCore(),
        tilingData.get_loopOfTailCore(), tilingData.get_perOfTailCore(), tilingData.get_tailOfTailCore(),
        tilingData.get_workspaceSize(), tilingData.get_tilingKey(),
        ops::ToStringWithSize(tilingData.get_key(), ALG_KEY_SIZE).c_str(),
        ops::ToStringWithSize(tilingData.get_counter(), ALG_COUNTER_SIZE).c_str());
}

static ge::graphStatus CalcDropOutV3Tiling(
    const gert::TilingContext* context, DropOutV3TilingData& tilingData, DataType dataType)
{
    OP_LOGD(context->GetNodeName(), "TilingDropOutV3 Enter CalcDropOutV3Tiling.");

    auto xInputShapePtr = context->GetRequiredInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputShapePtr);
    auto xInputShape = xInputShapePtr->GetStorageShape();

    auto& elementShape = Ops::Math::OpTiling::EnsureNotScalar(xInputShape);
    int64_t dimNum = elementShape.GetDimNum();
    int64_t elementNum = (dimNum == 0) ? 1 : GetPartShapeSize(elementShape, 0, dimNum);
    OP_LOGD(context->GetNodeName(), "Input element num is %ld.", elementNum);

    int64_t numOfPerCore = elementNum;
    int64_t usedCoreNum = 1;
    if (elementNum > CORE_MINEST_NUM) {
        numOfPerCore =
            Align256CeilSize((elementNum + tilingData.get_totalCoreNum() - 1) / tilingData.get_totalCoreNum());
        usedCoreNum = min((elementNum + numOfPerCore - 1) / numOfPerCore, tilingData.get_totalCoreNum());
    }
    int64_t numOfTailCore = elementNum - (usedCoreNum - 1) * numOfPerCore;

    int64_t ubCap = tilingData.get_hardWareUbSize() / DOUBLE_UB_SIZE;
    int64_t ubNum = (ubCap - tilingData.get_shareTmpUbSize()) / ge::GetSizeByDataType(dataType);
    int64_t ubOneLoopNum = 1;
    if (dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        ubOneLoopNum = Align256FloorSize(ubNum / UB_CALCU_RADIO);
    } else {
        return ge::GRAPH_FAILED;
    }

    int64_t perOfPerCore = (numOfPerCore < ubOneLoopNum) ? numOfPerCore : ubOneLoopNum;
    int64_t loopOfPerCore = Ops::Base::CeilDiv(numOfPerCore, perOfPerCore);
    int64_t tailOfPerCore = numOfPerCore - perOfPerCore * (loopOfPerCore - 1);
    int64_t perOfTailCore = (numOfTailCore < ubOneLoopNum) ? numOfTailCore : ubOneLoopNum;
    int64_t loopOfTailCore = Ops::Base::CeilDiv(numOfTailCore, perOfTailCore);
    int64_t tailOfTailCore = numOfTailCore - perOfTailCore * (loopOfTailCore - 1);

    tilingData.set_numOfPerCore(numOfPerCore);
    tilingData.set_loopOfPerCore(loopOfPerCore);
    tilingData.set_perOfPerCore(perOfPerCore);
    tilingData.set_tailOfPerCore(tailOfPerCore);

    tilingData.set_numOfTailCore(numOfTailCore);
    tilingData.set_loopOfTailCore(loopOfTailCore);
    tilingData.set_perOfTailCore(perOfTailCore);
    tilingData.set_tailOfTailCore(tailOfTailCore);
    tilingData.set_ubOneLoopNum(ubOneLoopNum);
    tilingData.set_usedCoreNum(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4DropOutV3(gert::TilingContext* context)
{
    OP_LOGD(context, "[DropOutV3] Tiling4DropOutV3 running begin.");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Context should not be nullptr."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckParamsIsValid(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Tiling4DropOutV3 check input params islegal."), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const DropOutV3CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    int64_t totalCoreNum = static_cast<int64_t>(compileInfo->totalCoreNum);

    // set ubSize
    int64_t ubSize = compileInfo->ubSizePlatForm;
    // 实例化对象op
    DropOutV3TilingData tilingData;
    // 设置totalCoreNum && hardWareUbSize
    tilingData.set_totalCoreNum(totalCoreNum);
    tilingData.set_hardWareUbSize(ubSize);

    // 设置key && counter
    tilingData.set_key(key_);
    tilingData.set_counter(counter_);

    // 设置shareTmpBuffer
    OP_CHECK_IF(
        CalcShareTmpBuffer(tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "[CalcShareTmpBuffer] TilingDropOutV3 fail to get share buffer."),
        return ge::GRAPH_FAILED);

    // InputDesc判空 && 设置tilingKey
    auto inputDesc = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        CalcTilingKey(dataType, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "[CalcTilingKey] TilingDropOutV3 fail to set tiling key."),
        return ge::GRAPH_FAILED);

    // 设置DropOutV3计算中的参数
    OP_CHECK_IF(
        CalcDropOutV3Tiling(context, tilingData, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "[CalcDropOutV3Tiling] TilingDropOutV3 fail to calulate tiling param."),
        return ge::GRAPH_FAILED);

    // 设置workspace
    size_t userSize = tilingData.get_totalCoreNum() * sizeof(int32_t);
    size_t sysWorkspaceSize = WORKSPACE_SIZE;
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = userSize + sysWorkspaceSize;
    tilingData.set_workspaceSize(userWorkspaceSize[0]);

    OP_CHECK_IF(
        DropOutV3SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "[DropOutV3SetTilingData] TilingDropOutV3 fail to set tiling data."),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    TilingDataToLogging(tilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDropOutV3(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DropOutV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepareForDropOutV3 fail to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepareForDropOutV3 fail to get ub size."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DropOutV3)
    .TilingInputsDataDependency({INDEX_INPUT_P, INDEX_INPUT_SEED, INDEX_INPUT_OFFSET})
    .Tiling(Tiling4DropOutV3)
    .TilingParse<DropOutV3CompileInfo>(TilingPrepareForDropOutV3);

} // namespace optiling
