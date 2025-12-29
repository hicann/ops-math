/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stateless_random_uniform_v2_tiling_arch35.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {

static const std::unordered_map<ge::DataType, uint32_t> OUTPUT_DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 3}};

static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_KEY = 1;
static constexpr uint16_t INPUT_IDX_COUNTER = 2;
static constexpr uint16_t INPUT_IDX_ALG = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;

ge::graphStatus StatelessRandomUniformV2Tiling::GetPlatformInfo()
{
    auto compileInfoPtr = reinterpret_cast<const StatelessRandomUniformV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (compileInfoPtr->aivNum <= 0), OP_LOGE(opName, "StatelessRandomUniformV2Tiling fail to get coreNum."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (compileInfoPtr->ubSize <= 0), OP_LOGE(opName, "ub size less than 0 Size. please check"),
        return ge::GRAPH_FAILED);
    coreNum_ = compileInfoPtr->aivNum;
    ubSize_ = compileInfoPtr->ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomUniformV2Tiling::GetShapeAttrsInfo()
{
    auto res = GetInputInfo();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    res = GetOutputInfo();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t StatelessRandomUniformV2Tiling::GetCounterSize(Algorithm alg) const
{
    if (alg == Algorithm::RNG_ALG_PHILOX) {
        return 2; // 2 if for philox
    } else if (alg == Algorithm::RNG_ALG_THREEFRY) {
        return 1;
    }
    return 2; // 2 is for philox
}

void StatelessRandomUniformV2Tiling::GetKeyFromMem(const uint64_t key)
{
    key_[0] = static_cast<uint32_t>(key);
    key_[1] = static_cast<uint32_t>(key >> 32); // 32 for lower 32 bits
}
void StatelessRandomUniformV2Tiling::GetCounterFromMem(const std::vector<uint64_t>& counter)
{
    counter_[0] = static_cast<uint32_t>(counter[0]);
    counter_[1] = static_cast<uint32_t>(counter[0] >> 32); // 32 for lower 32 bits
    counter_[2] = static_cast<uint32_t>(counter[1]);
    counter_[3] = static_cast<uint32_t>(counter[1] >> 32); // 32 for lower 32 bits
}

ge::graphStatus StatelessRandomUniformV2Tiling::GetInputKeyCounter()
{
    auto keyDesc = context_->GetInputDesc(INPUT_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    auto keyDtype = keyDesc->GetDataType();
    if (keyDtype != ge::DataType::DT_UINT64) {
        OP_LOGE(opName, "input key Dtype should be uint64, but got [%d]", keyDtype);
        return ge::GRAPH_FAILED;
    }
    auto keyShape = context_->GetInputShape(INPUT_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    if (keyShape->GetStorageShape().GetShapeSize() != 1) {
        OP_LOGE(opName, "input key number should be 1, but got [%ld]", keyShape->GetStorageShape().GetShapeSize());
        return ge::GRAPH_FAILED;
    }

    auto counterDesc = context_->GetInputDesc(INPUT_IDX_COUNTER);
    OP_CHECK_NULL_WITH_CONTEXT(context_, counterDesc);
    auto counterDtype = counterDesc->GetDataType();
    if (counterDtype != ge::DataType::DT_UINT64) {
        OP_LOGE(opName, "input counter Dtype should be uint64, but got [%d]", counterDtype);
        return ge::GRAPH_FAILED;
    }
    // input key has one uint64, Philox counter need 2 element.
    std::vector<uint64_t> counter = {0, 0};
    auto keyTensor = context_->GetInputTensor(INPUT_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyTensor);
    int32_t keyNum = keyTensor->GetShapeSize();
    OP_CHECK_IF(
        keyNum != 1, OP_LOGE(opName, "key data must be 1 tensor scalar, but get %d.", keyNum), return ge::GRAPH_FAILED);
    const uint64_t* key = keyTensor->GetData<uint64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, key);

    auto counterTensor = context_->GetInputTensor(INPUT_IDX_COUNTER);
    OP_CHECK_NULL_WITH_CONTEXT(context_, counterTensor);
    int64_t counterNum = static_cast<int64_t>(counterTensor->GetShapeSize());
    OP_CHECK_IF(
        counterNum < GetCounterSize(Algorithm(alg_)),
        OP_LOGE(
            opName, "counter tensor elements number at least %ld, but get %ld.", GetCounterSize(Algorithm(alg_)),
            counterNum),
        return ge::GRAPH_FAILED);
    const uint64_t* counterVal = counterTensor->GetData<uint64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, counterVal);
    counter[0] = counterVal[0];
    counter[1] = counterVal[1];
    OP_LOGD(opName, "key = %ld, counter value is [%lu, %lu]", key[0], counterVal[0], counterVal[1]);

    GetKeyFromMem(key[0]);
    GetCounterFromMem(counter);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomUniformV2Tiling::GetInputInfo()
{
    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto shapeValue = outputShape->GetStorageShape();
    uint32_t shapeRank = shapeValue.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        outputSize_ *= shapeValue.GetDim(idx);
    }

    auto algTensor = context_->GetInputTensor(INPUT_IDX_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, algTensor);
    if (algTensor->GetShapeSize() != 1) {
        OP_LOGE(opName, "alg data must be 1 tensor scalar, but got [%ld]", algTensor->GetShapeSize());
        return ge::GRAPH_FAILED;
    }
    const int32_t* algVal = algTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, algVal);
    alg_ = Algorithm(algVal[0]);
    if (alg_ == Algorithm::RNG_ALG_AUTO_SELECT) {
        alg_ = Algorithm::RNG_ALG_PHILOX;
    }
    OP_CHECK_IF(
        alg_ != Algorithm::RNG_ALG_PHILOX,
        OP_LOGE(
            opName, "alg only support %d, but got %d.", static_cast<int32_t>(Algorithm::RNG_ALG_PHILOX),
            static_cast<int32_t>(alg_)),
        return ge::GRAPH_FAILED);

    auto res = GetInputKeyCounter();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomUniformV2Tiling::GetOutputInfo()
{
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    outputDtype_ = outputDesc->GetDataType();
    auto iter = OUTPUT_DATA_TYPE_TO_INT.find(outputDtype_);
    if (iter != OUTPUT_DATA_TYPE_TO_INT.end()) {
        outputDtypeVal_ = iter->second;
    } else {
        OP_LOGE(opName, "output dtype = %d not supported, please check.", outputDtype_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T1, typename T2>
inline T1 StatelessRandomUniformV2Tiling::CeilDiv(const T1 a, const T2 b) const
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void StatelessRandomUniformV2Tiling::BlockTiling()
{
    outputDtypeSize_ = ge::GetSizeByDataType(outputDtype_);
    auto coreAlignFactor = CORE_ALIGN_SIZE / outputDtypeSize_;
    auto blockFactor = CeilDiv(outputSize_, coreNum_);
    auto blockAlignFactor = CeilDiv(blockFactor, coreAlignFactor) * coreAlignFactor;
    auto minTilingSize = MIN_TILING_SIZE;
    blockTilingSize_ = std::max(static_cast<uint32_t>(blockAlignFactor), minTilingSize);
    blockNum_ = CeilDiv(outputSize_, blockTilingSize_);
    tailBlockTilingSize_ = outputSize_ - blockTilingSize_ * (blockNum_ - 1);
    OP_LOGD(
        opName,
        "outputSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
        "blockTilingSize = %d, tailBlockTilingSize = %d",
        outputSize_, blockFactor, blockAlignFactor, blockTilingSize_, tailBlockTilingSize_);
    return;
}

ge::graphStatus StatelessRandomUniformV2Tiling::UbTiling()
{
    // quarterUbSize: 2 for double buffer; coefVal for temp RNG
    auto coefVal = 1;
    if (outputDtype_ == ge::DataType::DT_FLOAT16 || outputDtype_ == ge::DataType::DT_BF16) {
        coefVal = 2; // philox temp buff need int32 to float16/bfloat16
    }
    auto quarterUbSize = ubSize_ / (2 + coefVal);
    auto alignFactor = BLOCK_SIZE_BYTES / outputDtypeSize_;
    ubTilingSize_ = CeilDiv(quarterUbSize / outputDtypeSize_, alignFactor) * alignFactor;
    OP_LOGD(opName, "quarterUbSize = %u, ubTilingSize = %u", quarterUbSize, ubTilingSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomUniformV2Tiling::DoOpTiling()
{
    BlockTiling();
    ge::graphStatus res = UbTiling();
    if (res == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomUniformV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StatelessRandomUniformV2Tiling::GetTilingKey() const
{
    uint64_t tilingKey = 100;
    tilingKey += outputDtypeVal_;
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus StatelessRandomUniformV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomUniformV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum_);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void StatelessRandomUniformV2Tiling::SetTilingData()
{
    tilingData.set_blockNum(blockNum_);
    tilingData.set_blockTilingSize(blockTilingSize_);
    tilingData.set_tailBlockTilingSize(tailBlockTilingSize_);
    tilingData.set_ubTilingSize(ubTilingSize_);
    tilingData.set_alg(static_cast<uint32_t>(alg_));
    tilingData.set_key(key_);
    tilingData.set_counter(counter_);
    return;
}

ge::graphStatus Tiling4StatelessRandomUniformV2(gert::TilingContext* context)
{
    StatelessRandomUniformV2Tiling tilingObj(context);
    return tilingObj.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessRandomUniformV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "op[StatelessRandomUniformV2] tiling start.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<StatelessRandomUniformV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StatelessRandomUniformV2)
    .Tiling(Tiling4StatelessRandomUniformV2)
    .TilingParse<StatelessRandomUniformV2CompileInfo>(TilingPrepare4StatelessRandomUniformV2)
    .TilingInputsDataDependency({INPUT_IDX_KEY, INPUT_IDX_COUNTER, INPUT_IDX_ALG});

} // namespace optiling
