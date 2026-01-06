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
 * \file as_strided_tiling_arch35.cpp
 * \brief as_strided_tiling_arch35
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "util/const_util.h"
#include "register/op_def_registry.h"
#include "as_strided_helper_tiling.h"
#include "as_strided_tiling_arch35.h"
#include "util/math_util.h"

using namespace std;
using namespace ge;

namespace optiling {
constexpr size_t IN_SIZE = 1;
constexpr size_t IN_STRIDE = 2;
constexpr size_t IN_OFFSET = 3;
constexpr size_t UB_BUFFER_B8 = 126976;
constexpr size_t UB_BUFFER_B16 = 63488;
constexpr size_t UB_BUFFER_B32 = 31744;
constexpr size_t UB_BUFFER_B64 = 15872;
constexpr size_t INPUT_DTYPE_B64 = 8;
constexpr size_t INPUT_DTYPE_B32 = 4;
constexpr size_t INPUT_DTYPE_B16 = 2;
constexpr size_t INPUT_DTYPE_B8 = 1;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr size_t MOVEALIGN_DIM5 = 5;
constexpr size_t MOVEALIGN_DIM4 = 4;
constexpr size_t MOVEALIGN_DIM3 = 3;
constexpr size_t MOVEALIGN_DIM2 = 2;
constexpr size_t MOVEALIGN_KEY = 100;
constexpr int64_t MOVEALIGN_FLAG = 128;
constexpr int64_t MOVEALIGN_STRIDE_CON = 32;
constexpr size_t DUAL_CUT_KEY = 200;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DUAL_CUT_CONDITION = 64;
constexpr int32_t VALID_DIM = 8;

std::map<ge::DataType, uint32_t> tilingTypeKeyMap = {
    {ge::DT_INT64, INPUT_DTYPE_B64},      {ge::DT_UINT64, INPUT_DTYPE_B64},       {ge::DT_COMPLEX64, INPUT_DTYPE_B64},
    {ge::DT_FLOAT, INPUT_DTYPE_B32},      {ge::DT_INT32, INPUT_DTYPE_B32},        {ge::DT_UINT32, INPUT_DTYPE_B32},
    {ge::DT_COMPLEX32, INPUT_DTYPE_B32},  {ge::DT_FLOAT16, INPUT_DTYPE_B16},      {ge::DT_BF16, INPUT_DTYPE_B16},
    {ge::DT_INT16, INPUT_DTYPE_B16},      {ge::DT_UINT16, INPUT_DTYPE_B16},       {ge::DT_UINT8, INPUT_DTYPE_B8},
    {ge::DT_INT8, INPUT_DTYPE_B8},        {ge::DT_BOOL, INPUT_DTYPE_B8},          {ge::DT_HIFLOAT8, INPUT_DTYPE_B8},
    {ge::DT_FLOAT8_E5M2, INPUT_DTYPE_B8}, {ge::DT_FLOAT8_E4M3FN, INPUT_DTYPE_B8},
};

inline static ge::graphStatus AsStridedSetTilingData(gert::TilingContext* context, AsStridedTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam)
{
    tiling.set_axisOutTotalFactor(tilingParam.axisOutTotalFactor);
    tiling.set_outerAxisFactor(tilingParam.outerAxisFactor);
    tiling.set_innerAxisFactor(tilingParam.innerAxisFactor);
    tiling.set_tilingAxisIdx(tilingParam.tilingAxisIdx);
    tiling.set_outerAxisNum(tilingParam.outerAxisNum);
    tiling.set_outStrideArr(tilingParam.outStrideArr);
    tiling.set_nddmaSrcStride(tilingParam.nddmaSrcStride);
    tiling.set_ubSize(tilingParam.ubSize);
    tiling.set_innerAxisNum(tilingParam.innerAxisNum);
    tiling.set_storageOffset(tilingParam.storageOffset);
    tiling.set_blockNum(tilingParam.blockNum);
    tiling.set_innerAxisFactorTail(tilingParam.innerAxisFactorTail);
    tiling.set_ubFactorTail(tilingParam.ubFactorTail);
    tiling.set_loopsPerCore(tilingParam.loopsPerCore);
    tiling.set_nddmaLoop(tilingParam.nddmaLoop);
    tiling.set_nddmaDstStride(tilingParam.nddmaDstStride);
    tiling.set_innerAxis(tilingParam.innerAxis);
    tiling.set_outLoopArr(tilingParam.outLoopArr);
    tiling.set_nddmaTailLoop(tilingParam.nddmaTailLoop);
    tiling.set_en32BAligned(tilingParam.en32BAligned);
    tiling.set_gmInStride(tilingParam.gmInStride);
    tiling.set_gmShape(tilingParam.gmShape);
}

static void NoTilingMergeAxis(AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam, gert::Shape outSize)
{
    OP_LOGD("As_Strided", "NoTilingMergeAxis");
    tilingParam.ubUseFactor = 1;
    tilingParam.loopsPerCore = 1;

    tilingParam.loopsPerCore = (tilingParam.axisOutTotalFactor + tilingParam.blockNum - 1) / tilingParam.blockNum;

    for (size_t i = 0; i < tilingParam.outerAxisNum; i++) {
        tilingParam.outLoopArr[i + TILING_ARRAY_LEN - tilingParam.outerAxisNum] = outSize[i];
    }

    for (int32_t i = TILING_NDDMA_LEN - 2; i >= 0; i--) {
        tilingParam.nddmaDstStride[i] = outSize[i + tilingParam.outerAxisNum + 1] * tilingParam.nddmaDstStride[i + 1];
    }

    for (uint32_t i = 0; i < TILING_NDDMA_LEN; i++) {
        tilingParam.nddmaLoop[i] = outSize[i + tilingParam.outerAxisNum];
        tilingParam.ubUseFactor *= outSize[i + tilingParam.outerAxisNum];
    }

    tiling.set_ubFactor(tilingParam.ubUseFactor);
}

static void MergeAxisAfterTiling(
    [[maybe_unused]] const AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam, gert::Shape outSize)
{
    OP_LOGD("As_Strided", "Start fusing");
    tilingParam.axisOutTotalFactor = tilingParam.outerAxisFactor;
    for (uint32_t i = 0; i < tilingParam.tilingAxisIdx; i++) {
        tilingParam.axisOutTotalFactor *= outSize[i];
    }
    tilingParam.loopsPerCore = 1;
    tilingParam.loopsPerCore = (tilingParam.axisOutTotalFactor + tilingParam.blockNum - 1) / tilingParam.blockNum;

    if (tilingParam.tilingFlag == 1) { // Axis tiling is complete.
        for (uint32_t i = 0; i < tilingParam.outerAxisNum - 1; i++) {
            tilingParam.outLoopArr[i + TILING_ARRAY_LEN - tilingParam.outerAxisNum] = outSize[i];
        }
        tilingParam.outLoopArr[TILING_ARRAY_LEN - 1] = tilingParam.outerAxisFactor;
    }

    for (int32_t i = tilingParam.innerAxisNum - 1; i > 0; i--) {
        tilingParam.innerAxis[i] = outSize[tilingParam.tilingAxisIdx + i];
    }
    tilingParam.innerAxis[0] = tilingParam.innerAxisFactor;

    if (tilingParam.innerAxisNum >= TILING_NDDMA_LEN) {
        for (int32_t i = TILING_NDDMA_LEN - 2; i >= 0; i--) {
            tilingParam.nddmaDstStride[i] = tilingParam.innerAxis[i + 1] * tilingParam.nddmaDstStride[i + 1];
        }
        for (uint32_t i = 0; i < TILING_NDDMA_LEN; i++) {
            tilingParam.nddmaLoop[i] = tilingParam.innerAxis[i + tilingParam.innerAxisNum - TILING_NDDMA_LEN];
            tilingParam.nddmaTailLoop[i] = tilingParam.innerAxis[i + tilingParam.innerAxisNum - TILING_NDDMA_LEN];
            tilingParam.nddmaTailLoop[0] = tilingParam.innerAxisFactorTail;
        }
    } else {
        for (uint32_t i = 1; i < tilingParam.innerAxisNum; i++) {
            tilingParam.nddmaDstStride[TILING_NDDMA_LEN - 1 - i] =
                tilingParam.innerAxis[tilingParam.innerAxisNum - i] *
                tilingParam.nddmaDstStride[TILING_NDDMA_LEN - 1 - i + 1];
        }
        for (uint32_t i = 0; i < tilingParam.innerAxisNum; i++) {
            tilingParam.nddmaLoop[i + TILING_NDDMA_LEN - tilingParam.innerAxisNum] = tilingParam.innerAxis[i];
            tilingParam.nddmaTailLoop[i + TILING_NDDMA_LEN - tilingParam.innerAxisNum] = tilingParam.innerAxis[i];
            tilingParam.nddmaTailLoop[TILING_NDDMA_LEN - tilingParam.innerAxisNum] = tilingParam.innerAxisFactorTail;
        }
    }
}

static void MergeAxis4MoveAlign(
    AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride, AsStridedTilingData& tiling)
{
    OP_LOGD("As_Strided", "MergeAxis4MoveAlign");
    tilingParam.axisOutTotalFactor = 1;
    tilingParam.ubUseFactor = 1;
    if (tilingParam.innerAxisNum > MOVEALIGN_DIM5) {
        tilingParam.outerAxisNum += 1;
    }
    for (uint32_t i = 0; i < tilingParam.outerAxisNum; i++) {
        tilingParam.axisOutTotalFactor *= outSize[i];
        tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.outerAxisNum + i] = outStride[i];
    }
    tilingParam.blockNum =
        tilingParam.axisOutTotalFactor > tilingParam.numCore ? tilingParam.numCore : tilingParam.axisOutTotalFactor;
    OP_LOGD("As_Strided", "BlockNum: %u", tilingParam.blockNum);

    tilingParam.loopsPerCore = (tilingParam.axisOutTotalFactor + tilingParam.blockNum - 1) / tilingParam.blockNum;

    for (size_t i = 0; i < tilingParam.outerAxisNum; i++) {
        tilingParam.outLoopArr[i + TILING_ARRAY_LEN - tilingParam.outerAxisNum] = outSize[i];
    }

    for (uint32_t i = 1; i < TILING_NDDMA_LEN; i++) {
        tilingParam.ubUseFactor *= outSize[i + tilingParam.outerAxisNum - 1];
    }
    tiling.set_ubFactor(tilingParam.ubUseFactor);
    if ((tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
        tilingParam.nddmaDstStride[MOVEALIGN_DIM2] =
            Ops::Base::CeilDiv(tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
            UB_ALIGN_SIZE;
        tilingParam.nddmaDstStride[1] =
            tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.nddmaLoop[MOVEALIGN_DIM2];
        tilingParam.en32BAligned = 1;
    } else if ((tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
        tilingParam.nddmaDstStride[1] =
            Ops::Base::CeilDiv(tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        tilingParam.en32BAligned = 1;
    }
}

inline static bool HasDuplicate(gert::Shape outStride)
{
    // MoveAlign Condition 3
    int32_t numStride[TILING_ARRAY_LEN] = {0};
    for (uint32_t i = 0; i < outStride.GetDimNum(); i++) {
        numStride[i] = outStride[i];
    }
    std::sort(numStride, numStride + outStride.GetDimNum());
    for (uint32_t i = 1; i < outStride.GetDimNum(); i++) {
        if (numStride[i] == numStride[i - 1]) {
            return true;
        }
    }
    return false;
}

inline static bool CalStrideRange(gert::Shape outStride, const AsStridedTilingParam& tilingParam)
{
    // MoveAlign Condition 2
    int64_t minVal = outStride[0];
    int64_t maxVal = outStride[0];
    for (uint32_t i = 0; i < outStride.GetDimNum(); i++) {
        if (outStride[i] < minVal) {
            minVal = outStride[i];
        }
        if (outStride[i] > maxVal) {
            maxVal = outStride[i];
        }
    }
    if ((maxVal - minVal) * tilingParam.sizeofDtype < MOVEALIGN_STRIDE_CON) {
        return false;
    }
    return true;
}

inline static bool CheckLastDim(gert::Shape outSize, gert::Shape outStride, const AsStridedTilingParam& tilingParam)
{
    // MoveAlign Condition 1
    auto outShapeSize = outSize.GetDimNum();
    if (((outSize[outShapeSize - 1] * tilingParam.sizeofDtype) >= MOVEALIGN_FLAG) &&
        (outStride[outShapeSize - 1] == 1)) {
        return true;
    }
    return false;
}

/*
 * MoveAlign Condition:
 * 1. The size of last dim is larger than 128 and its stride is 1.
 * 2. The outStride range is larger than 32B.
 * 3. The outStride doesn' t have duplicate value.
 */
inline static bool IsMoveAlign(gert::Shape outSize, gert::Shape outStride, AsStridedTilingParam& tilingParam)
{
    if (CheckLastDim(outSize, outStride, tilingParam) && CalStrideRange(outStride, tilingParam) &&
        (!HasDuplicate(outStride))) {
        OP_LOGD("As_Strided", "Need MoveAlign");
        return true;
    }
    return false;
}

inline static void MoveAlignForAsStrided(
    AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride, AsStridedTilingData& tiling)
{
    if (tilingParam.innerAxisNum == MOVEALIGN_DIM3) {
        if ((tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
            tilingParam.nddmaDstStride[MOVEALIGN_DIM2] =
                Ops::Base::CeilDiv(
                    tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
                UB_ALIGN_SIZE;
            tilingParam.nddmaDstStride[1] = 0;
            tilingParam.en32BAligned = 1;
        }
    } else if (tilingParam.innerAxisNum == MOVEALIGN_DIM4) {
        if ((tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
            tilingParam.nddmaDstStride[MOVEALIGN_DIM2] =
                Ops::Base::CeilDiv(
                    tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
                UB_ALIGN_SIZE;
            tilingParam.nddmaDstStride[1] =
                tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.nddmaLoop[MOVEALIGN_DIM2];
            tilingParam.en32BAligned = 1;
        } else if ((tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
            tilingParam.nddmaDstStride[1] =
                Ops::Base::CeilDiv(tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
                UB_ALIGN_SIZE;
            tilingParam.en32BAligned = 1;
        }
    } else if (tilingParam.innerAxisNum >= MOVEALIGN_DIM5) {
        MergeAxis4MoveAlign(tilingParam, outSize, outStride, tiling);
    }
}

inline static bool CheckDualCut(gert::Shape& outStride, const AsStridedTilingParam& tilingParam)
{
    // 条件1：最小的OutStrided 小于64Byte
    // 条件2：单切分內沒有包含所有小于等于64Byte的轴
    // 条件3：单切分內的轴包含了所有小于等于64Byte的轴，最小的轴沒有完全放进UB，且该维度小于等于128Byte
    int dimNums = outStride.GetDimNum();
    uint32_t minimumStrideAxisIdx = -1;
    int64_t minimumStridedByte = 0xFFFFFFFF;
    int64_t byteStrides[dimNums];
    for (int i = 0; i < dimNums; i++) {
        byteStrides[i] = outStride.GetDim(i) * tilingParam.sizeofDtype;
        if (minimumStridedByte > byteStrides[i]) {
            minimumStridedByte = byteStrides[i];
            minimumStrideAxisIdx = i;
        }
    }
    OP_CHECK_IF(
        (tilingParam.sizeofDtype * minimumStridedByte >= DUAL_CUT_CONDITION),
        OP_LOGW("CheckDualCut", "Case#1: minimumStride >= 64 Byte, no need to do dual cut!"), return false);

    for (uint32_t i = 0; i < tilingParam.tilingAxisIdx; i++) {
        OP_CHECK_IF(
            (byteStrides[i] <= DUAL_CUT_CONDITION),
            OP_LOGW("CheckDualCut", "Case#2: Sole cut outer axis have axis stride smaller than 64 Byte, do dual cut"),
            return true);
    }
    OP_CHECK_IF(
        ((minimumStrideAxisIdx == tilingParam.tilingAxisIdx) && (minimumStridedByte <= 128)),
        OP_LOGW("CheckDualCut", "Case#3: Minimum Stride Axis cutted by Sole cut and smaller than 128B, do dual cut"),
        return true);

    return false;
}

inline static void SetTilingDataForDualCutting(AsStridedTilingParam& tilingParam, DualCutAxisSeeker& seeker)
{
    for (int i = 0; i < TILING_ARRAY_LEN; i++) {
        tilingParam.gmShape[i] = seeker.gmShape[i];
    }
    for (int j = 0; j < TILING_ARRAY_LEN; j++) {
        tilingParam.gmInStride[j] = seeker.gmInStride[j];
    }
    for (int k = 0; k < TILING_ARRAY_LEN; k++) {
        tilingParam.gmOutStride[k] = seeker.gmOutStride[k];
    }

    for (int i = 0; i < TILING_NDDMA_LEN; i++) {
        tilingParam.nddmaLoop[i] = seeker.ubShape[i];
        tilingParam.nddmaSrcStride[i] = seeker.ubInStride[i];
        tilingParam.nddmaDstStride[i] = seeker.ubOutStride[i];
    }

    tilingParam.axisOutTotalFactor = seeker.cutAxisNums;
    tilingParam.outerAxisNum = seeker.outerAxisNums;
    tilingParam.innerAxisNum = seeker.innerAxisNums;
    tilingParam.blockNum = seeker.useCoreNum;
    tilingParam.loopsPerCore = seeker.dimsPerCore;
    tilingParam.innerAxisFactorTail = seeker.dimsPerCoreTail;

    tilingParam.nddmaTailLoop[0] = seeker.cutAxisIdx01;
    tilingParam.nddmaTailLoop[1] = seeker.cutAxisIdx02;
    // 2 means the second nature number
    tilingParam.nddmaTailLoop[2] = seeker.cutAxisTail01;
    // 3 means the third nature number
    tilingParam.nddmaTailLoop[3] = seeker.cutAxisTail02;
}

ge::graphStatus NDDMAForAsStrided(
    gert::TilingContext* context, AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride,
    AsStridedTilingData& tiling)
{
    OP_LOGD(context, "Enter SingleTilingForAsStrided");

    auto outShapeSize = outSize.GetDimNum();
    auto outStrideSize = outStride.GetDimNum();
    if (outStrideSize > TILING_NDDMA_LEN) {
        for (uint32_t i = 0; i < TILING_NDDMA_LEN; i++) {
            tilingParam.nddmaSrcStride[i] = outStride[i + outStrideSize - TILING_NDDMA_LEN];
        }
    } else {
        for (uint32_t i = 0; i < outStrideSize; i++) {
            tilingParam.nddmaSrcStride[i + TILING_NDDMA_LEN - outStrideSize] = outStride[i];
        }
    }
    // to find tiling axis
    for (int32_t i = outShapeSize - 1; i >= 0; i--) {
        tilingParam.curAxisFactor = outSize[i] * tilingParam.preSize;
        if (tilingParam.curAxisFactor >= tilingParam.ubSize) {
            tilingParam.tilingAxisIdx = i;
            tilingParam.outerAxisNum = i + 1;
            tilingParam.innerAxisNum = outShapeSize - tilingParam.outerAxisNum + 1;
            if (tilingParam.innerAxisNum > TILING_NDDMA_LEN) {
                OP_LOGD("As_Strided", "NDDMA max axis is 5! So there is no need to tile!");
                tilingParam.outerAxisNum = outShapeSize - TILING_NDDMA_LEN;
                tilingParam.axisOutTotalFactor = 1;
                for (uint32_t j = 0; j < tilingParam.outerAxisNum; j++) {
                    tilingParam.axisOutTotalFactor *= outSize[j];
                    tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.outerAxisNum + j] = outStride[j];
                }
                tilingParam.outerAxisFactor = 1;
                tilingParam.blockNum = tilingParam.axisOutTotalFactor > tilingParam.numCore ?
                                           tilingParam.numCore :
                                           tilingParam.axisOutTotalFactor;
                NoTilingMergeAxis(tiling, tilingParam, outSize);
                tilingParam.tilingFlag = 1;
                break;
            }
            // 0.9: Priority schemes with average UB usage >90%
            for (uint32_t j = tilingParam.ubSize; j >= (tilingParam.ubSize * 0.9); j--) {
                if ((tilingParam.curAxisFactor % j) == 0) {
                    tilingParam.outerAxisFactor = tilingParam.curAxisFactor / j;
                    if (outSize[i] % tilingParam.outerAxisFactor == 0) {
                        OP_LOGD("As_Strided", "Can be total tiling");
                        tilingParam.innerAxisFactor = outSize[i] / tilingParam.outerAxisFactor;
                        tilingParam.tilingFlag = 1;
                        tilingParam.ubFactor = j;
                        tiling.set_ubFactor(tilingParam.ubFactor);
                        break;
                    }
                }
            }

            if (tilingParam.tilingFlag == 0) {
                OP_LOGD("As_Strided", "There is no total tiling!");
                tilingParam.innerAxisFactor = tilingParam.ubSize / tilingParam.preSize;
                tilingParam.outerAxisFactor =
                    (outSize[i] + tilingParam.innerAxisFactor - 1) / tilingParam.innerAxisFactor;
                tilingParam.innerAxisFactorTail =
                    (tilingParam.innerAxisFactor * tilingParam.outerAxisFactor == outSize[i]) ?
                        0 :
                        outSize[i] - tilingParam.innerAxisFactor * (tilingParam.outerAxisFactor - 1);
                tilingParam.ubFactor = tilingParam.preSize * tilingParam.innerAxisFactor;
                tilingParam.ubFactorTail = tilingParam.preSize * tilingParam.innerAxisFactorTail;
                tilingParam.tilingFlag = 1;
                tiling.set_ubFactor(tilingParam.ubFactor);
            }
            tilingParam.axisOutTotalFactor = tilingParam.outerAxisFactor;
            for (uint32_t j = 0; j < tilingParam.tilingAxisIdx; j++) {
                tilingParam.axisOutTotalFactor *= outSize[j];
            }
            tilingParam.blockNum = tilingParam.axisOutTotalFactor > tilingParam.numCore ?
                                       tilingParam.numCore :
                                       tilingParam.axisOutTotalFactor;
            MergeAxisAfterTiling(tiling, tilingParam, outSize);

            for (uint32_t j = 0; j <= tilingParam.tilingAxisIdx; j++) {
                tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.tilingAxisIdx - 1 + j] = outStride[j];
            }
            if (tilingParam.innerAxisNum <= TILING_NDDMA_LEN) {
                tilingParam.outStrideArr[TILING_ARRAY_LEN - 1] *= tilingParam.innerAxisFactor;
            }
            break;
        } else {
            tilingParam.preSize *= outSize[i];
        }
    }

    if (tilingParam.tilingFlag == 0) {
        if (outShapeSize > TILING_NDDMA_LEN) {
            OP_LOGD("As_Strided", "NDDMA max axis is 5! So all is no need to tile!");
            tilingParam.outerAxisNum = outShapeSize - TILING_NDDMA_LEN;
            tilingParam.axisOutTotalFactor = 1;
            for (uint32_t i = 0; i <= tilingParam.outerAxisNum - 1; i++) {
                tilingParam.axisOutTotalFactor *= outSize[i];
                tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.outerAxisNum + i] = outStride[i];
            }
            tilingParam.outerAxisFactor = 1;
            tilingParam.innerAxisNum = outShapeSize;
            tilingParam.blockNum = tilingParam.axisOutTotalFactor > tilingParam.numCore ?
                                       tilingParam.numCore :
                                       tilingParam.axisOutTotalFactor;
            NoTilingMergeAxis(tiling, tilingParam, outSize);
        } else {
            OP_LOGD("As_Strided", "No need to tile!");
            tilingParam.axisOutTotalFactor = 1;
            tilingParam.outerAxisFactor = 1;
            tilingParam.outerAxisNum = 1;
            tilingParam.innerAxisFactor = outSize[0];
            tilingParam.tilingAxisIdx = 0;
            tilingParam.innerAxisNum = outShapeSize;
            tilingParam.blockNum = 1;
            MergeAxisAfterTiling(tiling, tilingParam, outSize);
            tiling.set_ubFactor(tilingParam.curAxisFactor);
        }
    }

    tilingParam.movealignFlag = IsMoveAlign(outSize, outStride, tilingParam);
    OP_LOGD("As_Strided", "MovealignFlag: %u", tilingParam.movealignFlag);
    if (tilingParam.movealignFlag) {
        MoveAlignForAsStrided(tilingParam, outSize, outStride, tiling);
        tilingParam.tilingKey += MOVEALIGN_KEY;
        return ge::GRAPH_SUCCESS;
    }

    tilingParam.dualCutFlag = CheckDualCut(outStride, tilingParam);
    bool dualFlag = (tilingParam.dualCutFlag) && (tilingParam.numCore > 0) && (tilingParam.tilingFlag != 0);
    OP_LOGI(
        "As_Strided", "dualFlag: %d, dualCutFlag = %d, numCore = %u, tilingFlag = %u", dualFlag,
        tilingParam.dualCutFlag, tilingParam.numCore, tilingParam.tilingFlag);
    if (dualFlag) {
        int64_t shape[outShapeSize];
        int64_t strides[outShapeSize];
        for (uint32_t i = 0; i < outShapeSize; i++) {
            shape[i] = outSize.GetDim(i);
            strides[i] = outStride.GetDim(i);
        }

        DualCutAxisSeeker seeker(shape, strides, outShapeSize, tilingParam.sizeofDtype);
        bool cutSuccess = seeker.FindDualCutAxis(tilingParam.ubSizePlatForm, BUFFER_NUM);
        if (cutSuccess) {
            seeker.GenTilingData();
            seeker.ComputeBlockTiling(tilingParam.numCore);
            seeker.PrintDebug();
            tilingParam.tilingKey = DUAL_CUT_KEY;
            SetTilingDataForDualCutting(tilingParam, seeker);

            tiling.set_gmOutStride(tilingParam.gmOutStride);

            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool CheckInputInfo(gert::Shape outSize, gert::Shape outStride, const gert::Shape& xShape, int64_t storageOffset)
{
    uint32_t requiredStorageSize = 0;
    uint32_t originalTensorStorageSize = 1;
    for (size_t i = 0; i < outSize.GetDimNum(); i++) {
        requiredStorageSize += (outSize[i] - 1) * outStride[i];
    }
    for (uint32_t i = 0; i < xShape.GetDimNum(); i++) {
        originalTensorStorageSize *= xShape.GetDim(i);
    }
    if ((storageOffset + requiredStorageSize) >= originalTensorStorageSize) {
        return false;
    }
    return true;
}

ge::graphStatus TilingForAsStridedOfAsc(gert::TilingContext* context, uint32_t maxCoreNum, uint32_t ubSizePlatform)
{
    OP_LOGD("TilingForAsStridedOfAsc", "Enter TilingForAsStridedOfAsc");

    gert::Shape outSize;
    gert::Shape outStride;
    AsStridedTilingParam tilingParam;
    AsStridedTilingData tilingData;

    tilingParam.numCore = maxCoreNum;
    tilingParam.ubSizePlatForm = ubSizePlatform;
    auto xTensorShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensorShape);
    const gert::Shape& xShape = xTensorShape->GetStorageShape();

    OP_CHECK_IF(
        !Ops::Base::GetConstIntToShape(context, IN_SIZE, outSize),
        OP_LOGE(context->GetNodeName(), "Get const of size failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !Ops::Base::GetConstIntToShape(context, IN_STRIDE, outStride),
        OP_LOGE(context->GetNodeName(), "Get const of stride failed"), return ge::GRAPH_FAILED);
    if (Ops::Base::GetConstInt(context, IN_OFFSET, tilingParam.storageOffset)) {
        OP_LOGD(context->GetNodeName(), "The storageOffset is const");
        OP_CHECK_IF(
            tilingParam.storageOffset < 0,
            OP_LOGE(context->GetNodeName(), "The storageOffset cannot be negative value! "), return ge::GRAPH_FAILED);
    } else {
        OP_LOGD(context->GetNodeName(), "The storageOffset is not const, will use default value 0");
        tilingParam.storageOffset = 0;
    }
    OP_CHECK_IF(
        outSize.GetDimNum() == 0, OP_LOGE(context->GetNodeName(), "The output size dim is 0!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        outSize.GetDimNum() > VALID_DIM,
        OP_LOGE(context->GetNodeName(), "The output size dim is larger than 8, the max dim is 8!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        outSize.GetDimNum() != outStride.GetDimNum(),
        OP_LOGE(
            context->GetNodeName(), "The dimension count of size and stride should be same! But %zu Vs %zu",
            outSize.GetDimNum(), outStride.GetDimNum()),
        return ge::GRAPH_FAILED);

    // To check input data can cover output
    OP_CHECK_IF(
        !CheckInputInfo(outSize, outStride, xShape, tilingParam.storageOffset),
        OP_LOGE(context->GetNodeName(), "The output element is out of input range!"), return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < outSize.GetDimNum(); i++) {
        OP_CHECK_IF(
            outSize.GetDim(i) == 0, OP_LOGE(context->GetNodeName(), "The output size has 0!"), return ge::GRAPH_FAILED);
    }

    auto xTensorType = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensorType);
    auto dataType = xTensorType->GetDataType();
    OP_CHECK_IF(
        tilingTypeKeyMap.count(dataType) == 0, OP_LOGE(context->GetNodeName(), "Not support data type"),
        return ge::GRAPH_FAILED);
    tilingParam.ubSize = (ubSizePlatform / BUFFER_NUM) / tilingTypeKeyMap[dataType];
    tilingParam.sizeofDtype = tilingTypeKeyMap[dataType];
    tilingParam.tilingKey = tilingTypeKeyMap[dataType];

    ge::graphStatus resOfTiling = ge::GRAPH_FAILED;
    resOfTiling = NDDMAForAsStrided(context, tilingParam, outSize, outStride, tilingData);
    OP_CHECK_IF(
        resOfTiling != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Tiling fail."), return ge::GRAPH_FAILED);

    SetTilingData(tilingData, tilingParam);
    OP_CHECK_IF(
        AsStridedSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "AsStridedSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    size_t usrSize = 0;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    context->SetBlockDim(tilingParam.blockNum);
    context->SetTilingKey(tilingParam.tilingKey);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling