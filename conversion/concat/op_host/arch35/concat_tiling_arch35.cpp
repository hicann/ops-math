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
 * \file concat_tiling_arch35.cpp
 * \brief concat tiling for ascendC impl
 */

#include "concat_tiling_arch35.h"
#include "log/log.h"
#include <cmath>
#include <sstream>
#include <cctype>
#include "op_common/op_host/util/shape_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_api/op_util.h"
#include <algorithm>
#include <limits>

using namespace std;
using namespace ge;

namespace optiling {

constexpr size_t CONCAT_DIM_IDX = 0;
constexpr int64_t INVLID_CONCAT_DIM_IDX = static_cast<int64_t>(-1);
constexpr size_t PACK_ATTR_AXIS_IDX = 0;
constexpr size_t PACK_INPUT_IDX = 0;
constexpr int64_t PACK_AXIS_DEFAULT_VALUE = 1;
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t HALF = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DIM1_ALIGN_THRESHOLD = 128;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_RESERVED_SIZE = 2048; // 2k
constexpr size_t SYSTEM_WORKSPACE_SIZE = 0;
constexpr int64_t INDEX_USE_UB = 1024; // 预留1k空间给索引
constexpr int64_t TENS_DIGITS = 10;
constexpr int64_t HUNDREDS_DIGITS = 100;
constexpr int64_t THOUSANDS_DIGITS = 1000;
constexpr int64_t TEN_THOUSANDS_DIGITS = 10000;
constexpr int64_t COMPACT_THRESHOLD = std::numeric_limits<int32_t>::max();
constexpr int64_t LEAST_ROWS = 64; // ub切分的最小行数
constexpr int64_t LEAST_COLS = 256;
constexpr bool ENABLE_DB = true;
constexpr int64_t B64_BYTES = 8;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B8_BYTES = 1;
constexpr int64_t B4_BYTES = 1004; // ge::GetSizeByDataType 对 FP4 类型的返回值（枚举值，非实际字节数）
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t FP4_TO_B8_RATIO = 2; // 用于FP4到B8模板的转换 2个FP4= 1字节
constexpr int64_t GATHER_MODE = 3;
constexpr int64_t ORIG_ARRAY_TILING_DATA = 0;       // 万位0: 原始有数组 (ConcatTilingData)
constexpr int64_t ORIG_NO_ARRAY_TILING_DATA = 1;    // 万位1: 原始无数组 (ConcatTilingDataNoArray)
constexpr int64_t COMPACT_NO_ARRAY_TILING_DATA = 2; // 万位2: compact无数组 (ConcatTilingDataNoArrayCompact)
constexpr int64_t COMPACT_ARRAY_TILING_DATA = 3;    // 万位3: compact有数组 (ConcatTilingDataCompact)
constexpr int64_t EVERY_CORE_THRESHOLD = 2048;      // 2k
constexpr int64_t LEAST_BLOCK_BYTES = 512;
constexpr int64_t PURE_COPY_COL_THRESHOLD_BASE = 256;
constexpr int64_t PURE_COPY_COL_THRESHOLD_ALIGN = 1024;
constexpr int64_t PURE_COPY_COL_THRESHOLD_NOALIGN = 2048;
constexpr int64_t BLOCK_THRESHOLD = 49152; // 48k
constexpr double LARGE_TENSOR_RATIO_THRESHOLD = 0.9;
constexpr int64_t PURE_COPY_NO_SPLIT_DIM1_TILINGKEY = 20001;
constexpr int64_t PURE_COPY_SPLIT_DIM1_TILINGKEY = 20002;
constexpr int64_t PURE_COPY_NO_SPLIT_DIM1_COMPACT_TILINGKEY = 20003;
constexpr int64_t PURE_COPY_SPLIT_DIM1_COMPACT_TILINGKEY = 20004;
constexpr int64_t SIMT_PER_CORE_THRESHOLD = 65536; // 64k
constexpr int64_t SIMT_TILINGKEY_PREFIX = 30000;
constexpr int64_t SIMT_COMPARE_THRESHOLD = 1024;
constexpr int64_t SMALL_BAG = 128;
constexpr int64_t ALL_DATA_SMALL = 8192;

constexpr int32_t NUM_2 = 2;
constexpr int32_t NUM_3 = 3;

template <typename T>
inline static ge::graphStatus ConcatSetTilingData(gert::TilingContext* context, T& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static inline void PrintTilingDataList(T& tilingData)
{
    auto strideList = tilingData.arrays.get_strideList();
    auto concatDimList = tilingData.arrays.get_concatDimList();
    for (int32_t i = 0; i < tilingData.get_tensorNum(); i++) {
        OP_LOGI("[Concat list]", "tensor: %d, stride: %ld, concatDim: %ld, segNum: %ld, isRowConcat: %d", i,
                strideList[i], concatDimList[i], tilingData.get_rowConcatSegNum(), tilingData.get_isRowConcat());
    }
}

template <typename T>
static inline void PrintTilingData(T& tilingData, int64_t tilingKey, int64_t usedCoreNum)
{
    OP_LOGI("[Concat]", "ubSplitDim1: %d, dim: %d, blockFactor: %ld, tailBlockFactor: %ld, \
ubFactorDim0: %d, ubFactorDim1: %d, tailUbFactorDim0: %d, tailUbFactorDim1: %d, uoDim0: %ld, uoDim1: %ld, \
tensorNum: %d, catDim1: %ld, isnon: %d, tilingKey: %ld, usedCoreNum: %ld",
            tilingData.get_ubSplitDim1(), tilingData.get_dim(), tilingData.get_blockFactor(),
            tilingData.get_tailBlockFactor(), tilingData.get_ubFactorDim0(), tilingData.get_ubFactorDim1(),
            tilingData.get_tailUbFactorDim0(), tilingData.get_tailUbFactorDim1(), tilingData.get_uoDim0(),
            tilingData.get_uoDim1(), tilingData.get_tensorNum(), tilingData.get_catDim1(),
            tilingData.get_isNonContiguous(), tilingKey, usedCoreNum);
    PrintTilingDataList(tilingData);
}

inline static ge::graphStatus GetTensorList(const gert::TilingContext* context, ConcatTilingParam& param,
                                            int64_t inputIdx)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    for (uint32_t i = 0; i < inputNum; ++i) {
        gert::Shape inputTensorShape = GetShapeByAll(context, param.isNonContiguous, inputIdx, i);
        size_t inputTensorDimNum = inputTensorShape.GetDimNum();
        vector<int64_t> inputShapeList(inputTensorDimNum, 0);
        for (size_t j = 0; j < inputTensorDimNum; j++) {
            inputShapeList[j] = inputTensorShape.GetDim(j);
        }
        param.tensorList.push_back(inputShapeList);
    }
    return ge::GRAPH_SUCCESS;
}

inline static int64_t MergeDim(const vector<int64_t>& tensorSize, int64_t startIdx, int64_t endIdx)
{
    int64_t ans = 1;
    for (int64_t i = startIdx; i < endIdx; i++) {
        ans *= tensorSize[i];
    }
    return ans;
}

inline static void GetTensorListDim(ConcatTilingParam& param)
{
    vector<int64_t> tmpTensor(DIM2);
    for (const auto& tensorSize : param.tensorList) {
        tmpTensor[DIM0] = MergeDim(tensorSize, 0, param.dim);
        tmpTensor[DIM1] = MergeDim(tensorSize, param.dim, tensorSize.size());
        param.mergeTensorList.push_back(tmpTensor);
    }

    for (const auto& tensorSize : param.mergeTensorList) {
        param.tensorListDim0.push_back(tensorSize[0]);
        param.tensorListDim1.push_back(tensorSize[1]);
    }
}

inline static void GetTensorSameDim1(ConcatTilingParam& param)
{
    if (static_cast<int64_t>(param.tensorListDim1.size()) > 0) {
        if (param.inputShapeSame == 1) {
            param.sameShapeTensorDim1 = param.tensorListDim1[0];
        } else {
            // shape 不同时只保存concat轴之后的相同部分
            param.sameShapeTensorDim1 = MergeDim(param.tensorList[0], param.dim + 1, param.tensorList[0].size());
        }
    }
}

inline static int64_t CalcSum(const vector<int64_t>& vec)
{
    int64_t sum = 0;
    for (const auto& it : vec) {
        sum += it;
    }
    return sum;
}

inline static void GenerateOutputShape(ConcatTilingParam& param)
{
    if (static_cast<int64_t>(param.tensorListDim0.size()) > 0) {
        param.catDim0 = param.tensorListDim0[0];
    } else {
        param.catDim0 = 0;
    }
    param.catDim1 = CalcSum(param.tensorListDim1);
    param.isEmpty = (param.catDim0 * param.catDim1) == 0;
}

inline static void CalcNoAlignTensorNum(ConcatTilingParam& param, int64_t dtypeSize)
{
    int64_t num = 0;
    for (const auto& tensorSize : param.tensorListDim1) {
        if (tensorSize * dtypeSize % BLOCK_SIZE != 0) {
            num += 1;
        }
    }
    param.noAlignTensorNum = num;
    OP_LOGD("[Concat]", "noAlignTensorNum: %ld", param.noAlignTensorNum);
}

inline static bool CheckCatDimAlign(vector<vector<int64_t>>& mergeTensorList, int64_t dtypeSize)
{
    // 用合轴之后的1轴去判断是否对齐
    for (int64_t i = 0; i < static_cast<int64_t>(mergeTensorList.size()); i++) {
        if (mergeTensorList[i][DIM1] * dtypeSize % BLOCK_SIZE != 0) {
            return false;
        }
    }
    return true;
}

inline static bool CheckDim1Align(vector<vector<int64_t>>& mergeTensorList, int64_t dtypeSize)
{
    // 用合轴之后的1轴去判断是否128对齐
    for (int64_t i = 0; i < static_cast<int64_t>(mergeTensorList.size()); i++) {
        if (mergeTensorList[i][DIM1] * dtypeSize % DIM1_ALIGN_THRESHOLD != 0) {
            return false;
        }
    }
    return true;
}

inline static bool CheckInputShapeSame(vector<vector<int64_t>>& tensorList)
{
    for (int64_t i = 0; i < static_cast<int64_t>(tensorList.size()) - 1; i++) {
        if (tensorList[i] != tensorList[i + 1]) {
            return false;
        }
    }
    return true;
}

inline static ge::graphStatus CheckFP4Dim1Even(const ConcatTilingParam& param)
{
    for (const auto& tensorSize : param.tensorListDim1) {
        if (tensorSize % FP4_TO_B8_RATIO != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Concat", "tensor_dim1", std::to_string(tensorSize).c_str(),
                                                  "The value of tensor_dim1 must be an even number for FP4 dtype.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertFP4DimsToB8(ConcatTilingParam& param)
{
    if (!param.isFP4Type) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus ret = CheckFP4Dim1Even(param);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    param.catDim1 /= FP4_TO_B8_RATIO;
    if (param.inputShapeSame == 1) {
        param.sameShapeTensorDim1 /= FP4_TO_B8_RATIO;
    }
    for (auto& tensorSize : param.tensorListDim1) {
        tensorSize /= FP4_TO_B8_RATIO;
    }
    for (auto& tensorSize : param.mergeTensorList) {
        tensorSize[1] /= FP4_TO_B8_RATIO;
    }
    if (param.isNonContiguous) {
        for (int16_t i = 0; i < param.tensorNum; ++i) {
            param.strideList[i] /= FP4_TO_B8_RATIO;
        }
    }
    param.dtypeSize = B8_BYTES;
    param.orgDtypeSize = B8_BYTES;
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus CalcBaseTilingParam(const gert::TilingContext* context, ConcatTilingParam& param)
{
    auto compileInfo = reinterpret_cast<const ConcatDCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    param.totalCoreNum = min(static_cast<int64_t>(compileInfo->totalCoreNum), TILING_ARRAY_LENGTH);
    if (compileInfo->totalCoreNum > TILING_ARRAY_LENGTH) {
        OP_LOGW("[Concat]", "Currently, more than 72 cores are not supported; only 72 cores are used.");
    }
    param.ubSize = compileInfo->ubSize;
    param.tensorNum = param.tensorList.size();
    param.gatherThreshold = compileInfo->vectorLen / DIGIT_TWO;
    GetTensorListDim(param);
    GenerateOutputShape(param);
    param.orgDtypeSize = param.dtypeSize;
    param.inputShapeSame = CheckInputShapeSame(param.mergeTensorList) ? 1 : 0;
    GetTensorSameDim1(param);
    // FP4 预处理：在对齐判断之前，将 FP4(4bit) 转换为 B8(1byte) 视角
    param.isFP4Type = (param.orgDtypeSize == B4_BYTES) ? 1 : 0;
    OP_CHECK_IF(ConvertFP4DimsToB8(param) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ConvertFP4DimsToB8 failed."), return ge::GRAPH_FAILED);
    param.isAllTensorAlign = CheckCatDimAlign(param.mergeTensorList, param.dtypeSize) ? 1 : 0;
    param.isDim1AllAlign = CheckDim1Align(param.mergeTensorList, param.dtypeSize) ? 1 : 0;
    OP_CHECK_IF(
        param.dtypeSize <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input", std::to_string(param.dtypeSize).c_str(),
                                              "The dtype size of input must be greater than 0."),
        return ge::GRAPH_FAILED);
    param.leastCopyNumber = MIN_RESERVED_SIZE / param.dtypeSize;
    param.everyBlockNumber = BLOCK_SIZE / param.dtypeSize;
    CalcNoAlignTensorNum(param, param.dtypeSize);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static ge::graphStatus GetConcatDimInput(const gert::TilingContext* context, ConcatTilingParam& param,
                                                int64_t dimIdx)
{
    auto concatDimTensor = context->GetRequiredInputTensor(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimTensor);
    const T* concatDimValPtr = concatDimTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimValPtr);
    param.dim = concatDimValPtr[0];
    return ge::GRAPH_SUCCESS;
}

inline static bool IsInvalidType(const DataType dtype)
{
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT,       ge::DT_FLOAT16,  ge::DT_BF16,        ge::DT_UINT8,       ge::DT_INT8,
        ge::DT_UINT16,      ge::DT_INT16,    ge::DT_UINT32,      ge::DT_INT32,       ge::DT_UINT64,
        ge::DT_INT64,       ge::DT_BOOL,     ge::DT_DOUBLE,      ge::DT_COMPLEX64,   ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E2M1};
    bool isInvalidType = (supportedDtype.count(dtype) == 0);

    return isInvalidType;
}

inline static ge::graphStatus GetDtypeSize(const gert::TilingContext* context, ConcatTilingParam& param,
                                           size_t inputIndex)
{
    auto inputDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    param.dtypeSize = ge::GetSizeByDataType(inputDataType);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static void DupTensor(vector<T>& dst, const vector<T>& src, int64_t num)
{
    int64_t index = 0;
    for (int i = 0; i < num; i++) {
        for (int64_t j = 0; (j < static_cast<int64_t>(src.size()) && index < TILING_ARRAY_LENGTH); j++) {
            dst[index] = src[j];
            index++;
        }
    }
}

inline static bool IsEnableGather(const ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 0 && param.inputShapeSame == 1 &&
        param.sameShapeTensorDim1 * param.dtypeSize < param.gatherThreshold) {
        return true;
    }
    return false;
}

inline static bool IsEnableScatter(const ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 0 && param.inputShapeSame == 0) {
        return true;
    }
    return false;
}

inline static bool IsEnableRowConcat(const ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 1 && param.dim == 0 && !param.isEmpty) {
        return true;
    }
    return false;
}

inline static void CalcLargeTensorNum(const ConcatTilingParam& param, int64_t tensorCol, int64_t rowsUsedCoreNum,
                                      int64_t& largeInputNum, int64_t& totalInputNum)
{
    if (tensorCol * param.ubFactorDim0 * param.dtypeSize >= BLOCK_THRESHOLD) {
        largeInputNum += (rowsUsedCoreNum - 1);
    }
    if (tensorCol * param.tailUbFactorDim0 * param.dtypeSize >= BLOCK_THRESHOLD) {
        largeInputNum += 1;
    }
    totalInputNum += rowsUsedCoreNum;
}

inline static bool IsEnablePureCopyTemplate(const ConcatTilingParam& param, int64_t rowsUsedCoreNum,
                                            int64_t colsUsedCoreNum)
{
    int64_t threshold = 0;
    if (param.isDim1AllAlign == 1 && param.inputShapeSame == 1) {
        threshold = PURE_COPY_COL_THRESHOLD_BASE;
    } else if (param.isDim1AllAlign == 1 || param.inputShapeSame == 1) {
        threshold = PURE_COPY_COL_THRESHOLD_ALIGN;
    } else {
        threshold = PURE_COPY_COL_THRESHOLD_NOALIGN;
    }
    for (const auto& tensorSize : param.tensorListDim1) {
        if (tensorSize * param.dtypeSize < threshold) {
            return false;
        }
    }
    int64_t totalInputNum = 0;
    int64_t largeInputNum = 0;
    if (param.blockSplitAxis == 0) {
        for (const auto& tensorSize : param.tensorListDim1) {
            CalcLargeTensorNum(param, tensorSize, param.usedCoreNum, largeInputNum, totalInputNum);
        }
    } else {
        for (int64_t i = 0; i < colsUsedCoreNum; i++) {
            if (param.startTensorIdx[i] == param.endTensorIdx[i]) {
                int64_t tensorCol = param.endTensorOffset[i] - param.startTensorOffset[i];
                CalcLargeTensorNum(param, tensorCol, rowsUsedCoreNum, largeInputNum, totalInputNum);
                continue;
            }
            int16_t startIdx = param.startTensorIdx[i];
            CalcLargeTensorNum(param, param.tensorListDim1[startIdx] - param.startTensorOffset[i], rowsUsedCoreNum,
                               largeInputNum, totalInputNum);
            for (int16_t k = param.startTensorIdx[i] + 1; k < param.endTensorIdx[i]; k++) {
                CalcLargeTensorNum(param, param.tensorListDim1[k], rowsUsedCoreNum, largeInputNum, totalInputNum);
            }
            int64_t lastTensorCol = param.endTensorOffset[i];
            CalcLargeTensorNum(param, lastTensorCol, rowsUsedCoreNum, largeInputNum, totalInputNum);
        }
    }
    if (totalInputNum <= 0) {
        return false;
    }
    double largeRatio = static_cast<double>(largeInputNum) / static_cast<double>(totalInputNum);
    if (largeRatio >= LARGE_TENSOR_RATIO_THRESHOLD) {
        return true;
    }
    return false;
}

inline static void GenTilingKey(ConcatTilingParam& param)
{
    // tilingKey按5位设计：个位->字节数(1/2/4/8),十位->input shape相同/不相同(1/2)
    // 百位->输入cat部分全对齐/不对齐/不对齐gather模板(1/2/3),千位->首轴cat/非首轴cat(1/2),万位->tilingData类型
    //   万位 0=原始有数组, 1=原始无数组, 2=compact无数组, 3=compact有数组
    if (param.isEmpty) {
        param.tilingKey = 0;
        return;
    }
    bool shapeSame = param.inputShapeSame == 1;
    bool isAllTensorAlign = param.isAllTensorAlign == 1;

    int64_t isCatDimAlign = isAllTensorAlign ? 1 : 2;
    int64_t dtypeSize = param.dtypeSize;
    if (IsEnableScatter(param)) {
        dtypeSize = param.orgDtypeSize;
    }
    if (param.isGather || param.isRowConcat) {
        // 非连续 gather/rowconcat 强制走 diff-shape 模板（1222x），避免命中 same-shape 模板
        isCatDimAlign = 2;
    } else if (IsEnableGather(param)) {
        isCatDimAlign = GATHER_MODE;
    }
    int64_t isInputShapeSame = (shapeSame && !param.isGather && !param.isRowConcat) ? 1 : 2;
    int64_t isFirstDim = DIGIT_TWO;

    // 判断是否可用 compact: 所有 tensor 的 dim1 < UINT32_MAX
    bool canCompact = true;
    for (const auto& dim1 : param.tensorListDim1) {
        if (static_cast<uint64_t>(dim1) >= COMPACT_THRESHOLD) {
            canCompact = false;
            break;
        }
    }

    int64_t isUseSpcTilingData;
    if (param.blockSplitAxis == 0) {
        isUseSpcTilingData = canCompact ? COMPACT_NO_ARRAY_TILING_DATA : ORIG_NO_ARRAY_TILING_DATA;
    } else {
        isUseSpcTilingData = canCompact ? COMPACT_ARRAY_TILING_DATA : ORIG_ARRAY_TILING_DATA;
    }

    param.tilingKey = dtypeSize + isInputShapeSame * TENS_DIGITS + isCatDimAlign * HUNDREDS_DIGITS +
                      isFirstDim * THOUSANDS_DIGITS + isUseSpcTilingData * TEN_THOUSANDS_DIGITS;
}

inline static ge::graphStatus IsDimValid(const gert::TilingContext* context, int64_t& dim, int64_t inputIdx,
                                         bool isNonContiguous, int64_t& strideDim)
{
    gert::Shape inputShape = GetShapeByAll(context, isNonContiguous, inputIdx, 0);
    int64_t shapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    int64_t minDim = shapeSize * static_cast<int64_t>(-1);
    int64_t maxDim = shapeSize - 1;
    if (!(dim >= minDim && dim <= maxDim)) {
        return ge::GRAPH_FAILED;
    }
    // convert negative dim to positive dim
    if (dim < 0) {
        dim += shapeSize;
    }
    strideDim = dim - 1;
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus IsShapeValid(const gert::TilingContext* context, vector<vector<int64_t>>& tensorList,
                                           int64_t realDim)
{
    if (tensorList.size() < 1) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t dimSize = tensorList[0].size();
    auto shape0 = tensorList[0];
    for (const auto& tensorSize : tensorList) {
        int64_t curDimSize = tensorSize.size();
        OP_CHECK_IF(curDimSize != dimSize,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context->GetNodeName(), "input_tensors",
                        (std::to_string(dimSize) + ", " + std::to_string(curDimSize)).c_str(),
                        "The shape dims of input tensors must be the same."),
                    return ge::GRAPH_FAILED);
        for (int64_t j = 0; j < dimSize; j++) {
            if (realDim == j) {
                continue;
            }
            OP_CHECK_IF(shape0[j] != tensorSize[j],
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context->GetNodeName(), "input_tensors",
                            (std::to_string(shape0[j]) + ", " + std::to_string(tensorSize[j])).c_str(),
                            ("Shape [" + std::to_string(j) + "] of input tensors must be the same.").c_str()),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingUbForNosplitDim1(gert::TilingContext* context, int64_t maxAvaliableUb,
                                              ConcatTilingParam& param)
{
    OP_CHECK_IF(param.catDim1 <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "output_tensor",
                                                      std::to_string(param.catDim1).c_str(),
                                                      "Shape concat_axis of output_tensor must be greater than 0."),
                return ge::GRAPH_FAILED);
    int64_t ubDim1 = param.catDim1;
    if (param.isRowConcat && param.rowConcatSegNum > 1) {
        // rowconcat kernel 逐段搬入 UB,仅需容纳单个 tensor 一个段(seg = dim1/segNum)的数据,
        // 段超出 UB 容量时 kernel 内部会再分块搬运
        int64_t maxSeg = 0;
        for (int64_t i = 0; i < param.tensorNum; i++) {
            // concatDimList tensor i 沿 concat轴的长度，sameShapeTensorDim1
            // dim+1到N的乘积，rowConcatSegNum从非连续轴+1到dim轴的乘积 rowconcat: kernel 逐段搬 → 一个段就够 seg =
            // ceil(dim1_i / segNum)        // 每个 tensor 的段长 ubDim1 = max over tensors          // 最宽的段决定预算
            // 例: (A,B·C) 合轴, segNum=B → 段长=C → ubDim1 = C
            //  (kernel CopyInNoSplitDim1 rowconcat 分支按 段/段组 循环,一次只占一个段的 UB, 段超容量时 kernel 内部再按
            //  bufferSize 切)
            int64_t dim1 = static_cast<int64_t>(param.concatDimList[i]) * param.sameShapeTensorDim1;
            int64_t seg = (dim1 + param.rowConcatSegNum - 1) / param.rowConcatSegNum;
            maxSeg = std::max(maxSeg, seg);
        }
        ubDim1 = maxSeg;
    }
    // gather: kernel 逐 tensor 搬 → 最大 tensor 摊平宽(32B 上取整)
    // ubDim1 = CeilAlign(max(dim1_i), everyBlockNumber)
    // 例: 33×(2,16) f16 → 各 tensor 摊平 32 元素 → ubDim1 = 32
    //(kernel 的 workLocal 行按 CeilAlign 落 UB, 预算必须按对齐后算, 否则行数超预算越界)
    if (param.isGather) {
        int64_t maxDim1 = 0;
        for (int64_t i = 0; i < param.tensorNum; i++) {
            int64_t dim1 = static_cast<int64_t>(param.concatDimList[i]) * param.sameShapeTensorDim1;
            maxDim1 = std::max(maxDim1, dim1);
        }
        ubDim1 = (maxDim1 + param.everyBlockNumber - 1) / param.everyBlockNumber * param.everyBlockNumber;
        // 32B 对齐上取整可能把刚好贴着预算的摊平宽推过线(maxDim1+31 内), 明确报错
        // 并给出预算值, 避免下游 ubFactorDim0=0 的间接报错
        OP_CHECK_IF(ubDim1 > maxAvaliableUb,
                    OP_LOGE(context->GetNodeName(),
                            "gather flattened tensor width %ld (aligned) exceeds UB budget %ld, gather path "
                            "unsupported on this device, fallback to regular path is required",
                            ubDim1, maxAvaliableUb),
                    return ge::GRAPH_FAILED);
    }
    param.ubFactorDim0 = min(maxAvaliableUb / ubDim1, param.catDim0);
    if (param.isRowConcat) {
        // rowconcat 仅支持 dim0 切分(NoSplitDim1 模板),usedCoreNum == uoDim0 == ceil(catDim0/ubFactorDim0)。
        // kernel 逐行逐段搬运,单次搬运大小只由段/段组决定,与 ubFactorDim0 无关;
        // ubFactorDim0 受 UB 预算(maxAvaliableUb/maxSeg)钳制时 uoDim0 会退化为 1~4 核,
        // 整份拷贝压在少数核上,此处固定 1 行/块,将行充分摊到多核并行。
        param.ubFactorDim0 = 1;
    }
    OP_CHECK_IF(
        param.ubFactorDim0 <= 0,
        OP_LOGE(context->GetNodeName(), "ubFactorDim0 must be greater than 0, ubFactorDim0: %ld", param.ubFactorDim0),
        return ge::GRAPH_FAILED);
    param.uoDim0 = (param.catDim0 + param.ubFactorDim0 - 1) / param.ubFactorDim0;
    param.uoDim1 = 1;
    param.ubFactorDim1 = param.catDim1;
    param.tailUbFactorDim0 = param.catDim0 % param.ubFactorDim0;
    if (param.tailUbFactorDim0 == 0) {
        param.tailUbFactorDim0 = param.ubFactorDim0;
    }
    param.tailUbFactorDim1 = param.catDim1;
    param.ubSplitDim1 = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingUbForSplitDim1(gert::TilingContext* context, int64_t maxAvaliableUb,
                                            int64_t storageAlignUsed, int64_t maxDim1Factor, ConcatTilingParam& param)
{
    int64_t realFactorDim1 = maxDim1Factor;
    if (param.isAllTensorAlign == 0 && param.inputShapeSame == 1) {
        // tensor不对齐且需要切列的场景需要kernel侧重新进行ub切分，此处不再预留storage_align空间
        realFactorDim1 = maxAvaliableUb / std::min(LEAST_ROWS, param.catDim0);
        OP_CHECK_IF(param.everyBlockNumber <= 0,
                    OP_LOGE(context->GetNodeName(), "everyBlockNumber must be greater than 0, everyBlockNumber: %ld",
                            param.everyBlockNumber),
                    return ge::GRAPH_FAILED);
        // 默认 32B 块
        int64_t alignFactorDim1 = param.everyBlockNumber;
        // 小段场景按 tensor 单元对齐
        if (param.inputShapeSame == 1 && param.sameShapeTensorDim1 * param.dtypeSize <= param.gatherThreshold) {
            alignFactorDim1 = param.sameShapeTensorDim1;
        }
        // 向下对齐
        realFactorDim1 = realFactorDim1 / alignFactorDim1 * alignFactorDim1;
    } else {
        maxAvaliableUb -= storageAlignUsed;
    }
    param.ubFactorDim1 = min(realFactorDim1, param.catDim1);
    if (param.isGather && param.gatherSeg > 0) {
        // 单个连续段(gatherSeg)必须装得进本设备 gather UB 预算: 段是 kernel 按 2D DMA 整段
        // 搬运的最小单元, 无法再切; 超预算时明确报错并给出预算值, 避免"下限一个段"兜底
        // 把 ubFactorDim1 抬超预算, 导致下游 ubFactorDim0=0 的间接报错(根因难定位,
        // 且随设备 ubSize 不同 256K/512K 表现不一致)
        OP_CHECK_IF(param.gatherSeg > maxAvaliableUb,
                    OP_LOGE(context->GetNodeName(),
                            "gatherSeg %ld exceeds UB budget %ld, gather path unsupported on this device, "
                            "segment size is shape-determined, fallback to regular path is required",
                            param.gatherSeg, maxAvaliableUb),
                    return ge::GRAPH_FAILED);
        // gatherSeg 非连续轴后b+1到结尾N的所有dim乘积
        // gather kernel 逐行搬运，总调用数 = catDim0 * uoDim1 * 2，仅随列块数 uoDim1 增长、与行数无关：
        // 按单行预算取 UB 能容纳的最大 gatherSeg
        // 整数倍列宽（下限一个段），全预留给列。按照段对齐后，至少要有一个段max(maxColsByUb, param.gatherSeg)
        // 使每行一次 DataCopyPad 合并尽量多的段，避免退化成单段(512B级)小搬运。
        int64_t maxColsByUb = maxAvaliableUb / param.gatherSeg * param.gatherSeg;
        param.ubFactorDim1 = min(param.catDim1, max(maxColsByUb, param.gatherSeg));
    }
    OP_CHECK_IF(param.ubFactorDim1 <= 0,
                OP_LOGE(context->GetNodeName(), "param.ubFactorDim1 must be greater than 0, param.ubFactorDim1: %ld",
                        param.ubFactorDim1),
                return ge::GRAPH_FAILED);
    param.ubFactorDim0 = min(maxAvaliableUb / param.ubFactorDim1, param.catDim0);
    if (param.isGather) {
        // kernel 侧 gather 逐行按 32B 对齐落 UB，ubFactorDim0 必须按对齐后的行宽回算，
        // 否则 rows*CeilAlign(ubFactorDim1, everyBlockNumber) 会超出 bufferSize 导致越界
        int64_t paddedDim1 = (param.ubFactorDim1 + param.everyBlockNumber - 1) / param.everyBlockNumber *
                             param.everyBlockNumber;
        param.ubFactorDim0 = min(param.ubFactorDim0, maxAvaliableUb / paddedDim1);
    }
    OP_CHECK_IF(param.ubFactorDim0 <= 0,
                OP_LOGE(context->GetNodeName(), "param.ubFactorDim0 must be greater than 0, param.ubFactorDim0: %ld",
                        param.ubFactorDim0),
                return ge::GRAPH_FAILED);
    param.uoDim1 = (param.catDim1 + param.ubFactorDim1 - 1) / param.ubFactorDim1;
    param.uoDim0 = (param.catDim0 + param.ubFactorDim0 - 1) / param.ubFactorDim0;
    param.tailUbFactorDim0 = param.catDim0 % param.ubFactorDim0;
    if (param.tailUbFactorDim0 == 0) {
        param.tailUbFactorDim0 = param.ubFactorDim0;
    }
    param.tailUbFactorDim1 = param.catDim1 % param.ubFactorDim1;
    if (param.tailUbFactorDim1 == 0) {
        param.tailUbFactorDim1 = param.ubFactorDim1;
    }
    param.ubSplitDim1 = 1;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingUb(gert::TilingContext* context, ConcatTilingParam& param)
{
    OP_CHECK_IF(
        param.dtypeSize <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "x", Ops::Base::ToString(context->GetDynamicInputDesc(0, 0)->GetDataType()).c_str(),
            "The dtype size of x must be greater than 0."),
        return ge::GRAPH_FAILED);
    int64_t maxAvaliableUb = (param.ubSize - INDEX_USE_UB) / param.dtypeSize;
    if (param.isAllTensorAlign == 0 || param.isGather) {
        // tensor不对齐的场景下，需要在UB中拼接，内存分成输入输出2部分
        maxAvaliableUb = maxAvaliableUb / BUFFER_NUM;
        // 非对齐场景scatter/gather索引为u16/u32,需确保ub内每个tensor的元素个数不超过U16上限
        maxAvaliableUb = std::min(maxAvaliableUb, static_cast<int64_t>(std::numeric_limits<uint16_t>::max()));
    }
    param.bufferSize = maxAvaliableUb;
    int64_t realFactorDim1 = maxAvaliableUb / std::min(LEAST_ROWS, param.catDim0);
    int64_t storageAlignUsed = 0;
    if (param.isAllTensorAlign == 0) {
        // tensor不对齐的场景下，预留输入和输出storage_align空间
        storageAlignUsed = param.everyBlockNumber * (param.noAlignTensorNum + 1);
        realFactorDim1 = (maxAvaliableUb - storageAlignUsed) / std::min(LEAST_ROWS, param.catDim0);
    }
    OP_CHECK_IF(
        param.everyBlockNumber <= 0,
        OP_LOGE(context->GetNodeName(), "param.everyBlockNumber must be greater than 0, param.everyBlockNumber: %ld",
                param.everyBlockNumber),
        return ge::GRAPH_FAILED);
    realFactorDim1 = realFactorDim1 / param.everyBlockNumber * param.everyBlockNumber;

    if (param.isRowConcat && param.rowConcatSegNum <= 1) {
        param.isRowConcat = 0;
    }

    if (param.catDim1 < realFactorDim1 || param.isRowConcat) {
        maxAvaliableUb = maxAvaliableUb - storageAlignUsed;
        OP_CHECK_IF(TilingUbForNosplitDim1(context, maxAvaliableUb, param) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "TilingUbForNosplitDim1 failed"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            TilingUbForSplitDim1(context, maxAvaliableUb, storageAlignUsed, realFactorDim1, param) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "TilingUbForSplitDim1 failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus TilingBlock(gert::TilingContext* context, ConcatTilingParam& param)
{
    // 非连续 rowconcat 场景 kernel 仅支持 NoSplitDim1 模板，必须保持 dim0 切分；
    // 其余非连续场景（gather/断点轴==strideDim 的老路径）kernel 侧 ProcessBlockSplitDim1
    // 均有对应分支（老路径分支即提交前原始代码），catDim0 小时 uoDim0 仅为 1~2，
    // 强制 dim0 切分会把整份拷贝压在极少数核上，此处放行 dim1 借轴多核切分。
    bool nonConDim1SplitAllowed = param.isNonContiguous && param.isRowConcat == 0;
    if (param.uoDim0 > (param.totalCoreNum / HALF) || (param.isNonContiguous && !nonConDim1SplitAllowed)) {
        OP_CHECK_IF(param.totalCoreNum <= 0,
                    OP_LOGE(context->GetNodeName(),
                            "param.totalCoreNum must be greater than 0, param.totalCoreNum: %ld", param.totalCoreNum),
                    return ge::GRAPH_FAILED);
        param.blockFactor = (param.uoDim0 + param.totalCoreNum - 1) / param.totalCoreNum;
        OP_CHECK_IF(param.blockFactor <= 0,
                    OP_LOGE(context->GetNodeName(), "param.blockFactor must be greater than 0, param.blockFactor: %ld",
                            param.blockFactor),
                    return ge::GRAPH_FAILED);
        param.usedCoreNum = (param.uoDim0 + param.blockFactor - 1) / param.blockFactor;
        param.tailBlockFactor = param.uoDim0 - (param.usedCoreNum - 1) * param.blockFactor;
        param.blockSplitAxis = 0;
    } else {
        int64_t rowsUsedCoreNum = param.uoDim0;
        OP_CHECK_IF(rowsUsedCoreNum <= 0,
                    OP_LOGE(context->GetNodeName(), "rowsUsedCoreNum must be greater than 0, rowsUsedCoreNum: %ld",
                            rowsUsedCoreNum),
                    return ge::GRAPH_FAILED);
        int64_t leftCore = param.totalCoreNum / rowsUsedCoreNum;
        int64_t alignFactorDim1 = param.everyBlockNumber;
        if (param.inputShapeSame == 1 && param.sameShapeTensorDim1 * param.dtypeSize <= param.gatherThreshold) {
            alignFactorDim1 = param.sameShapeTensorDim1;
        }
        // gather 借轴减半 ubFactorDim1 时必须保持 gatherSeg 整数倍对齐：
        // kernel 侧 gather 搬运以 startSeg = colsOffset/gatherSeg、numSeg = copyCols/gatherSeg
        // 整除计算段号/段数，非整数倍会静默截断导致数据错位。
        if (param.isGather == 1 && param.gatherSeg > alignFactorDim1) {
            alignFactorDim1 = param.gatherSeg;
        } else if (param.isGather == 1 && param.gatherSeg > 1 && alignFactorDim1 % param.gatherSeg != 0) {
            // seg ≤ af 且不整除: 仅按 af 对齐会在借轴减半时丢失 seg 整数倍性
            // (如 seg=5/af=16: 2160→1080→对齐16得1072, 1072%5≠0 → kernel 段号截断错位)。
            // 对齐到 lcm(af, seg), 同时保住 32B 块对齐与 seg 整数倍两种性质。
            int64_t gcdAfSeg = alignFactorDim1;
            int64_t segRem = param.gatherSeg;
            while (segRem != 0) {
                int64_t tmp = gcdAfSeg % segRem;
                gcdAfSeg = segRem;
                segRem = tmp;
            }
            alignFactorDim1 = alignFactorDim1 / gcdAfSeg * param.gatherSeg;
        }
        OP_CHECK_IF(alignFactorDim1 <= 0,
                    OP_LOGE(context->GetNodeName(), "alignFactorDim1 must be greater than 0, alignFactorDim1: %ld",
                            alignFactorDim1),
                    return ge::GRAPH_FAILED);
        // 核未开满时，dim1借轴
        while (param.uoDim1 < leftCore && param.ubFactorDim1 * param.dtypeSize >= LEAST_COLS &&
               param.ubFactorDim1 * param.ubFactorDim0 >= HALF * param.leastCopyNumber &&
               param.ubFactorDim1 >= HALF * alignFactorDim1) {
            param.ubFactorDim1 = (param.ubFactorDim1 / HALF) / alignFactorDim1 * alignFactorDim1;
            OP_CHECK_IF(
                param.ubFactorDim1 <= 0,
                OP_LOGE(context->GetNodeName(), "param.ubFactorDim1 must be greater than 0, param.ubFactorDim1: %ld",
                        param.ubFactorDim1),
                return ge::GRAPH_FAILED);
            param.uoDim1 = (param.catDim1 + param.ubFactorDim1 - 1) / param.ubFactorDim1;
            param.tailUbFactorDim1 = param.catDim1 % param.ubFactorDim1;
            if (param.tailUbFactorDim1 == 0) {
                param.tailUbFactorDim1 = param.ubFactorDim1;
            }
        }
        if (param.uoDim1 > 1) {
            param.blockSplitAxis = 1;
            OP_CHECK_IF(leftCore <= 0,
                        OP_LOGE(context->GetNodeName(), "leftCore must be greater than 0, leftCore: %ld", leftCore),
                        return ge::GRAPH_FAILED);
            param.blockFactor = (param.uoDim1 + leftCore - 1) / leftCore;
            OP_CHECK_IF(param.blockFactor <= 0,
                        OP_LOGE(context->GetNodeName(),
                                "param.blockFactor must be greater than 0, param.blockFactor: %ld", param.blockFactor),
                        return ge::GRAPH_FAILED);
            int64_t colsUsedCoreNum = (param.uoDim1 + param.blockFactor - 1) / param.blockFactor;
            param.usedCoreNum = rowsUsedCoreNum * colsUsedCoreNum;
            param.tailBlockFactor = param.uoDim1 - (colsUsedCoreNum - 1) * param.blockFactor;
        } else {
            param.blockSplitAxis = 0;
            param.blockFactor = 1;
            param.tailBlockFactor = 1;
            param.usedCoreNum = rowsUsedCoreNum;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline static void SetTensorListTilingData(ConcatTilingData& tilingData, ConcatTilingParam& param)
{
    std::copy(param.endTensorIdx.begin(), param.endTensorIdx.end(), param.endIdxArr);
    tilingData.arrays.set_endTensorIdx(param.endIdxArr);

    std::copy(param.endTensorOffset.begin(), param.endTensorOffset.end(), param.endOffsetArr);
    tilingData.arrays.set_endTensorOffset(param.endOffsetArr);
}

template <typename T>
inline static void SetTilingData(T& tilingData, ConcatTilingParam& param)
{
    // set tiling data
    tilingData.set_ubSplitDim1(param.ubSplitDim1);
    tilingData.set_dim(static_cast<int16_t>(param.dim));
    tilingData.set_blockFactor(param.blockFactor);
    tilingData.set_tailBlockFactor(param.tailBlockFactor);
    tilingData.set_ubFactorDim0(static_cast<int32_t>(param.ubFactorDim0));
    tilingData.set_ubFactorDim1(static_cast<int32_t>(param.ubFactorDim1));
    tilingData.set_tailUbFactorDim0(static_cast<int32_t>(param.tailUbFactorDim0));
    tilingData.set_tailUbFactorDim1(static_cast<int32_t>(param.tailUbFactorDim1));
    tilingData.set_uoDim0(param.uoDim0);
    tilingData.set_uoDim1(param.uoDim1);
    tilingData.set_tensorNum(param.tensorNum);
    tilingData.set_catDim1(param.catDim1);
    tilingData.set_sameShapeTensorDim1(param.sameShapeTensorDim1);
    tilingData.set_isFP4Type(param.isFP4Type);
    tilingData.set_bufferSize(static_cast<int32_t>(param.bufferSize));
    tilingData.set_dtypeSize(static_cast<int16_t>(param.dtypeSize));

    int64_t preLoadSize = std::min(TILING_PRELOAD_DIM1_LENGTH, static_cast<int64_t>(param.tensorListDim1.size()));
    std::copy(param.tensorListDim1.begin(), param.tensorListDim1.begin() + preLoadSize, param.preLoadDim1Arr);
    tilingData.arrays.set_preLoadDim1(param.preLoadDim1Arr);
    tilingData.set_isNonContiguous(static_cast<int16_t>(param.isNonContiguous ? 1 : 0));
    tilingData.set_isGather(param.isGather);
    tilingData.set_isRowConcat(param.isRowConcat);
    tilingData.set_gatherSeg(param.gatherSeg);
    tilingData.set_rowConcatSegNum(param.rowConcatSegNum);
    if (param.isNonContiguous) {
        uint64_t strideList[NON_CON_TENSOR_SIZE];
        std::copy(param.strideList.begin(), param.strideList.end(), strideList);
        tilingData.arrays.set_strideList(strideList);
        std::copy(param.concatDimList.begin(), param.concatDimList.end(), strideList);
        tilingData.arrays.set_concatDimList(strideList);
    }
}

// compact 版 SetTensorListTilingData
inline static void SetTensorListTilingData(ConcatTilingDataCompact& tilingData, ConcatTilingParam& param)
{
    std::copy(param.endTensorIdx.begin(), param.endTensorIdx.end(), param.endIdxArr);
    tilingData.arrays.set_endTensorIdx(param.endIdxArr);

    for (size_t i = 0; i < param.endTensorOffset.size() && i < TILING_ARRAY_LENGTH; i++) {
        param.endOffsetArrCompact[i] = static_cast<uint32_t>(param.endTensorOffset[i]);
    }
    tilingData.arrays.set_endTensorOffset(param.endOffsetArrCompact);
}

// compact 版 SetTilingData 特化
template <>
void SetTilingData<ConcatTilingDataCompact>(ConcatTilingDataCompact& tilingData, ConcatTilingParam& param)
{
    tilingData.set_ubSplitDim1(param.ubSplitDim1);
    tilingData.set_dim(static_cast<int16_t>(param.dim));
    tilingData.set_blockFactor(param.blockFactor);
    tilingData.set_tailBlockFactor(param.tailBlockFactor);
    tilingData.set_ubFactorDim0(static_cast<int32_t>(param.ubFactorDim0));
    tilingData.set_ubFactorDim1(static_cast<int32_t>(param.ubFactorDim1));
    tilingData.set_tailUbFactorDim0(static_cast<int32_t>(param.tailUbFactorDim0));
    tilingData.set_tailUbFactorDim1(static_cast<int32_t>(param.tailUbFactorDim1));
    tilingData.set_uoDim0(param.uoDim0);
    tilingData.set_uoDim1(param.uoDim1);
    tilingData.set_tensorNum(param.tensorNum);
    tilingData.set_catDim1(param.catDim1);
    tilingData.set_sameShapeTensorDim1(param.sameShapeTensorDim1);
    tilingData.set_isFP4Type(param.isFP4Type);
    tilingData.set_bufferSize(static_cast<int32_t>(param.bufferSize));
    tilingData.set_dtypeSize(static_cast<int16_t>(param.dtypeSize));

    int64_t preLoadSize = std::min(TILING_PRELOAD_DIM1_LENGTH, static_cast<int64_t>(param.tensorListDim1.size()));
    for (int64_t i = 0; i < preLoadSize; i++) {
        param.preLoadDim1ArrCompact[i] = static_cast<uint32_t>(param.tensorListDim1[i]);
    }
    tilingData.arrays.set_preLoadDim1(param.preLoadDim1ArrCompact);
    tilingData.set_isNonContiguous(static_cast<int16_t>(param.isNonContiguous ? 1 : 0));
    tilingData.set_isGather(param.isGather);
    tilingData.set_isRowConcat(param.isRowConcat);
    tilingData.set_gatherSeg(param.gatherSeg);
    tilingData.set_rowConcatSegNum(param.rowConcatSegNum);
    if (param.isNonContiguous) {
        for (size_t i = 0; i < std::min(param.strideList.size(), static_cast<size_t>(NON_CON_TENSOR_SIZE)); i++) {
            param.strideListCompact[i] = static_cast<uint32_t>(param.strideList[i]);
        }
        tilingData.arrays.set_strideList(param.strideListCompact);
        for (size_t i = 0; i < std::min(param.concatDimList.size(), static_cast<size_t>(NON_CON_TENSOR_SIZE)); i++) {
            param.concatDimListCompact[i] = static_cast<uint32_t>(param.concatDimList[i]);
        }
        tilingData.arrays.set_concatDimList(param.concatDimListCompact);
    }
}

template <>
void SetTilingData<ConcatTilingDataNoArrayCompact>(ConcatTilingDataNoArrayCompact& tilingData, ConcatTilingParam& param)
{
    tilingData.set_ubSplitDim1(param.ubSplitDim1);
    tilingData.set_dim(static_cast<int16_t>(param.dim));
    tilingData.set_blockFactor(param.blockFactor);
    tilingData.set_tailBlockFactor(param.tailBlockFactor);
    tilingData.set_ubFactorDim0(static_cast<int32_t>(param.ubFactorDim0));
    tilingData.set_ubFactorDim1(static_cast<int32_t>(param.ubFactorDim1));
    tilingData.set_tailUbFactorDim0(static_cast<int32_t>(param.tailUbFactorDim0));
    tilingData.set_tailUbFactorDim1(static_cast<int32_t>(param.tailUbFactorDim1));
    tilingData.set_uoDim0(param.uoDim0);
    tilingData.set_uoDim1(param.uoDim1);
    tilingData.set_tensorNum(param.tensorNum);
    tilingData.set_catDim1(param.catDim1);
    tilingData.set_sameShapeTensorDim1(param.sameShapeTensorDim1);
    tilingData.set_isFP4Type(param.isFP4Type);
    tilingData.set_bufferSize(static_cast<int32_t>(param.bufferSize));
    tilingData.set_dtypeSize(static_cast<int16_t>(param.dtypeSize));

    int64_t preLoadSize = std::min(TILING_PRELOAD_DIM1_LENGTH, static_cast<int64_t>(param.tensorListDim1.size()));
    for (int64_t i = 0; i < preLoadSize; i++) {
        param.preLoadDim1ArrCompact[i] = static_cast<uint32_t>(param.tensorListDim1[i]);
    }
    tilingData.arrays.set_preLoadDim1(param.preLoadDim1ArrCompact);
    tilingData.set_isNonContiguous(static_cast<int16_t>(param.isNonContiguous ? 1 : 0));
    tilingData.set_isGather(param.isGather);
    tilingData.set_isRowConcat(param.isRowConcat);
    tilingData.set_gatherSeg(param.gatherSeg);
    tilingData.set_rowConcatSegNum(param.rowConcatSegNum);
    if (param.isNonContiguous) {
        for (size_t i = 0; i < std::min(param.strideList.size(), static_cast<size_t>(NON_CON_TENSOR_SIZE)); i++) {
            param.strideListCompact[i] = static_cast<uint32_t>(param.strideList[i]);
        }
        tilingData.arrays.set_strideList(param.strideListCompact);
        for (size_t i = 0; i < std::min(param.concatDimList.size(), static_cast<size_t>(NON_CON_TENSOR_SIZE)); i++) {
            param.concatDimListCompact[i] = static_cast<uint32_t>(param.concatDimList[i]);
        }
        tilingData.arrays.set_concatDimList(param.concatDimListCompact);
    }
}

inline static void CalcTensorList(ConcatTilingParam& param, int64_t everyCoreData, int64_t rowsUsedCoreNum)
{
    int64_t curOffset = 0;
    int64_t curTensorOffset = 0;

    for (int16_t i = 0; i < param.tensorNum; i++) {
        if (param.blockStartTensorIdx.size() == param.blockEndTensorIdx.size()) {
            param.blockStartTensorIdx.push_back(i);
            param.blockStartTensorOffset.push_back(0);
        }
        while (curOffset + param.tensorListDim1[i] - curTensorOffset >= everyCoreData) {
            param.blockEndTensorIdx.push_back(i);
            curTensorOffset = curTensorOffset + everyCoreData - curOffset;
            param.blockEndTensorOffset.push_back(curTensorOffset);
            if (curTensorOffset == param.tensorListDim1[i]) {
                curOffset = 0;
                break;
            } else {
                param.blockStartTensorIdx.push_back(i);
                param.blockStartTensorOffset.push_back(curTensorOffset);
                curOffset = 0;
            }
        }
        if (curTensorOffset == 0) {
            curOffset += param.tensorListDim1[i];
        } else {
            curOffset = param.tensorListDim1[i] - curTensorOffset;
        }
        curTensorOffset = 0;
    }
    if (curOffset != 0) {
        param.blockEndTensorIdx.push_back(param.tensorNum - 1);
        param.blockEndTensorOffset.push_back(param.tensorListDim1[param.tensorNum - 1]);
    }

    DupTensor(param.startTensorIdx, param.blockStartTensorIdx, rowsUsedCoreNum);
    DupTensor(param.endTensorIdx, param.blockEndTensorIdx, rowsUsedCoreNum);
    DupTensor(param.startTensorOffset, param.blockStartTensorOffset, rowsUsedCoreNum);
    DupTensor(param.endTensorOffset, param.blockEndTensorOffset, rowsUsedCoreNum);
}

inline static bool IsEnableb8ToB16(const ConcatTilingParam& param)
{
    // b8 dim1为偶数 不对齐场景可升为b16处理
    if (param.isNonContiguous) {
        // gather/rowconcat 泛化分支的单位系（行距 strideList[b]/段长 gatherSeg/
        // 合轴 stride）与 b8->b16 折算的 dim1/2 列宽语义不兼容（I 为奇数时无解），
        // 1B 原生 gather 模板路径已验证正确，此处禁用折算
        if (param.isGather == 1 || param.isRowConcat == 1) {
            return false;
        }
        if (param.dtypeSize != B8_BYTES || param.inputShapeSame != 1 || param.sameShapeTensorDim1 % DIGIT_TWO != 0 ||
            param.strideList[0] % DIGIT_TWO != 0) {
            return false;
        }
        for (const auto& tensorSize : param.tensorListDim1) {
            if (tensorSize % DIGIT_TWO != 0) {
                return false;
            };
        }
        for (int16_t i = 0; i < param.tensorNum; ++i) {
            if (param.strideList[i] % DIGIT_TWO != 0) {
                return false;
            }
        }
    } else {
        if (param.dtypeSize != B8_BYTES || param.inputShapeSame != 1 || param.sameShapeTensorDim1 % DIGIT_TWO != 0) {
            return false;
        }
        for (const auto& tensorSize : param.tensorListDim1) {
            if (tensorSize % DIGIT_TWO != 0) {
                return false;
            };
        }
    }
    return true;
}

static ge::graphStatus PreProcessForNoAlign(ConcatTilingParam& param)
{
    // gather/rowconcat 泛化分支强制走 NoAlignDiffShape 模板（222x/322x/122x），该模板的
    // 8 字节实例按 uint32 折算视角消费数据（存量设计：老 8B noalign 场景必然经此处折算）。
    // 因此 8B 且 isGather/isRowConcat 时即使 32B 对齐也必须折算，否则 tilingData 按 8B
    // 元素数下发而 kernel 按 4B 消费，产生 8B 元素高低 32 位错位
    bool forceB64Fold = (param.isGather == 1 || param.isRowConcat == 1) && param.dtypeSize == B64_BYTES;
    if (param.isAllTensorAlign == 1 && !forceB64Fold) {
        return ge::GRAPH_SUCCESS;
    }
    if (param.dtypeSize == B64_BYTES) {
        // b64 不对齐场景降为b32处理
        param.sameShapeTensorDim1 *= DIGIT_TWO;
        param.dtypeSize = B32_BYTES;
        param.leastCopyNumber = MIN_RESERVED_SIZE / param.dtypeSize;
        param.everyBlockNumber = BLOCK_SIZE / param.dtypeSize;
        param.catDim1 *= DIGIT_TWO;
        for (auto& tensorSize : param.tensorListDim1) {
            tensorSize *= DIGIT_TWO;
        }
        for (auto& tensorSize : param.mergeTensorList) {
            tensorSize[1] *= DIGIT_TWO;
        }
        if (param.isNonContiguous) {
            for (int16_t i = 0; i < param.tensorNum; ++i) {
                param.strideList[i] *= DIGIT_TWO;
            }
        }
        // b64 降 b32 时 gather/rowconcat 泛化分支的新增参数同步 ×2 折算：
        // kernel 侧按 4B 元素视角消费（seg 段长/段间 stride 均为元素数），
        // 漏折算会导致段长减半、8B 元素高低 32 位错位
        if (param.isGather && param.gatherSeg > 0) {
            param.gatherSeg *= DIGIT_TWO;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (IsEnableb8ToB16(param)) {
        // b8 dim1为偶数 不对齐场景升为b16处理
        param.sameShapeTensorDim1 /= DIGIT_TWO;
        param.dtypeSize = B16_BYTES;
        param.leastCopyNumber = MIN_RESERVED_SIZE / param.dtypeSize;
        param.everyBlockNumber = BLOCK_SIZE / param.dtypeSize;
        param.catDim1 /= DIGIT_TWO;
        for (auto& tensorSize : param.tensorListDim1) {
            tensorSize /= DIGIT_TWO;
        }
        for (auto& tensorSize : param.mergeTensorList) {
            tensorSize[1] /= DIGIT_TWO;
        }
        if (param.isNonContiguous) {
            for (int16_t i = 0; i < param.tensorNum; ++i) {
                param.strideList[i] /= DIGIT_TWO;
            }
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

inline static std::vector<int64_t> FindUniqueCut(int64_t coreNum)
{
    std::vector<int64_t> candidateSet;
    int64_t upBound = static_cast<int64_t>(std::ceil(std::sqrt(coreNum) + 1.0));
    for (int64_t m = 1; m < upBound; m++) {
        int64_t y = coreNum / m;
        candidateSet.push_back(m);
        candidateSet.push_back(y);
    }
    return candidateSet;
}

static std::pair<int64_t, int64_t> AutoBlockTiling(int64_t rows, int64_t cols, int64_t coreNum)
{
    std::vector<int64_t> candidateSet = FindUniqueCut(coreNum);
    std::vector<std::vector<int64_t>> allTiling;
    for (int64_t m : candidateSet) {
        if (m > rows) {
            continue;
        }
        int64_t mPart = (rows + m - 1) / m;
        int64_t n = coreNum / m;
        if (n > cols) {
            continue;
        }
        int64_t nPart = (cols + n - 1) / n;
        int64_t delta = mPart * nPart;
        if (m * n == coreNum) {
            if (rows % m == 0 && cols % n == 0) {
                delta = 0;
            } else if (rows % m == 0) {
                delta = delta - mPart * (cols - nPart * (n - 1));
            } else if (cols % n == 0) {
                delta = delta - nPart * (rows - mPart * (m - 1));
            } else {
                delta = delta - (cols - nPart * (n - 1)) * (rows - mPart * (m - 1));
            }
        }
        allTiling.push_back({m, n, m * n, delta});
    }
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        return std::make_pair(a[DIGIT_THREE], -a[DIGIT_ONE]) < std::make_pair(b[DIGIT_THREE], -b[DIGIT_ONE]);
    });
    if (allTiling.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(allTiling[0][0], allTiling[0][DIGIT_ONE]);
}

static bool IsEnableUsedSimt(ConcatTilingParam& param)
{
    int64_t totalDataNum = param.catDim0 * param.catDim1 * param.dtypeSize;
    int64_t useCoreNum = std::min(static_cast<int64_t>(param.tensorNum), param.totalCoreNum);
    int64_t maxDim1 = param.tensorListDim1[0];
    int64_t minDim1 = param.tensorListDim1[0];

    if (param.isNonContiguous) {
        return false;
    }

    // 总数据量大于 64K * 使用核数，不使用simt模板
    if (totalDataNum >= useCoreNum * SIMT_PER_CORE_THRESHOLD) {
        return false;
    }

    // 合轴后对齐场景和相同shape场景，不使用simt模板
    if (param.isAllTensorAlign == 1 || param.inputShapeSame == 1) {
        return false;
    }

    // 输入tensor的个数对核数取模，余数小于等于核数的一半场景，不使用simt模板
    if (static_cast<int64_t>(param.tensorNum) % (param.totalCoreNum + 1) <= (param.totalCoreNum / DIGIT_TWO)) {
        return false;
    }

    // tensor数目大于128不使用simt模板
    if (param.tensorNum > TILING_COLS_OFFSET_LENGTH) {
        return false;
    }

    // 数据量大于单个tensor数据量大于1024，波动在2倍以上的数据不使用simt模板
    for (auto dim1Ptr = param.tensorListDim1.begin(); dim1Ptr != param.tensorListDim1.end(); ++dim1Ptr) {
        maxDim1 = std::max(maxDim1, *dim1Ptr);
        minDim1 = std::min(minDim1, *dim1Ptr);
        if (maxDim1 / minDim1 >= DIGIT_TWO && maxDim1 * param.catDim0 * param.dtypeSize > SIMT_COMPARE_THRESHOLD) {
            return false;
        }
    }

    return true;
}

inline static void CalcTensorColsOffset(ConcatTilingParam& param)
{
    // 计算每个tensor行方向的数据偏移个数，从第0个开始记录，即如果是tensor 1的偏移要使用tensorColsOffset[0]数值
    int32_t curConcatDimOffset = 0;
    int32_t digitSize = param.dtypeSize > B64_BYTES ? param.dtypeSize / B64_BYTES : DIGIT_ONE;
    for (auto dimPtr = param.tensorListDim1.begin(); dimPtr != param.tensorListDim1.end(); ++dimPtr) {
        curConcatDimOffset = static_cast<int32_t>(curConcatDimOffset + *dimPtr * digitSize);
        param.tensorColsOffset.push_back(curConcatDimOffset);
    }
}

inline static void SetTensorColsOffset(ConcatTilingDataForSimt& tilingData, ConcatTilingParam& param)
{
    int32_t tilingList[TILING_COLS_OFFSET_LENGTH];

    std::copy(param.tensorColsOffset.begin(), param.tensorColsOffset.end(), tilingList);
    tilingData.arrays.set_tensorColsOffset(tilingList);
}

static inline void PrintSimtTilingData(ConcatTilingDataForSimt& tilingData)
{
    OP_LOGI("[Concat]", "tensorNumPerCore: %d, get_tensorNum: %d,catDim0: %d,catDim1: %d",
            tilingData.get_tensorNumPerCore(), tilingData.get_tensorNum(), tilingData.get_catDim0(),
            tilingData.get_catDim1());
}

static ge::graphStatus TilingForConcatDSimt(gert::TilingContext* context, ConcatTilingParam& param)
{
    ConcatTilingDataForSimt tilingData;
    int64_t dtypeSize = param.dtypeSize;
    int64_t catDim1 = static_cast<int64_t>(param.catDim1);
    // 大于B64的数据类型使用B64类型计算,concat轴数据总量不变，数据个数需翻倍
    catDim1 = dtypeSize > B64_BYTES ? (dtypeSize / B64_BYTES) * catDim1 : catDim1;
    dtypeSize = dtypeSize > B64_BYTES ? B64_BYTES : dtypeSize;

    // 计算simt模板使用的核数，每核处理的tensor个数，计算tilingKey
    param.usedCoreNum = std::min(static_cast<int64_t>(param.tensorNum), param.totalCoreNum);
    param.tensorNumPerCore = (static_cast<int64_t>(param.tensorNum) + param.usedCoreNum - 1) / param.usedCoreNum;
    param.tilingKey = SIMT_TILINGKEY_PREFIX + dtypeSize;

    // 计算每个tensor行方向上的偏移
    CalcTensorColsOffset(param);

    // 设置tilingData的值
    tilingData.set_tensorNumPerCore(param.tensorNumPerCore);
    tilingData.set_tensorNum(static_cast<int32_t>(param.tensorNum));
    tilingData.set_catDim0(static_cast<int32_t>(param.catDim0));
    tilingData.set_catDim1(catDim1);
    SetTensorColsOffset(tilingData, param);

    PrintSimtTilingData(tilingData);
    context->SetBlockDim(param.usedCoreNum);
    context->SetTilingKey(param.tilingKey);
    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE;
    OP_CHECK_IF(ConcatSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SimtSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool TilingForPureCopy(ConcatTilingParam& param)
{
    // PureCopy kernel 通过 strideList 感知行 stride，支持断点轴==strideDim 的老非连续场景；
    // gather/rowconcat 的 strideList 语义不同（段间 stride/合轴 stride），必须排除。
    if (param.isNonContiguous && (param.isGather == 1 || param.isRowConcat == 1)) {
        return false;
    }
    param.usedCoreNum = std::min(param.totalCoreNum,
                                 (param.catDim0 * param.catDim1 + EVERY_CORE_THRESHOLD - 1) / EVERY_CORE_THRESHOLD);
    int64_t nCols = (param.catDim1 + LEAST_BLOCK_BYTES - 1) / LEAST_BLOCK_BYTES;
    int64_t mRows = param.catDim0;
    int64_t rowsCutPart = 0;
    int64_t colsCutPart = 0;
    std::tie(rowsCutPart, colsCutPart) = AutoBlockTiling(mRows, nCols, param.usedCoreNum);
    if (rowsCutPart == 0 || colsCutPart == 0) {
        return false;
    }
    param.ubFactorDim0 = (param.catDim0 + rowsCutPart - 1) / rowsCutPart;
    param.tailUbFactorDim0 = param.catDim0 - param.ubFactorDim0 * (rowsCutPart - 1);
    param.ubFactorDim1 = (param.catDim1 + colsCutPart - 1) / colsCutPart;
    param.tailUbFactorDim1 = param.catDim1 - param.ubFactorDim1 * (colsCutPart - 1);
    if (param.tailUbFactorDim0 < 0 || param.tailUbFactorDim1 < 0) {
        return false;
    }
    int64_t everyBlockCols = (param.catDim1 + colsCutPart - 1) / colsCutPart;
    if (colsCutPart > 1) {
        param.blockSplitAxis = 1;
        CalcTensorList(param, everyBlockCols, rowsCutPart);
    } else {
        param.blockSplitAxis = 0;
    }
    bool isUsedPureCopy = IsEnablePureCopyTemplate(param, rowsCutPart, colsCutPart);
    if (isUsedPureCopy) {
        if (ENABLE_DB) {
            param.ubSize = param.ubSize / DIGIT_TWO;
        }
        param.bufferSize = param.ubSize / param.dtypeSize;
        // 判断是否可用 compact
        bool canCompact = true;
        for (const auto& dim1 : param.tensorListDim1) {
            if (static_cast<uint64_t>(dim1) >= COMPACT_THRESHOLD) {
                canCompact = false;
                break;
            }
        }
        if (param.blockSplitAxis == 0) {
            param.tilingKey = canCompact ? PURE_COPY_NO_SPLIT_DIM1_COMPACT_TILINGKEY :
                                           PURE_COPY_NO_SPLIT_DIM1_TILINGKEY;
        } else {
            param.tilingKey = canCompact ? PURE_COPY_SPLIT_DIM1_COMPACT_TILINGKEY : PURE_COPY_SPLIT_DIM1_TILINGKEY;
        }
        param.uoDim0 = rowsCutPart;
        param.uoDim1 = colsCutPart;
        param.inputShapeSame = 0;
        GetTensorSameDim1(param);
    } else {
        param.blockStartTensorIdx.clear();
        param.blockEndTensorIdx.clear();
        param.blockStartTensorOffset.clear();
        param.blockEndTensorOffset.clear();
    }
    return isUsedPureCopy;
}

inline static ge::graphStatus DoTiling(gert::TilingContext* context, ConcatTilingParam& param)
{
    if (param.isEmpty) {
        param.usedCoreNum = 0;
        return ge::GRAPH_SUCCESS;
    }
    if (TilingForPureCopy(param)) {
        // 先尝试纯搬运模板
        return ge::GRAPH_SUCCESS;
    }
    bool forceB64FoldCall = (param.isGather == 1 || param.isRowConcat == 1) && param.dtypeSize == B64_BYTES;
    if ((param.isAllTensorAlign == 0 && (param.dtypeSize == B64_BYTES || param.dtypeSize == B8_BYTES)) ||
        forceB64FoldCall) {
        OP_CHECK_IF(PreProcessForNoAlign(param) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "check PreProcessForNoAlign failed"), return ge::GRAPH_FAILED);
    }
    if (ENABLE_DB) {
        param.ubSize = param.ubSize / HALF;
    }
    // ub切分,不切列
    OP_CHECK_IF(TilingUb(context, param) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "check tiling_ub failed"), return ge::GRAPH_FAILED);
    // block切分
    OP_CHECK_IF(TilingBlock(context, param) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "check tiling_block failed"), return ge::GRAPH_FAILED);
    if (param.blockSplitAxis == 1) {
        CalcTensorList(param, param.blockFactor * param.ubFactorDim1, param.uoDim0);
    }
    GenTilingKey(param);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static int64_t GetAxis(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis = attrs->GetAttrPointer<T>(PACK_ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    return *axis;
}

inline static ge::graphStatus IsPackDimValid(gert::TilingContext* context, int64_t& dim)
{
    auto inputShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, 0);
    gert::Shape inputShape = inputShapePtr->GetStorageShape();
    int64_t shapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    int64_t minAxis = (shapeSize + PACK_AXIS_DEFAULT_VALUE) * (-1);
    int64_t maxAxis = shapeSize;
    if (!(dim >= minAxis && dim <= maxAxis)) {
        return ge::GRAPH_FAILED;
    }
    // convert negative dim to positive dim
    if (dim < 0) {
        dim = dim + shapeSize + PACK_AXIS_DEFAULT_VALUE;
    }
    return ge::GRAPH_SUCCESS;
}

bool IsInvalidTypeForPack(const DataType dtype)
{
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT,  ge::DT_FLOAT16,   ge::DT_BF16,      ge::DT_UINT8,    ge::DT_INT8,        ge::DT_UINT16,
        ge::DT_INT16,  ge::DT_UINT32,    ge::DT_INT32,     ge::DT_UINT64,   ge::DT_INT64,       ge::DT_BOOL,
        ge::DT_DOUBLE, ge::DT_COMPLEX64, ge::DT_COMPLEX32, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN};
    bool isInvalidType = (supportedDtype.count(dtype) == 0);

    return isInvalidType;
}

ge::graphStatus CheckInputShapeSameForPack(gert::TilingContext* context)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(PACK_INPUT_IDX);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    if (inputNum < 1) {
        return ge::GRAPH_FAILED;
    }
    auto firstInputTensorShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, 0);
    gert::Shape firstInputTensorShape = firstInputTensorShapePtr->GetStorageShape();
    size_t firstInputTensorDimNum = firstInputTensorShape.GetDimNum();
    vector<int64_t> fisrtInputShapeList(firstInputTensorDimNum, 0);
    for (size_t i = 0; i < firstInputTensorDimNum; i++) {
        fisrtInputShapeList[i] = firstInputTensorShape.GetDim(i);
    }

    for (uint32_t i = 1; i < inputNum; ++i) {
        auto inputTensorShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, i);
        gert::Shape inputTensorShape = inputTensorShapePtr->GetStorageShape();
        size_t inputTensorDimNum = inputTensorShape.GetDimNum();
        vector<int64_t> inputShapeList(inputTensorDimNum, 0);
        for (size_t j = 0; j < inputTensorDimNum; j++) {
            inputShapeList[j] = inputTensorShape.GetDim(j);
            if (inputTensorShape.GetDim(j) != fisrtInputShapeList[j]) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void GetTensorListForPack(gert::TilingContext* context, ConcatTilingParam& param)
{
    auto inputTensorShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, 0);
    gert::Shape inputTensorShape = inputTensorShapePtr->GetStorageShape();
    size_t inputTensorDimNum = inputTensorShape.GetDimNum();
    vector<int64_t> inputShapeList(inputTensorDimNum, 0);
    for (size_t i = 0; i < inputTensorDimNum; i++) {
        inputShapeList[i] = inputTensorShape.GetDim(i);
    }

    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(PACK_INPUT_IDX);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();

    for (uint32_t i = 0; i < inputNum; ++i) {
        param.tensorList.push_back(inputShapeList);
    }
}

ge::graphStatus Tiling4PackToConcatForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PackToConcatForAscendC running begin");
    ConcatTilingParam param;
    param.dim = GetAxis<int64_t>(context);
    OP_CHECK_IF(IsPackDimValid(context, param.dim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "check pack_axis failed, please check pack_axis."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeSameForPack(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input_tensors", "different_shapes",
                                                       "The shapes of input tensors must be the same."),
                return ge::GRAPH_FAILED);
    auto inputDesc = context->GetDynamicInputDesc(PACK_INPUT_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        IsInvalidTypeForPack(inputDataType),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "input", Ops::Base::ToString(inputDataType).c_str(),
            "The dtype of input must be within the range [DT_UINT8, DT_INT8, DT_BOOL, DT_FLOAT, DT_INT32, DT_UINT32, "
            "DT_INT16, DT_FLOAT16, DT_BF16, DT_UINT16, DT_INT64, DT_UINT64, DT_DOUBLE, DT_COMPLEX32, DT_COMPLEX64, "
            "DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDtypeSize(context, param, PACK_INPUT_IDX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetDtypeSize failed."), return ge::GRAPH_FAILED);
    GetTensorListForPack(context, param);
    OP_CHECK_IF(CalcBaseTilingParam(context, param) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CalcBaseTilingParam failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoTiling(context, param) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "DoTiling failed."),
                return ge::GRAPH_FAILED);
    context->SetTilingKey(param.tilingKey);
    context->SetBlockDim(param.usedCoreNum);
    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    // 20001/20002 万位也是 2 但是原始版 PureCopy, 需排除; 20003/20004 是 compact PureCopy
    bool isCompact = (param.tilingKey / TEN_THOUSANDS_DIGITS) >= 2 &&
                     param.tilingKey != PURE_COPY_NO_SPLIT_DIM1_TILINGKEY &&
                     param.tilingKey != PURE_COPY_SPLIT_DIM1_TILINGKEY;
    if (param.blockSplitAxis == 1) {
        if (isCompact) {
            ConcatTilingDataCompact tilingData;
            SetTilingData<ConcatTilingDataCompact>(tilingData, param);
            SetTensorListTilingData(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingDataCompact>(context, tilingData);
        } else {
            ConcatTilingData tilingData;
            SetTilingData<ConcatTilingData>(tilingData, param);
            SetTensorListTilingData(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingData>(context, tilingData);
        }
    } else {
        if (isCompact) {
            ConcatTilingDataNoArrayCompact tilingData;
            SetTilingData<ConcatTilingDataNoArrayCompact>(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingDataNoArrayCompact>(context, tilingData);
        } else {
            ConcatTilingDataNoArray tilingData;
            SetTilingData<ConcatTilingDataNoArray>(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingDataNoArray>(context, tilingData);
        }
    }
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "PackSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetConcatDim(gert::TilingContext* context, ConcatTilingParam& param, int64_t dimIdx)
{
    if (dimIdx == INVLID_CONCAT_DIM_IDX) {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t* axis = attrs->GetAttrPointer<int64_t>(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, axis);
        param.dim = *axis;
    } else {
        auto concatDimPtr = context->GetRequiredInputDesc(dimIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, concatDimPtr);
        ge::DataType concatDimType = concatDimPtr->GetDataType();
        if (concatDimType == ge::DT_INT32) {
            OP_CHECK_IF(GetConcatDimInput<int32_t>(context, param, dimIdx) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "get concat_dim failed."), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(GetConcatDimInput<int64_t>(context, param, dimIdx) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "get concat_dim failed."), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

gert::Shape GetShapeByAll(const gert::TilingContext* context, bool isNonContiguous, int inputIdx, int index)
{
    auto inputTensorShapePtr = context->GetDynamicInputShape(inputIdx, index);
    if (isNonContiguous) {
        return inputTensorShapePtr->GetShape();
    } else {
        return inputTensorShapePtr->GetStorageShape();
    }
}

// 校验是否为全连续
bool IsAllContiguous(gert::TilingContext* context, ConcatTilingParam& param, int64_t inputIdx)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    param.tensorNum = anchorInstanceInfo->GetInstanceNum();
    for (int16_t i = 0; i < param.tensorNum; ++i) {
        bool isViewI = context->DynamicInputIsView(inputIdx, i);
        auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
        if (isViewI && nonStrideI != nullptr && nonStrideI->GetDimNum() > 0) {
            return false;
        }
    }
    return true;
}

ge::graphStatus CheckNonConBasic(gert::TilingContext* context, ConcatTilingParam& param)
{
    OP_CHECK_IF(
        param.tensorNum <= 1 || param.tensorNum > NON_CON_TENSOR_SIZE,
        OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "input_tensors", static_cast<int64_t>(param.tensorNum),
                                      ("within the range [2, " + std::to_string(NON_CON_TENSOR_SIZE) + "]").c_str()),
        return ge::GRAPH_FAILED);
    // dim=0 泛化（非连续 gather）场景 strideDim = dim-1 = -1 合法：
    // 断点轴 breakAxis >= dim=0 恒成立，走 gather 分支，strideDim 不作索引用
    OP_CHECK_IF(param.strideDim < 0 && param.dim != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "stride_dim",
                                                      std::to_string(param.strideDim).c_str(),
                                                      "The value of stride_dim must be greater than or equal to 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateDtypeConsistency(gert::TilingContext* context, int64_t inputIdx, int64_t tensorIdx,
                                                ge::DataType input0DataType)
{
    auto inputIDesc = context->GetDynamicInputDesc(inputIdx, tensorIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputIDesc);
    OP_CHECK_IF(
        inputIDesc->GetDataType() != input0DataType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "input_tensors",
            (Ops::Base::ToString(input0DataType) + ", " + Ops::Base::ToString(inputIDesc->GetDataType())).c_str(),
            "The dtypes of input_tensors must be the same."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 输入是否带有效 view stride（动态输入且 stride 非空）
inline static bool HasEffectiveStride(gert::TilingContext* context, int64_t inputIdx, int64_t i)
{
    bool isViewI = context->DynamicInputIsView(inputIdx, i);
    auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
    return isViewI && nonStrideI != nullptr && nonStrideI->GetDimNum() > 0;
}
// 返回非连续断点轴：
// -1：全连续（无断点）；-2：存在多个断点（不支持）；否则为唯一断点所在轴
static int64_t FindBreakAxis(const gert::Stride* nonStrideI, const vector<int64_t>& tensorShape, int64_t dim)
{
    int64_t size = static_cast<int64_t>(tensorShape.size());
    if (size == 0) {
        return -1;
    }
    int64_t breaks = 0;
    int64_t breakAxis = -1;
    if (nonStrideI->GetStride(size - 1) != 1) {
        breaks = 1;
        breakAxis = size - 1;
    }
    for (int64_t j = size - 2; j >= 0; j--) {
        // 仅 dim==0(gather 域)时对最外层 size==1 轴豁免断点投票:
        // 该轴 stride 不参与寻址, 物理连续的视图(如 (1,I)/stride(X,1))按连续处理;
        // dim!=0(legacy 域)时不豁免——保持断点投票走 legacy 轻模板(与基线行为一致,
        // 常规路径对微小 tensor 比 legacy 直通慢); 其余轴保持既有判定公式
        if (j == 0 && tensorShape[0] == 1 && dim == 0) {
            continue;
        }
        int64_t expected = nonStrideI->GetStride(j + 1) * tensorShape[j + 1];
        if (nonStrideI->GetStride(j) != expected) {
            if (breaks > 0) {
                return -2;
            }
            breaks = 1;
            breakAxis = j;
        }
    }
    return breakAxis;
}

static ge::graphStatus ValidateAndSetStride(gert::TilingContext* context, ConcatTilingParam& param, int64_t inputIdx,
                                            int64_t i)
{
    int64_t size = static_cast<int64_t>(param.tensorList[i].size());
    if (param.dimsMerged) {
        // 合轴（rowconcat）场景：tensorList 已被归一为 2 维 [rows, cols]。
        // strideList = 行间源stride（断点轴物理stride），
        // concatDimList = 每个tensor的完整cols（与非rowconcat场景一致,
        // kernel用concatDimList*sameShapeTensorDim1得到dim1）。
        // 段间 stride 由 kernel 从 seg = dim1/rowConcatSegNum 推导（多中间轴时
        // 断点轴下一轴物理 stride 会读错源地址，kernel 已弃用该数组）
        param.strideList[i] = static_cast<uint64_t>(param.mergedStrideList[i][0]);
        param.concatDimList[i] = static_cast<uint64_t>(param.tensorList[i][1]);
        return ge::GRAPH_SUCCESS;
    }
    bool hasStride = HasEffectiveStride(context, inputIdx, i);
    if (!hasStride) {
        // 连续输入（非 view 或无有效 stride）：strideList 记录合轴后 dim0 行方向的自然 stride
        param.strideList[i] = param.strideDim >= 0 ? MergeDim(param.tensorList[i], param.strideDim + 1, size) :
                                                     MergeDim(param.tensorList[i], 0, size);
        param.concatDimList[i] = static_cast<uint32_t>(param.tensorList[i][param.dim]);
        return ge::GRAPH_SUCCESS;
    }

    auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
    int64_t breakAxis = FindBreakAxis(nonStrideI, param.tensorList[i], param.dim);
    OP_CHECK_IF(breakAxis == -2,
                OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride", "multiple_break", "single"),
                return ge::GRAPH_FAILED);

    // 既有非连续场景：断点轴必须为 dim-1，保持原有校验逻辑与 stride 语义完全不变
    if (param.strideDim >= 0 && breakAxis == param.strideDim) {
        OP_CHECK_IF(nonStrideI->GetStride(size - 1) != 1,
                    OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride",
                                               std::to_string(nonStrideI->GetStride(size - 1)).c_str(), "1"),
                    return ge::GRAPH_FAILED);
        for (int64_t j = size - 2; j >= 0; j--) {
            if (param.strideDim != j) {
                OP_CHECK_IF(
                    nonStrideI->GetStride(j) != nonStrideI->GetStride(j + 1) * param.tensorList[i][j + 1],
                    OP_LOGE_FOR_INVALID_STRIDE(
                        context->GetNodeName(), "input_stride", std::to_string(nonStrideI->GetStride(j)).c_str(),
                        std::to_string(nonStrideI->GetStride(j + 1) * param.tensorList[i][j + 1]).c_str()),
                    return ge::GRAPH_FAILED);
            }
        }
        param.strideList[i] = static_cast<uint64_t>(nonStrideI->GetStride(param.strideDim));
        param.concatDimList[i] = static_cast<uint32_t>(param.tensorList[i][param.dim]);
        return ge::GRAPH_SUCCESS;
    }

    // 全连续 view（无断点）：按连续语义处理
    if (breakAxis == -1) {
        param.strideList[i] = param.strideDim >= 0 ? MergeDim(param.tensorList[i], param.strideDim + 1, size) :
                                                     MergeDim(param.tensorList[i], 0, size);
        param.concatDimList[i] = static_cast<uint32_t>(param.tensorList[i][param.dim]);
        return ge::GRAPH_SUCCESS;
    }

    // 新增 slice pattern（单根轴）场景：全局唯一非连续轴，所有输入必须共享同一轴，轴维度不限
    if (param.slicePatternAxis == -1) {
        param.slicePatternAxis = breakAxis;
    }
    OP_CHECK_IF(param.slicePatternAxis != breakAxis,
                OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride", std::to_string(breakAxis).c_str(),
                                           std::to_string(param.slicePatternAxis).c_str()),
                return ge::GRAPH_FAILED);

    // 断点以下的轴必须连续（末维向前到断点上方），仅当断点位于末维时放开"末维 stride == 1"校验
    for (int64_t j = size - 2; j > breakAxis; j--) {
        OP_CHECK_IF(nonStrideI->GetStride(j) != nonStrideI->GetStride(j + 1) * param.tensorList[i][j + 1],
                    OP_LOGE_FOR_INVALID_STRIDE(
                        context->GetNodeName(), "input_stride", std::to_string(nonStrideI->GetStride(j)).c_str(),
                        std::to_string(nonStrideI->GetStride(j + 1) * param.tensorList[i][j + 1]).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (breakAxis != size - 1) {
        OP_CHECK_IF(nonStrideI->GetStride(size - 1) != 1,
                    OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride",
                                               std::to_string(nonStrideI->GetStride(size - 1)).c_str(), "1"),
                    return ge::GRAPH_FAILED);
    }

    // concatDimList：每个输入在 concat 轴的 size（维度不限）
    param.concatDimList[i] = static_cast<uint32_t>(param.tensorList[i][param.dim]);
    if (breakAxis >= param.dim) {
        // 末维断点（末维 stride != 1）不进非连续适配，报错回退连续逻辑
        OP_CHECK_IF(breakAxis == size - 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "input_stride", std::to_string(breakAxis).c_str(),
                        "Slice pattern on the last axis is unsupported; only non-last-axis break is supported."),
                    return ge::GRAPH_FAILED);
        // gather 场景：断点位于 concat 轴及其后，dim1 段内部不连续。
        // 每 tensor = rows(固定) x segNum_i 段 x seg 连续元素；
        // strideList=断点轴物理 stride(段间源 stride)；concatDimList=concat轴size(与非gather一致,
        // kernel乘sameShapeTensorDim1得到完整dim1)。
        param.isGather = 1;
        param.gatherSeg = MergeDim(param.tensorList[i], breakAxis + 1, size);
        param.strideList[i] = static_cast<uint64_t>(nonStrideI->GetStride(breakAxis));
        param.concatDimList[i] = static_cast<uint32_t>(param.tensorList[i][param.dim]);
    } else {
        // 断点 < concat 轴：dim1 段连续，走既有"rows x cols + 行 stride(DataCopyPad)"搬运。
        // 仅放行断点 == concat 轴-1(legacy 位置): 严格更早的断点使 2D 行块内行距非均匀,
        // 统一 stride[dim-1] 行距模型不适用; 全 view 输入已由 MergeDimsForEarlyBreakAxis
        // 合轴提前返回, 混合输入(连续成员+早断点 view)无法合轴, 落入此分支会在 b 轴
        // 换行处读错源地址, 恢复改动前的拒绝语义
        OP_CHECK_IF(
            breakAxis < param.dim - 1,
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride", std::to_string(breakAxis).c_str(),
                                       "break axis earlier than concat axis - 1 is unsupported"),
            return ge::GRAPH_FAILED);
        param.strideList[i] = static_cast<uint64_t>(nonStrideI->GetStride(param.strideDim));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateSmallBagScenario(gert::TilingContext* context, const ConcatTilingParam& param, int64_t i)
{
    int64_t allData = MergeDim(param.tensorList[i], 0, param.tensorList[i].size());
    if (!(param.tensorListDim1[i] * param.dtypeSize >= SMALL_BAG) &&
        !(param.strideList[i] * param.dtypeSize > SMALL_BAG) &&
        !(allData * param.dtypeSize < param.totalCoreNum * ALL_DATA_SMALL)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "tensorListDim1", std::to_string(param.tensorListDim1[i] * param.dtypeSize).c_str(),
            ("The combined size of the concat dim and subsequent dim must be at least " + std::to_string(SMALL_BAG) +
             " bytes.")
                .c_str());
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "strideList", std::to_string(param.strideList[i] * param.dtypeSize).c_str(),
            ("The stride of the non contiguous axis must be greater than " + std::to_string(SMALL_BAG) + " bytes.")
                .c_str());
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "allData", std::to_string(allData * param.dtypeSize).c_str(),
            ("The total data size must be less than " + std::to_string(param.totalCoreNum * ALL_DATA_SMALL) + " bytes.")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckNonContiguous(gert::TilingContext* context, ConcatTilingParam& param, int64_t inputIdx)
{
    if (CheckNonConBasic(context, param) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto input0Desc = context->GetDynamicInputDesc(inputIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
    auto input0DataType = input0Desc->GetDataType();

    param.slicePatternAxis = -1;
    param.isGather = 0;
    param.gatherSeg = 0;
    for (int64_t i = 0; i < param.tensorNum; i++) {
        if (ValidateDtypeConsistency(context, inputIdx, i, input0DataType) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (ValidateAndSetStride(context, param, inputIdx, i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (ValidateSmallBagScenario(context, param, i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // gather(dim=0) 混合输入（连续 tensor + 非连续 view，如宿主分批后 concat 中间结果
    // 与尾批 tensor 拼接）：连续 tensor 的行距应等于断点后连续块 I（= gatherSeg），
    // ValidateAndSetStride 对全连续 view（有 stride 但 breakAxis==-1）按合轴填的是
    // 全量元素数、对无 stride 输入同理，两者都会导致 kernel 行距放大造成源越界/错位，
    // 此处统一回填修正
    if (param.isGather && param.gatherSeg > 0) {
        for (int64_t i = 0; i < param.tensorNum; i++) {
            auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
            bool isContigTensor = !HasEffectiveStride(context, inputIdx, i);
            if (!isContigTensor && nonStrideI != nullptr) {
                isContigTensor = (FindBreakAxis(nonStrideI, param.tensorList[i], param.dim) == -1);
            }
            if (isContigTensor) {
                param.strideList[i] = static_cast<uint64_t>(param.gatherSeg);
            }
        }
    }

    param.isNonContiguous = true;
    return ge::GRAPH_SUCCESS;
}

// 合轴（rowconcat 场景）：断点轴 < dim-1 时，将断点及之上各轴合并为 dim0（行数）、
// 断点之下各轴合并为 dim1（列数），并归一化 stride，使后续逻辑与"断点 == dim-1"既有场景一致。
// 例：[2,3,4] dim=2 断点 0 → 合轴为 [2,12]（stride 24,1），dim 变为 1。
static ge::graphStatus MergeDimsForEarlyBreakAxis(gert::TilingContext* context, ConcatTilingParam& param,
                                                  int64_t inputIdx)
{
    if (!param.isNonContiguous || param.dim < 2) {
        return ge::GRAPH_SUCCESS;
    }
    // 探测全局唯一断点：仅当所有输入断点一致且严格早于 dim-1 时才合轴
    int64_t globalBreakAxis = -2;
    for (int64_t i = 0; i < param.tensorNum; i++) {
        auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
        if (nonStrideI == nullptr || nonStrideI->GetDimNum() == 0) {
            return ge::GRAPH_SUCCESS;
        }
        int64_t b = FindBreakAxis(nonStrideI, param.tensorList[i], param.dim);
        if (b < 0) {
            return ge::GRAPH_SUCCESS;
        }
        if (globalBreakAxis == -2) {
            globalBreakAxis = b;
        } else if (globalBreakAxis != b) {
            return ge::GRAPH_SUCCESS;
        }
    }
    if (globalBreakAxis < 0 || globalBreakAxis >= param.dim - 1) {
        return ge::GRAPH_SUCCESS;
    }
    param.dimsMerged = true;
    param.isRowConcat = 1;
    // 每行段数：断点轴与 concat 轴之间各轴乘积（所有输入一致）
    param.rowConcatSegNum = MergeDim(param.tensorList[0], globalBreakAxis + 1, param.dim);
    param.mergedStrideList.clear();
    for (int64_t i = 0; i < param.tensorNum; i++) {
        auto& shp = param.tensorList[i];
        int64_t rows = MergeDim(shp, 0, globalBreakAxis + 1);
        int64_t cols = MergeDim(shp, globalBreakAxis + 1, static_cast<int64_t>(shp.size()));
        auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
        // 行间源 stride（断点轴物理 stride）+ 段间源 stride（断点轴下一轴物理 stride）
        param.mergedStrideList.push_back(
            {nonStrideI->GetStride(globalBreakAxis), nonStrideI->GetStride(globalBreakAxis + 1)});
        param.strideList[i] = static_cast<uint64_t>(nonStrideI->GetStride(globalBreakAxis));
        param.concatDimList[i] = static_cast<uint32_t>(cols);
        shp = std::vector<int64_t>{rows, cols};
    }
    param.dim = 1;
    param.strideDim = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCommon(gert::TilingContext* context, int64_t inputIdx, int64_t dimIdx)
{
    ConcatTilingParam param;
    // get dim
    OP_CHECK_IF(GetConcatDim(context, param, dimIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get concat_dim failed."), return ge::GRAPH_FAILED);
    param.isNonContiguous = !(IsAllContiguous(context, param, inputIdx));
    OP_CHECK_IF(IsDimValid(context, param.dim, inputIdx, param.isNonContiguous, param.strideDim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "check concat_dim failed, please check concat_dim."),
                return ge::GRAPH_FAILED);
    auto inputDesc = context->GetDynamicInputDesc(inputIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        IsInvalidType(inputDataType),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "input", Ops::Base::ToString(inputDataType).c_str(),
            "The dtype of input must be within the range [DT_UINT8, DT_INT8, DT_BOOL, DT_FLOAT, DT_INT32, DT_UINT32, "
            "DT_INT16, DT_FLOAT16, DT_BF16, DT_UINT16, DT_INT64, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_HIFLOAT8, "
            "DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDtypeSize(context, param, inputIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetDtypeSize failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetTensorList(context, param, inputIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetTensorList failed."), return ge::GRAPH_FAILED);
    // 合轴：断点轴 < dim-1 的场景归一为 2 维（dim0=断点及之上，dim1=断点之下），避免新增模板
    OP_CHECK_IF(MergeDimsForEarlyBreakAxis(context, param, inputIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "MergeDimsForEarlyBreakAxis failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsShapeValid(context, param.tensorList, param.dim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "check input shape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcBaseTilingParam(context, param) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CalcBaseTilingParam failed."), return ge::GRAPH_FAILED);
    if (param.isNonContiguous) {
        OP_CHECK_IF(CheckNonContiguous(context, param, inputIdx) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "input tensor non contiguous validation failed."),
                    return ge::GRAPH_FAILED);
        // gather/rowconcat 场景下 concatDimList 记录的是 shape[dim](gather) 或合轴后 cols(rowconcat)，
        // sameShapeTensorDim1 需统一为"concat 轴之后各轴乘积"，否则 same-shape 输入会在 kernel 端二次相乘
        if (param.isGather || param.isRowConcat) {
            param.sameShapeTensorDim1 = MergeDim(param.tensorList[0], param.dim + 1, param.tensorList[0].size());
        }
    }
    // 处理simt模板
    if (IsEnableUsedSimt(param)) {
        return TilingForConcatDSimt(context, param);
    }
    OP_CHECK_IF(DoTiling(context, param) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "DoTiling failed."),
                return ge::GRAPH_FAILED);
    context->SetTilingKey(param.tilingKey);
    context->SetBlockDim(param.usedCoreNum);
    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    // 20001/20002 万位也是 2 但是原始版 PureCopy, 需排除; 20003/20004 是 compact PureCopy
    bool isCompact = (param.tilingKey / TEN_THOUSANDS_DIGITS) >= 2 &&
                     param.tilingKey != PURE_COPY_NO_SPLIT_DIM1_TILINGKEY &&
                     param.tilingKey != PURE_COPY_SPLIT_DIM1_TILINGKEY;
    if (param.blockSplitAxis == 1) {
        if (isCompact) {
            ConcatTilingDataCompact tilingData;
            SetTilingData<ConcatTilingDataCompact>(tilingData, param);
            SetTensorListTilingData(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingDataCompact>(context, tilingData);
        } else {
            ConcatTilingData tilingData;
            SetTilingData<ConcatTilingData>(tilingData, param);
            SetTensorListTilingData(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingData>(context, tilingData);
        }
    } else {
        if (isCompact) {
            ConcatTilingDataNoArrayCompact tilingData;
            SetTilingData<ConcatTilingDataNoArrayCompact>(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingDataNoArrayCompact>(context, tilingData);
        } else {
            ConcatTilingDataNoArray tilingData;
            SetTilingData<ConcatTilingDataNoArray>(tilingData, param);
            PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
            ret = ConcatSetTilingData<ConcatTilingDataNoArray>(context, tilingData);
        }
    }
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "ConcatSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ConcatForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ConcatForAscendC running begin");
    return TilingCommon(context, 1, 0);
}

ge::graphStatus TilingPrepareForConcat(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ConcatDCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0),
                OP_LOGE(context->GetNodeName(), "TilingPrepareForConcat Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "TilingPrepareForConcat Failed to get ub size."),
                return ge::GRAPH_FAILED);
    compileInfo->vectorLen = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF((compileInfo->vectorLen <= 0),
                OP_LOGE(context->GetNodeName(), "TilingPrepareForConcat Failed to get vectorLen."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Concat)
    .TilingInputsDataDependency({CONCAT_DIM_IDX})
    .Tiling(Tiling4ConcatForAscendC)
    .TilingParse<ConcatDCompileInfo>(TilingPrepareForConcat);
} // namespace optiling
