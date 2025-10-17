/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <dlfcn.h>
#include <set>
#include <utility>
#include "securec.h"

#include "graph/types.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"

#include "aclnn_npu_format_cast.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static constexpr size_t DIMS_TWO = 2;
static constexpr size_t DIMS_THREE = 3;
static constexpr size_t DIMS_FOUR = 4;
static constexpr size_t DIMS_FIVE = 5;
static constexpr size_t DIMS_SIX = 6;
static constexpr size_t DIMS_EIGHT = 8;
static constexpr int64_t BLOCK_SIZE = 32;

const std::set<std::pair<op::Format, op::Format>> kTransdataForwardFormatPairs = {
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {op::Format::FORMAT_NCL, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {op::Format::FORMAT_NCL, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_NCDHW, op::Format::FORMAT_NDC1HWC0},
    {op::Format::FORMAT_NCDHW, op::Format::FORMAT_FRACTAL_Z_3D},
};

static const std::initializer_list<DataType> ASCEND910_95_WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32};
static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8, DataType::DT_UINT8, DataType::DT_FLOAT, DataType::DT_FLOAT16,
    DataType::DT_BF16, DataType::DT_INT32, DataType::DT_UINT32};

static bool isNonQuantMatmulDtype(int dtype, op::Format dstFormat = op::Format::FORMAT_ND)
{
    // weight类型为FLOAT且dstFormat为FORMAT_FRACTAL_NZ_C0_16或FORMAT_FRACTAL_NZ_C0_32时，伪量化fp4场景
    return (dtype == ge::DT_FLOAT && dstFormat != op::Format::FORMAT_FRACTAL_NZ_C0_16 &&
            dstFormat != op::Format::FORMAT_FRACTAL_NZ_C0_32) ||
           dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

inline int64_t Ceil(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
        return INT64_MIN;
    }
    return ((x + y - 1) / y) * y;
}

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
        return INT64_MIN;
    }
    return (x + y - 1) / y;
}

static aclnnStatus ValidateNonQuantMatmulParams(
    [[maybe_unused]] int32_t additionalDtype, const gert::Shape& viewShape, size_t viewShapeDim,
    [[maybe_unused]] int32_t dstFormat)
{
    // 拦截mm非量化K=1场景, 防止后续matmul2Mul报错
    int64_t kDim = viewShape.GetDim(viewShapeDim - 2);
    OP_CHECK(
        kDim != 1,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Only support srcTensor's k Dim is not 1 when additionalDtype equals srcTensors's dtype."),
        return ACLNN_ERR_PARAM_INVALID);
    // mm非量化当前仅只支持2~6维
    OP_CHECK(
        viewShapeDim >= 2 && viewShapeDim <= 6,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Only support srcTensor's viewShapeDim is between 2 and 6 when "
            "additionalDtype equals srcTensors's dtype, current viewShapeDim: [%zu]",
            viewShapeDim),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateQuantMatmulParams(
    int32_t additionalDtype, [[maybe_unused]] const gert::Shape& viewShape, size_t viewShapeDim)
{
    OP_CHECK(
        additionalDtype == ge::DT_INT8 || additionalDtype == ge::DT_UINT8,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Only support additionalDtype is int8/uint8 when additionalDtype equals srcTensors's dtype and "
            "additionalDtype "
            "is not float16 or bfloat16, current additionalDtype: [%s].",
            op::ToString(static_cast<op::DataType>(additionalDtype)).GetString()),
        return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(
        viewShapeDim >= 2 && viewShapeDim <= 6,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Only support srcTensor's viewShapeDim is between 2 and 6 when additionalDtype equals srcTensors's dtype "
            "and is int8, current viewShapeDim: [%zu]",
            viewShapeDim),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateWeightQuantMatmulParams(
    DataType srcDtype, int32_t additionalDtype, size_t viewShapeDim, const int dstFormat)
{
    if (srcDtype == ge::DT_INT32) {
        OP_CHECK(
            additionalDtype == ge::DT_FLOAT16 || additionalDtype == ge::DT_BF16,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support additionalDtype is float16 or bfloat16 when srcTensors's dtype is int32, current "
                "additionalDtype: [%s].",
                op::ToString(static_cast<op::DataType>(additionalDtype)).GetString()),
            return ACLNN_ERR_PARAM_INVALID);
    } else if (srcDtype == ge::DT_FLOAT) {
        OP_CHECK(
            additionalDtype == ge::DT_FLOAT16 || additionalDtype == ge::DT_BF16 ||
                additionalDtype == ge::DT_FLOAT8_E4M3FN,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support additionalDtype is float16 or bfloat16 or float8_e4m3fn when srcTensors's dtype is "
                "float32, current additionalDtype: [%s].",
                op::ToString(static_cast<op::DataType>(additionalDtype)).GetString()),
            return ACLNN_ERR_PARAM_INVALID);
    }

    // WeightQuanBatchMatmul 场景拦截，仅支持srctensor的viewshape维度为2或3
    OP_CHECK(
        viewShapeDim == 2 || viewShapeDim == 3,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Only support srcTensor's viewShapeDim is 2 or 3 when additionalDtype is not equal to srcTensors's dtype, "
            "current viewShapeDim: [%zu]",
            viewShapeDim),
        return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(
        dstFormat == op::Format::FORMAT_FRACTAL_NZ,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Only support dstFormat is 29, when additionalDtype is not equal to srcTensors's dtype, current dstFormat: "
            "[%d].",
            dstFormat),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckCalculateSizeAndFormatInputs(
    const aclTensor* srcTensor, [[maybe_unused]] const int dstFormat, [[maybe_unused]] const int additionalDtype)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(
        socVersion == SocVersion::ASCEND910_95 || socVersion == SocVersion::ASCEND910_93 ||
            socVersion == SocVersion::ASCEND910B,
        OP_LOGW("Only support Ascend910_95/ASCEND910_93/ASCEND910B"), return ACLNN_ERR_RUNTIME_ERROR);
    OP_CHECK(
        srcTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "srcTensor is nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);

    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT(srcTensor, WEIGHT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus Check95NdToNzCalculateSizeAndFormatInputs(
    const aclTensor* srcTensor, const int dstFormat, const int additionalDtype)
{
    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT(srcTensor, ASCEND910_95_WEIGHT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    [[maybe_unused]] op::Format srcFormat = srcTensor->GetStorageFormat();
    auto srcDtype = srcTensor->GetDataType();
    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();

    // check input shape
    if (additionalDtype != static_cast<int>(srcDtype)) {
        aclnnStatus ret = ValidateWeightQuantMatmulParams(srcDtype, additionalDtype, viewShapeDim, dstFormat);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    } else {
        if (isNonQuantMatmulDtype(additionalDtype) && dstFormat == op::Format::FORMAT_FRACTAL_NZ) {
            aclnnStatus ret = ValidateNonQuantMatmulParams(additionalDtype, viewShape, viewShapeDim, dstFormat);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        } else {
            aclnnStatus ret = ValidateQuantMatmulParams(additionalDtype, viewShape, viewShapeDim);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
    }

    return ACLNN_SUCCESS;
}

static bool CheckFormatValid(DataType srcDtype, op::Format srcFormat, op::Format dstFormat)
{
    if (srcDtype == ge::DT_INT8 || srcDtype == ge::DT_UINT8) {
        // QuantBatchMatmul 拦截场景
        OP_CHECK(
            (srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL) &&
                dstFormat == op::Format::FORMAT_FRACTAL_NZ,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support srcFormat is ND/NCL and dstFormat is FRACTAL_NZ when srtDtype equals int8, which are "
                "[%s] and [%s].",
                op::ToString(srcFormat).GetString(), op::ToString(dstFormat).GetString()),
            return false);
    } else if (isNonQuantMatmulDtype(srcDtype, dstFormat)) {
        // 非量化Matmul 拦截场景
        OP_CHECK(
            (srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL) &&
                dstFormat == op::Format::FORMAT_FRACTAL_NZ,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support srcFormat is ND/NCL and dstFormat is FRACTAL_NZ when srtDtype equals float16 or "
                "bfloat16, which are [%s] and [%s].",
                op::ToString(srcFormat).GetString(), op::ToString(dstFormat).GetString()),
            return false);
    } else {
        // WeightQuantBatchMatmul 拦截场景
        OP_CHECK(
            (srcDtype == ge::DT_INT32 || srcDtype == ge::DT_FLOAT) && srcFormat == op::Format::FORMAT_ND &&
                (dstFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16 || dstFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support srcFormat is ND and dstFormat is FRACTAL_NZ_C0_16 or FRACTAL_NZ_C0_32 when srcDtype "
                "equals int32 or float32,, which are [%s] "
                "and [%s].",
                op::ToString(srcFormat).GetString(), op::ToString(dstFormat).GetString()),
            return false);
    }
    return true;
}

static aclnnStatus CheckGetWorkSpaceSizeInputs(const aclTensor* srcTensor, aclTensor* dstTensor)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(
        socVersion == SocVersion::ASCEND910_95 || socVersion == SocVersion::ASCEND910_93 ||
            socVersion == SocVersion::ASCEND910B,
        OP_LOGW("Only support Ascend910_95/ASCEND910_93/ASCEND910B"), return ACLNN_ERR_RUNTIME_ERROR);
    CHECK_RET(srcTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dstTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK(
        IsContiguous(srcTensor), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support srcTensor is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        IsContiguous(dstTensor), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support dstTensor is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_SUPPORT(srcTensor, WEIGHT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(dstTensor, WEIGHT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus Check95NdToNzGetWorkSpaceSizeInputs(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT(srcTensor, ASCEND910_95_WEIGHT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(dstTensor, ASCEND910_95_WEIGHT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    op::Format srcFormat = srcTensor->GetStorageFormat();
    auto srcViewShape = srcTensor->GetViewShape();
    auto srcviewShapeDim = srcViewShape.GetDimNum();
    DataType srcDtype = srcTensor->GetDataType();
    op::Format dstFormat = dstTensor->GetStorageFormat();
    auto storageShape = dstTensor->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    if (!CheckFormatValid(srcDtype, srcFormat, dstFormat)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (srcDtype == ge::DT_INT8 || srcDtype == ge::DT_UINT8 || isNonQuantMatmulDtype(srcDtype, dstFormat)) {
        // QuantBatchMatmul &&
        // 非量化matmul仅支持srcTensor的viewshape维度为2到6，转换后的dstTensor的storageshape维度为4到8
        OP_CHECK(
            srcviewShapeDim >= DIMS_TWO && srcviewShapeDim <= DIMS_SIX && storageShapeDim >= DIMS_FOUR &&
                storageShapeDim <= DIMS_EIGHT,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support srcViewShapeDim is between 2 and 6 and storageShapeDim is between 4 and 8 when srcDtype "
                "equals dstDtype, which are [%zu] and [%zu].",
                srcviewShapeDim, storageShapeDim),
            return ACLNN_ERR_PARAM_INVALID);
    } else {
        // WeightQuantBatchMatmul仅支持srcTensor的shape维度为2，转换后的dstTensor的shape的维度为4
        OP_CHECK(
            (srcviewShapeDim == DIMS_TWO || srcviewShapeDim == DIMS_THREE) &&
                (storageShapeDim == DIMS_FOUR || storageShapeDim == DIMS_FIVE),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Only support srcViewShapeDim is 2/3 and storageShapeDim is 4/5 when srcDtype is not equal to "
                "dstDtype, "
                "which are [%zu] and [%zu].",
                srcviewShapeDim, storageShapeDim),
            return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

bool isSupportedTransdataForwardPair(op::Format src, op::Format dst)
{
    return kTransdataForwardFormatPairs.count({src, dst});
}

aclnnStatus CalcNdToNz(
    const aclTensor* srcTensor, int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat)
{
    [[maybe_unused]] op::Format srcFormat = srcTensor->GetStorageFormat();
    DataType srcDtype = srcTensor->GetDataType();
    int64_t c0 = 16; // 默认NZ分型的c0为16

    // 根据A矩阵和B矩阵数据类型计算实际转换成NZ格式后的C0大小
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95 ||
        static_cast<op::DataType>(additionalDtype) == srcDtype) {
        // 当前要求C0 * ge::GetSizeByDataType(dtype) = 32B
        c0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
        *actualFormat = op::Format::FORMAT_FRACTAL_NZ;
    } else {
        // 当A矩阵数据类型大小为2B，C0 = 16
        if (ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype)) == 2) {
            *actualFormat = op::Format::FORMAT_FRACTAL_NZ_C0_16;
        }

        if (static_cast<op::DataType>(additionalDtype) == DataType::DT_FLOAT8_E4M3FN) {
            // A8场景，B32/1B=BLOCK_SIZE
            c0 = BLOCK_SIZE;
            *actualFormat = op::Format::FORMAT_FRACTAL_NZ_C0_32;
        }
    }

    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    int64_t srcTensorDimFirst = viewShape.GetDim(viewShapeDim - 2); // 倒数第2维为ND的shape第一维
    int64_t srcTensorDimLast = viewShape.GetDim(viewShapeDim - 1);  // 倒数第1维为ND的shape第二维
    *dstShapeSize =
        static_cast<uint64_t>(srcTensor->GetViewShape().GetDimNum()) + 2; // NZ维度大小固定为srcTensor的维度+2

    // 非报错情况下申请dstShape数组内存，由上层调用者释放
    try {
        *dstShape = new int64_t[*dstShapeSize]();
    } catch (std::bad_alloc& e) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for the NZ");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    for (uint64_t i = 0; i < *dstShapeSize; i++) {
        if (i == *dstShapeSize - 4) {        // 修改最后4维的NZ数据
            (*dstShape)[*dstShapeSize - 4] = // 当前Nz分型倒数第4维大小为CeilDiv(srcTensorDimLast, c0)
                CeilDiv(srcTensorDimLast, c0);
            (*dstShape)[*dstShapeSize - 3] = // 当前Nz分型倒数第3维大小固定为CeilDiv(srcTensorDimFirst, 16)
                CeilDiv(srcTensorDimFirst, 16);
            (*dstShape)[*dstShapeSize - 2] = 16; // 当前Nz分型倒数第2维大小固定为16
            (*dstShape)[*dstShapeSize - 1] = c0; // 当前Nz分型倒数第1维大小为C0
            break;
        }
        (*dstShape)[i] = viewShape.GetDim(i);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNd(
    const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
    int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    auto shapeDim = viewShape.GetDimNum();
    *dstShapeSize = shapeDim;
    std::vector<int64_t> shape(shapeDim);
    for (size_t i = 0; i < shapeDim; ++i) {
        shape[i] = viewShape.GetDim(i);
    }

    try {
        *dstShape = new int64_t[shapeDim];
        for (size_t i = 0; i < shapeDim; ++i) {
            (*dstShape)[i] = shape[i];
        }
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for ND");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    *actualFormat = op::Format::FORMAT_ND;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCDHWToNDC1HWC0(
    const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
    int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);
    int64_t C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(srcTensor->GetDataType()));
    int64_t C1 = (C + C0 - 1) / C0;

    *dstShapeSize = 6; // 6HD
    try {
        *dstShape = new int64_t[6]{N, D, C1, H, W, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NDC1HWC0");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NDC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCDHWToFZ3D(
    const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
    int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);
    int64_t C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(srcTensor->GetDataType()));
    int64_t N0 = 16; // 私有格式的分形要求
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t N1 = (N + N0 - 1) / N0;

    *dstShapeSize = 4; // 4HD
    try {
        *dstShape = new int64_t[4]{D * C1 * H * W, N1, N0, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for FZ3D");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_FRACTAL_Z_3D;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNCDHW(
    const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
    int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);

    *dstShapeSize = 5; // 5HD
    try {
        *dstShape = new int64_t[5]{N, C, D, H, W};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NCDHW");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NCDHW;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(
    const aclTensor* srcTensor, const int dstFormat, int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
    int* actualFormat)
{
    auto ret = CheckCalculateSizeAndFormatInputs(srcTensor, dstFormat, additionalDtype);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGW("Failed to check inputs"), return ACLNN_ERR_PARAM_INVALID);
    op::Format srcFormat = srcTensor->GetStorageFormat();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(
        (additionalDtype == -1 && (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93)) ||
            (additionalDtype != -1 && socVersion == SocVersion::ASCEND910_95),
        OP_LOGW("The current socVersion does not support additionalDtype."), return ACLNN_ERR_PARAM_INVALID);
    if (additionalDtype == -1) {
        additionalDtype = static_cast<int>(srcTensor->GetDataType());
    }
    if (dstFormat == op::Format::FORMAT_FRACTAL_NZ &&
        (srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL)) {
        // ASCEND910_95校验特殊场景
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
            auto retNdToNz = Check95NdToNzCalculateSizeAndFormatInputs(srcTensor, dstFormat, additionalDtype);
            OP_CHECK(retNdToNz == ACLNN_SUCCESS, OP_LOGW("Failed to check inputs"), return ACLNN_ERR_PARAM_INVALID);
        }
        return CalcNdToNz(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_NZ && dstFormat == op::Format::FORMAT_ND) {
        return CalcToNd(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NCDHW && dstFormat == op::Format::FORMAT_NDC1HWC0) {
        return CalcNCDHWToNDC1HWC0(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NDC1HWC0 && dstFormat == op::Format::FORMAT_NCDHW) {
        return CalcToNCDHW(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NCDHW && dstFormat == op::Format::FORMAT_FRACTAL_Z_3D) {
        return CalcNCDHWToFZ3D(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_Z_3D && dstFormat == op::Format::FORMAT_NCDHW) {
        return CalcToNCDHW(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    }
    OP_LOGW("aclnnNpuFormatCastCalculateSizeAndFormat unsupported format transformation");
    return ACLNN_ERR_RUNTIME_ERROR;
}

aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(
    const aclTensor* srcTensor, aclTensor* dstTensor, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckGetWorkSpaceSizeInputs(srcTensor, dstTensor);
    op::Format srcFormat = srcTensor->GetStorageFormat();
    op::Format dstFormat = dstTensor->GetStorageFormat();
    // ASCEND910_95校验特殊场景
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 &&
        dstFormat == op::Format::FORMAT_FRACTAL_NZ &&
        (srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL)) {
        ret = Check95NdToNzGetWorkSpaceSizeInputs(srcTensor, dstTensor);
    }
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to check aclnnNpuFormatCastGetWorkSpaceSizeInputs."),
        return ACLNN_ERR_PARAM_INVALID);

    L2_DFX_PHASE_1(aclnnNpuFormatCast, DFX_IN(srcTensor), DFX_OUT(dstTensor));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto formatTensor = const_cast<aclTensor*>(srcTensor);
    // 适配srcFormat为NCL的场景
    if (srcFormat == op::Format::FORMAT_NCL) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    if (isSupportedTransdataForwardPair(srcFormat, dstFormat)) {
        // 公有转私有Format
        formatTensor->SetOriginalShape(srcTensor->GetViewShape());
        formatTensor->SetStorageShape(srcTensor->GetViewShape());

        dstTensor->SetViewFormat(srcTensor->GetViewFormat());
        dstTensor->SetViewShape(srcTensor->GetViewShape());
        dstTensor->SetOriginalFormat(srcTensor->GetOriginalFormat());
        dstTensor->SetOriginalShape(srcTensor->GetOriginalShape());
    } else {
        // 私有转公有Format
        dstTensor->SetOriginalShape(dstTensor->GetViewShape());
        dstTensor->SetStorageShape(dstTensor->GetViewShape());

        formatTensor->SetViewFormat(dstTensor->GetViewFormat());
        formatTensor->SetViewShape(dstTensor->GetViewShape());
        formatTensor->SetOriginalFormat(dstTensor->GetOriginalFormat());
        formatTensor->SetOriginalShape(dstTensor->GetOriginalShape());
    }

    auto outTensor =
        const_cast<aclTensor*>(l0op::TransData(formatTensor, dstTensor->GetStorageFormat(), 1, uniqueExecutor.get()));
    CHECK_RET(outTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outTensor->SetStorageFormat(dstTensor->GetStorageFormat());
    auto viewCopyResult = l0op::ViewCopy(outTensor, dstTensor, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNpuFormatCast);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace
#ifdef __cplusplus
}
#endif