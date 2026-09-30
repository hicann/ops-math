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
 * \file aclnn_cat.cpp
 * \brief
 */
#include "aclnn_cat.h"
#include "concat_d.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr uint32_t MAX_UINT32_NUM = 4294967295;
constexpr uint32_t STRIDE_SIZE = 32;
constexpr uint32_t MAX_TENSOR_NUM = 64;
constexpr uint32_t SMALL_BAG = 128;
constexpr uint32_t SINGLE_CORE_PROCESS_SIZE = 8192;
constexpr int32_t DIM_TWO = 2;

constexpr size_t CAT_INPUT_NUM_32 = 32;
constexpr size_t CAT_INPUT_NUM_REGBASE_512 = 1536;
constexpr size_t CAT_INPUT_NUM_V2_512 = 1536;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,  DataType::DT_FLOAT16,   DataType::DT_INT16,
    DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16,
    DataType::DT_INT16,     DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE,
    DataType::DT_COMPLEX64, DataType::DT_BF16,  DataType::DT_BOOL};

// todo:应该就剩concatd的算子信息库还没搞完
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,    DataType::DT_INT32,       DataType::DT_INT64,           DataType::DT_FLOAT16,
    DataType::DT_INT16,    DataType::DT_INT8,        DataType::DT_UINT8,           DataType::DT_UINT16,
    DataType::DT_UINT32,   DataType::DT_UINT64,      DataType::DT_DOUBLE,          DataType::DT_COMPLEX64,
    DataType::DT_BF16,     DataType::DT_BOOL,        DataType::DT_FLOAT8_E4M3FN,   DataType::DT_FLOAT8_E5M2,
    DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E8M0, op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch npuArch)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (npuArch == NpuArch::DAV_2002 || npuArch == NpuArch::DAV_1001) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    } else if (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3002) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase(npuArch)) {
        return REGBASE_DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* y)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& dTypeSupportList = GetSupportDtypeList(npuArch);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!CheckType((*tensors)[i]->GetDataType(), dTypeSupportList)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s, should be in dtype support list %s.",
                    i, op::ToString((*tensors)[i]->GetDataType()).GetString(),
                    op::ToString(dTypeSupportList).GetString());
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(y, dTypeSupportList, return false);
    return true;
}

static bool CheckNotNull(const aclTensorList* tensors, const aclTensor* y)
{
    OP_CHECK_NULL(tensors, return false);

    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_NULL((*tensors)[i], return false);
    }
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckPromoteType(const aclTensorList* tensors, const aclTensor* y)
{
    op::DataType promoteType = (*tensors)[0]->GetDataType();
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        if (promoteType == DataType::DT_UNDEFINED) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor %lu dtype %s and dtype %s can not promote dtype.", i,
                    op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(promoteType).GetString());
            return false;
        }
        if (promoteType == DataType::DT_COMPLEX128) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "tensor dtype has been promoted to %s, which has not been implemented yet.",
                    op::ToString(promoteType).GetString());
            return false;
        }
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, y->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensorList* tensors, const aclTensor* y)
{
    op::Format format = (*tensors)[0]->GetStorageFormat();
    if (op::IsPrivateFormat(format)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only supports ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        if ((*tensors)[i]->GetStorageFormat() != format) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of tensors should be equal, tensor %lu [%s], tensor 0 [%s].", i,
                    op::ToString((*tensors)[i]->GetStorageFormat()).GetString(), op::ToString(format).GetString());
            return false;
        }
    }
    if (y->GetStorageFormat() != format) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, tensor 0 [%s] out [%s].",
                op::ToString(y->GetStorageFormat()).GetString(), op::ToString(y->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensorList* tensors, int64_t* realDim)
{
    OP_CHECK_MAX_DIM((*tensors)[0], MAX_SUPPORT_DIMS_NUMS, return false);
    op::Shape shape0 = (*tensors)[0]->GetViewShape();
    auto dimNum = (int64_t)shape0.GetDimNum();
    auto orgDim = *realDim;
    if (*realDim < 0) {
        (*realDim) += dimNum;
    }
    if ((*realDim) < 0 || (*realDim) >= dimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimnum %ld exceeds the dim range of the tensor %ld.", orgDim, dimNum);
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        op::Shape shape = (*tensors)[i]->GetViewShape();
        if (dimNum != (int64_t)shape.GetDimNum()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimnum of tensor %lu is [%zu], should be equal to tensor 0 [%ld].", i,
                    shape.GetDimNum(), dimNum);
            return false;
        }
        for (int64_t j = 0; j < dimNum; j++) {
            if (*realDim == j) {
                continue;
            }
            if (shape0.GetDim(j) != shape.GetDim(j)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim %ld of tensor %lu is [%ld], should be equal to tensor 0 [%ld].",
                        j, i, shape.GetDim(j), shape0.GetDim(j));
                return false;
            }
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, int64_t* realDim, const aclTensor* y)
{
    CHECK_RET(CheckNotNull(tensors, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(tensors, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckPromoteType(tensors, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(tensors, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(tensors, realDim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessOneTensor(const aclTensor* in, aclTensor* out, aclOpExecutor* executor)
{
    auto contiguous = l0op::Contiguous(in, executor);
    CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewIn = l0op::Cast(contiguous, out->GetDataType(), executor);
    if (viewIn == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Result type %s can't be cast to the desired output type %s.",
                op::ToString(contiguous->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyResult = l0op::ViewCopy(viewIn, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static bool CheckSocAndNonConBasic(op::FVector<const aclTensor*> tensors)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (!IsRegBase(npuArch)) {
        return false;
    }
    if (tensors.size() <= 1) {
        return false;
    }
    return true;
}

// 识别唯一的非尾断点轴: 尾轴 stride 须为 1, 除断点轴外其余轴须满足
// stride[j] = stride[j+1] * shape[j+1] 的连续关系, 且所有 tensor 的断点轴须同位

static bool FindBreakAxis(op::FVector<const aclTensor*> tensors, int64_t dimNum, int64_t realDim, int64_t& breakAxis)
{
    int64_t bk0 = -1;
    for (uint64_t i = 0; i < tensors.size(); i++) {
        op::Shape shapeI = tensors[i]->GetViewShape();
        op::Strides stridesI = tensors[i]->GetViewStrides();
        if (stridesI[dimNum - 1] != 1) {
            return false;
        }
        int64_t bkI = -1;
        for (int64_t j = dimNum - DIM_TWO; j >= 0; j--) {
            // 仅 dim==0(gather 域)时对最外层 size==1 轴豁免断点投票:
            // 该轴 stride 不参与寻址, 物理连续的视图(如 (1,I)/stride(X,1))按连续处理,
            // 避免被误路由到 gather 重模板(实测 0.81x 劣化);
            // dim!=0(legacy 域)时不豁免——(1,I)/stride(X,1) 保持断点投票走 legacy 轻模板
            // (与基线行为一致, 实测常规路径对微小 tensor 比 legacy 直通慢 0.58x)
            if (j == 0 && shapeI.GetDim(0) == 1 && realDim == 0) {
                continue;
            }
            if (stridesI[j] != stridesI[j + 1] * shapeI.GetDim(j + 1)) {
                if (bkI >= 0) {
                    return false;
                }
                bkI = j;
            }
        }
        if (bkI < 0) {
            continue;
        }
        if (bk0 < 0) {
            bk0 = bkI;
        } else if (bk0 != bkI) {
            return false;
        }
    }
    if (bk0 < 0) {
        return false;
    }
    breakAxis = bk0;
    return true;
}

// 该 tensor 的 stride 是否存在断点(与 FindBreakAxis 的判定公式一致, 含最外层 size=1 豁免)
static bool TensorHasStrideBreak(const aclTensor* t, int64_t dimNum)
{
    op::Shape shape = t->GetViewShape();
    op::Strides strides = t->GetViewStrides();
    for (int64_t j = dimNum - DIM_TWO; j >= 0; j--) {
        if (j == 0 && shape.GetDim(0) == 1) {
            continue;
        }
        if (strides[j] != strides[j + 1] * shape.GetDim(j + 1)) {
            return true;
        }
    }
    return false;
}

static bool IsNonConCases(op::FVector<const aclTensor*> tensors, int64_t dimNum, int64_t breakAxis)
{
    // 每个tensor都要校验，不满足尾轴大包；尾轴小包但stride大包；尾轴小包总体数据小包的场景，不支持非连续
    op::DataType shape0Dtype = tensors[0]->GetDataType();
    auto shape0DtypeSize = ge::GetSizeByDataType(shape0Dtype);
    auto coreNum = GetCurrentPlatformInfo().GetVectorCoreNum();
    for (uint64_t i = 0; i < tensors.size(); i++) {
        // 连续成员(无 stride 断点)在直通路径上是整块搬运, 不受小包约束:
        // 豁免, 避免混合输入中"块大行小"的连续成员误杀整表直通
        if (!TensorHasStrideBreak(tensors[i], dimNum)) {
            continue;
        }
        op::Shape shapeI = tensors[i]->GetViewShape();
        auto shapeIStride = tensors[i]->GetViewStrides();
        uint64_t lastAllData = 1;
        for (int64_t j = breakAxis + 1; j < dimNum; j++) {
            lastAllData *= shapeI.GetDim(j);
        }
        if (!(lastAllData * shape0DtypeSize >= SMALL_BAG) && !(shapeIStride[breakAxis] * shape0DtypeSize > SMALL_BAG) &&
            !(shapeI.GetShapeSize() * shape0DtypeSize < coreNum * SINGLE_CORE_PROCESS_SIZE)) {
            return false;
        }
    }
    return true;
}

// 断点轴识别放宽为任意非尾轴(b < size-1): dtype 一致、单一断点轴且各 tensor 同位、尾轴 stride 为 1、
// 断点轴 stride < 2^32, 且满足小包条件; 断点轴位置由 breakAxis 传出。
// 底层 ConcatD 已支持任意非尾断点轴: b == dim-1 走存量模板, 其余位置走 gather(b >= dim)/
// rowconcat(b < dim) 泛化分支(复用 PureCopy 模板), tiling 拒绝时由调用方回退常规 Contiguous 路径
static bool IsNonContiguousSupport(op::FVector<const aclTensor*> tensors, int64_t realDim, int64_t& breakAxis)
{
    if (!CheckSocAndNonConBasic(tensors)) {
        return false;
    }
    op::Shape shape0 = tensors[0]->GetViewShape();
    auto dimNum = static_cast<int64_t>(shape0.GetDimNum());
    if (dimNum < 1 || realDim < 0 || realDim >= dimNum) {
        return false;
    }
    op::DataType shape0Dtype = tensors[0]->GetDataType();
    for (uint64_t i = 0; i < tensors.size(); i++) {
        if (tensors[i]->GetDataType() != shape0Dtype) {
            return false;
        }
    }
    if (!FindBreakAxis(tensors, dimNum, realDim, breakAxis)) {
        return false;
    }
    // 断点位置范围(需求 2.3.5.3): 仅支持 b == realDim-1(存量 legacy)或 realDim == 0(泛化族, 断点任意非尾轴),
    // 其余位置回退常规 Contiguous 路径
    if (breakAxis != realDim - 1 && realDim != 0) {
        return false;
    }
    // 底层泛化分支准入预检(与 tiling 对齐, 避免依赖 tiling 拒绝回退):
    // FP4 不支持; 断点轴 stride 不得小于断点后连续块(重叠视图, 如 expand)
    if (shape0Dtype == DataType::DT_FLOAT4_E1M2 || shape0Dtype == DataType::DT_FLOAT4_E2M1) {
        return false;
    }
    // dim=0 泛化族底层走 gather 分支: 单个连续段(断点轴后各维乘积)必须装得进本设备 gather
    // UB 预算, 口径与底层 tiling 完全一致: tiling 在 DoTiling 中先按 ENABLE_DB 将 ubSize
    // 减半(双缓冲), 再 (ubSize/2-1024)/dtypeSize/2(BUFFER_NUM), 且受 u16 索引上限 65535
    // 钳制。段长仅由 shape 决定(与 tensor 数无关), 随设备 UB 大小不同表现不同;
    // 超预算时 tiling 会拒绝且 l0op::ConcatD 不回传 nullptr、带病进执行图导致整个调用
    // 失败(ubFactorDim0=0 / INNER_NULLPTR), 无法事后回退, 必须在此前置拦截,
    // 回退常规 Contiguous 路径
    if (realDim == 0) {
        int64_t segElems = 1;
        for (int64_t j = breakAxis + 1; j < dimNum; j++) {
            segElems *= shape0.GetDim(j);
        }
        uint64_t ubSize = 0;
        auto* platformInfos = GetCurrentPlatformInfo().GetPlatformInfos();
        if (platformInfos != nullptr) {
            platformInfos->GetLocalMemSize(fe::LocalMemType::UB, ubSize);
        }
        constexpr int64_t DB_BUFFER_SPLIT = 2;         // 与 concat tiling 侧 ENABLE_DB(HALF) 一致
        constexpr int64_t INDEX_USE_UB_RESERVE = 1024; // 与 concat tiling 侧 INDEX_USE_UB 一致
        constexpr int64_t GATHER_BUFFER_NUM = 2;       // 与 concat tiling 侧 BUFFER_NUM 一致
        constexpr int64_t U16_INDEX_LIMIT = 65535;
        auto shape0DtypeSize = ge::GetSizeByDataType(shape0Dtype);
        int64_t ubAfterDb = static_cast<int64_t>(ubSize) / DB_BUFFER_SPLIT;
        int64_t gatherUbBudget = ubAfterDb <= INDEX_USE_UB_RESERVE ?
                                     0 :
                                     (ubAfterDb - INDEX_USE_UB_RESERVE) / shape0DtypeSize / GATHER_BUFFER_NUM;
        if (gatherUbBudget > U16_INDEX_LIMIT) {
            gatherUbBudget = U16_INDEX_LIMIT;
        }
        if (ubSize == 0 || segElems > gatherUbBudget) {
            return false;
        }
    }
    for (uint64_t i = 0; i < tensors.size(); i++) {
        op::Shape shapeI = tensors[i]->GetViewShape();
        op::Strides stridesI = tensors[i]->GetViewStrides();
        // 断点轴 stride 上限校验(int64 域比较, 先 cast uint32 会截断高位):
        // stride ≥ 2^32 的视图会通过门控后在 compact tiling 的 uint32 strideListCompact
        // 中被静默截断, 导致错误源地址(老版本存在此检查, 门控重写时被移除, 此处恢复)
        if (stridesI[breakAxis] >= static_cast<int64_t>(MAX_UINT32_NUM)) {
            return false;
        }
        int64_t innerSize = 1;
        for (int64_t j = breakAxis + 1; j < dimNum; j++) {
            innerSize *= shapeI.GetDim(j);
        }
        if (stridesI[breakAxis] < 0 || stridesI[breakAxis] < innerSize) {
            return false;
        }
    }
    if (!IsNonConCases(tensors, dimNum, breakAxis)) {
        return false;
    }
    return true;
}

// 非连续直通尝试: 失败返回 false, 由调用方回退常规路径
static bool TryProcessNonContiguous(op::FVector<const aclTensor*> tensorList, int64_t dim, aclTensor* out,
                                    aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> tensorListA;
    for (uint64_t i = 0; i < tensorList.size(); i++) {
        auto viewTensor = executor->CreateView(tensorList[i], tensorList[i]->GetViewShape(),
                                               tensorList[i]->GetStorageShape(), tensorList[i]->GetViewStrides(),
                                               tensorList[i]->GetViewOffset());
        if (viewTensor == nullptr) {
            OP_LOGW("aclnnCat create non contiguous view failed, fallback to contiguous path.");
            return false;
        }
        tensorListA.emplace_back(viewTensor);
    }

    while (tensorListA.size() > 1) {
        op::FVector<const aclTensor*> tensorListOnce;
        op::FVector<const aclTensor*> tensorListB;
        for (auto tensor : tensorListA) {
            tensorListOnce.emplace_back(tensor);
            if (tensorListOnce.size() == MAX_TENSOR_NUM) {
                auto tensorAllocList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensor = l0op::ConcatD(tensorAllocList, dim, executor);
                if (concatTensor == nullptr) {
                    OP_LOGW("aclnnCat non contiguous ConcatD rejected, fallback to contiguous path.");
                    return false;
                }
                tensorListB.emplace_back(concatTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            if (tensorListOnce.size() == 1) {
                tensorListB.emplace_back(tensorListOnce.front());
            } else {
                auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensorTail = l0op::ConcatD(aclTensorListTail, dim, executor);
                if (concatTensorTail == nullptr) {
                    OP_LOGW("aclnnCat non contiguous ConcatD rejected, fallback to contiguous path.");
                    return false;
                }
                tensorListB.emplace_back(concatTensorTail);
            }
            tensorListOnce.clear();
        }
        tensorListA = tensorListB;
    }

    if (tensorListA.empty()) {
        return true;
    }
    if (!CheckShapeAndScalarSame(tensorListA.front(), out)) {
        OP_LOGW("aclnnCat non contiguous result shape mismatch, fallback to contiguous path.");
        return false;
    }
    auto castOut = l0op::Cast(tensorListA.front(), out->GetDataType(), executor);
    if (castOut == nullptr) {
        OP_LOGW("aclnnCat non contiguous cast failed, fallback to contiguous path.");
        return false;
    }
    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    if (viewCopyResult == nullptr) {
        OP_LOGW("aclnnCat non contiguous view copy failed, fallback to contiguous path.");
        return false;
    }
    return true;
}

static aclnnStatus SplitToConcat(const aclTensorList* tensors, int64_t dim, aclTensor* out, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> tensorListA;
    auto promoteType = (*tensors)[0]->GetDataType();
    if (!(*tensors)[0]->IsEmpty()) {
        tensorListA.emplace_back((*tensors)[0]);
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        if (!(*tensors)[i]->IsEmpty()) {
            tensorListA.emplace_back((*tensors)[i]);
        }
    }

    // 非连续输入: 满足门控时直通 ConcatD(断点=dim-1 走存量模板, 其余非尾断点位置走
    // gather/rowconcat 泛化分支), 被拒时自动回退常规 Contiguous 路径。
    // 注: 混合输入(连续+非连续)同样由此门控放行——FindBreakAxis 对连续成员跳过投票;
    // 此处不再做"连续成员 Contiguous 归一化后二次重试"——归一化不改变门控可见的
    // shape/stride, 二次判定必然与首次一致(实测不可达), 已删除。
    int64_t breakAxis = -1;
    if (IsNonContiguousSupport(tensorListA, dim, breakAxis)) {
        if (TryProcessNonContiguous(tensorListA, dim, out, executor)) {
            return ACLNN_SUCCESS;
        }
    }

    if (tensorListA.size() == 1) {
        // FP4 类型: TensorMove 的 AICORE 不支持 FP4，大数据量回退 AICPU 也不支持，导致报错。
        // 改为走 ConcatD 路径（ConcatD 支持单 tensor 和空 tensor）
        bool isFP4 = (promoteType == DataType::DT_FLOAT4_E1M2 || promoteType == DataType::DT_FLOAT4_E2M1);
        if (isFP4) {
            auto ct = l0op::Contiguous(tensorListA[0], executor);
            CHECK_RET(ct != nullptr, ACLNN_ERR_INNER_NULLPTR);
            op::FVector<const aclTensor*> fp4List;
            fp4List.emplace_back(ct);
            auto fp4TensorList = executor->AllocTensorList(fp4List.data(), fp4List.size());
            auto concatResult = l0op::ConcatD(fp4TensorList, dim, executor);
            CHECK_RET(concatResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto castOut = l0op::Cast(concatResult, out->GetDataType(), executor);
            auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            return ACLNN_SUCCESS;
        } else {
            return ProcessOneTensor(tensorListA[0], out, executor);
        }
    }

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    size_t catMaxInputs = (IsRegBase(npuArch)) ? CAT_INPUT_NUM_REGBASE_512 : CAT_INPUT_NUM_32;
    auto tensorListV2 = executor->AllocTensorList(tensorListA.data(), tensorListA.size());
    if (l0op::IsSupportConcatDV2(tensorListV2, dim)) {
        catMaxInputs = CAT_INPUT_NUM_V2_512;
    }
    bool firstLoop = true;
    while (tensorListA.size() > 1) {
        op::FVector<const aclTensor*> tensorListOnce;
        op::FVector<const aclTensor*> tensorListB;
        for (auto tensor : tensorListA) {
            if (firstLoop) {
                auto contiguous = l0op::Contiguous(tensor, executor);
                CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto castOut = l0op::Cast(contiguous, promoteType, executor);
                if (castOut == nullptr) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Result type %s can't be cast to the desired output type %s.",
                            op::ToString(contiguous->GetDataType()).GetString(), op::ToString(promoteType).GetString());
                    return ACLNN_ERR_INNER_NULLPTR;
                }
                tensorListOnce.emplace_back(castOut);
            } else {
                tensorListOnce.emplace_back(tensor);
            }
            if (tensorListOnce.size() == catMaxInputs) {
                auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensor = l0op::ConcatD(tensorList, dim, executor);
                CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListB.emplace_back(concatTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            if (tensorListOnce.size() == 1) {
                tensorListB.emplace_back(tensorListOnce.front());
            } else {
                auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensorTail = l0op::ConcatD(aclTensorListTail, dim, executor);
                CHECK_RET(concatTensorTail != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListB.emplace_back(concatTensorTail);
            }
            tensorListOnce.clear();
        }
        tensorListA = tensorListB;
        firstLoop = false;
    }

    if (tensorListA.empty()) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckShapeAndScalarSame(tensorListA.front(), out), ACLNN_ERR_PARAM_INVALID);
    auto castOut = l0op::Cast(tensorListA.front(), out->GetDataType(), executor);
    if (castOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Result type %s can't be cast to the desired output type %s.",
                op::ToString(tensorListA.front()->GetDataType()).GetString(),
                op::ToString(out->GetDataType()).GetString());
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclTensorList* EraseDim1EmptyTensor(const aclTensorList* tensors, aclOpExecutor* executor)
{
    if (tensors == nullptr) {
        return nullptr;
    }
    op::FVector<const aclTensor*> fTensorList;
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        op::Shape shape = (*tensors)[i]->GetViewShape();
        if ((shape.GetDimNum() == 1) && (shape.GetDim(0) == 0)) {
            continue;
        }
        fTensorList.push_back((*tensors)[i]);
    }
    return executor->AllocTensorList(fTensorList.data(), fTensorList.size());
}

aclnnStatus aclnnCatGetWorkspaceSize(const aclTensorList* tensors, int64_t dim, aclTensor* out, uint64_t* workspaceSize,
                                     aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCat, DFX_IN(tensors, dim), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclTensorList* tensorList = EraseDim1EmptyTensor(tensors, uniqueExecutor.get());
    if (tensorList != nullptr && tensorList->Size() == 0) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    int64_t realDim = dim;
    auto ret = CheckParams(tensorList, &realDim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = SplitToConcat(tensorList, realDim, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCat);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
