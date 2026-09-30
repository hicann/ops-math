/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_stack.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_check.h"
#include "../../concat_d/op_api/concat_d.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "pack.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,      op::DataType::DT_INT16,     op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,     op::DataType::DT_UINT16,    op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,     op::DataType::DT_BOOL,   op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,      op::DataType::DT_INT16,      op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,     op::DataType::DT_UINT16,     op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,      op::DataType::DT_BOOL,   op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,        op::DataType::DT_INT16,        op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,       op::DataType::DT_UINT16,       op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16,     op::DataType::DT_FLOAT,        op::DataType::DT_BOOL,   op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64,   op::DataType::DT_COMPLEX128,   op::DataType::DT_BF16,   op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};
constexpr uint32_t MAX_UINT32_NUM = 4294967295;
constexpr uint32_t SMALL_BAG = 128;
constexpr uint32_t SINGLE_CORE_PROCESS_SIZE = 8192;
// 单次 ConcatD 非连续分支支持的输入个数上限, 与内核侧 concat tiling 的约束一致
constexpr uint32_t MAX_TENSOR_NUM = 64;
constexpr int32_t DIM_TWO = 2;

static bool CheckNotNull(const aclTensorList* tensors, const int64_t* realDim, const aclTensor* out)
{
    if (tensors == nullptr || realDim == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Input of aclnnStack should not be null.");
        return false;
    }
    OP_CHECK_NULL(out, return false);
    return true;
}
static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase(curArch)) {
        return REGBASE_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!CheckType((*tensors)[i]->GetDataType(), supportList)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s, should be in dtype support list [%s].",
                    i, op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(supportList).GetString());
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    return true;
}

static bool CheckPromoteType(const aclTensorList* tensors, const aclTensor* out)
{
    op::DataType promoteType = (*tensors)[0]->GetDataType();
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        if (promoteType == DataType::DT_UNDEFINED) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor %lu dtype %s and dtype %s can not promote dtype.", i,
                    op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(promoteType).GetString());
            return false;
        }
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensorList* tensors, int64_t* realDim)
{
    op::Shape shapeFirst = (*tensors)[0]->GetViewShape();
    auto dimNum = (int64_t)shapeFirst.GetDimNum();
    static const int64_t MAX_DIM_LEN = 8;
    if (dimNum > MAX_DIM_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of self is %ld, can't be greater than %ld.", dimNum, MAX_DIM_LEN);
        return false;
    }
    auto originDim = *realDim;
    if (*realDim < 0) {
        (*realDim) += dimNum + 1;
    }
    if ((*realDim) < 0 || (*realDim) > dimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimnum %ld exceeds the dim range [%ld, %ld].", originDim, -(dimNum + 1),
                dimNum);
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        op::Shape shapeCurrent = (*tensors)[i]->GetViewShape();
        if (dimNum != (int64_t)shapeCurrent.GetDimNum()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimnum of tensor %lu is [%zu], should be equal to tensor 0 [%ld].", i,
                    shapeCurrent.GetDimNum(), dimNum);
            return false;
        }
        for (int64_t j = 0; j < dimNum; j++) {
            if (shapeFirst.GetDim(j) != shapeCurrent.GetDim(j)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim %ld of tensor %lu is [%ld], should be equal to tensor 0 [%ld].",
                        j, i, shapeCurrent.GetDim(j), shapeFirst.GetDim(j));
                return false;
            }
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, int64_t* realDim, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(tensors, realDim, out), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckPromoteType(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(tensors, realDim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool EmplaceTensorList(const aclTensorList* tensors, op::FVector<const aclTensor*>& tensorListA,
                              DataType& promoteType)
{
    tensorListA.emplace_back((*tensors)[0]);
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        tensorListA.emplace_back((*tensors)[i]);
    }
    return true;
}

// stack 语义等价于在 dim 处插一个 size=1 的轴后沿 dim 做 ConcatD, 此处构造插轴后的 view shape
static op::Shape MakeStackViewShape(const op::Shape& shape, int64_t dim)
{
    auto dimNum = static_cast<int64_t>(shape.GetDimNum());
    op::Shape viewShape;
    viewShape.SetDimNum(dimNum + 1);
    for (int64_t i = 0, j = 0; i <= dimNum; i++) {
        if (i == dim) {
            viewShape.SetDim(i, 1);
        } else {
            viewShape.SetDim(i, shape.GetDim(j));
            j++;
        }
    }
    return viewShape;
}

// 插入轴(size=1)的stride不影响寻址, 取后一轴 stride*shape 以满足"除 strideDim 外均连续"的校验; 末尾插轴取 1
static op::Strides MakeStackViewStrides(const op::Shape& shape, const op::Strides& strides, int64_t dim)
{
    auto dimNum = static_cast<int64_t>(shape.GetDimNum());
    op::Strides viewStrides(dimNum + 1);
    for (int64_t i = 0, j = 0; i <= dimNum; i++) {
        if (i == dim) {
            viewStrides[i] = (dim == dimNum) ? 1 : strides[dim] * shape.GetDim(dim);
        } else {
            viewStrides[i] = strides[j];
            j++;
        }
    }
    return viewStrides;
}

// 与 aclnnCat 的 CheckSocAndNonConBasic 一致的基础门控: RegBase、tensor 数 > 1
// (个数无上限, 超过单批上限时由 ProcessNonContiguous 分批)
static bool CheckSocAndNonConBasic(const op::FVector<const aclTensor*>& tensors)
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

// 在原输入坐标系识别唯一的非尾断点轴: 尾轴 stride 须为 1, 除断点轴外其余轴须满足
// stride[j] = stride[j+1] * shape[j+1] 的连续关系, 且所有 tensor 的断点轴须同位

static bool FindBreakAxis(const op::FVector<const aclTensor*>& tensors, int64_t realDim, int64_t& breakAxis)
{
    op::Shape shape0 = tensors[0]->GetViewShape();
    auto dimNum = static_cast<int64_t>(shape0.GetDimNum());
    if (dimNum < 1) {
        return false;
    }
    int64_t bk0 = -1;
    for (uint64_t i = 0; i < tensors.size(); i++) {
        op::Shape shapeI = tensors[i]->GetViewShape();
        if (static_cast<int64_t>(shapeI.GetDimNum()) != dimNum) {
            return false;
        }
        op::Strides stridesI = tensors[i]->GetViewStrides();
        if (stridesI[dimNum - 1] != 1) {
            return false;
        }
        int64_t bkI = -1;
        for (int64_t j = dimNum - DIM_TWO; j >= 0; j--) {
            // 仅 dim==0(gather 域)时对最外层 size==1 轴豁免断点投票(与 aclnnCat 对齐);
            // dim!=0(legacy 域)时不豁免——保持断点投票走 legacy 轻模板(与基线行为一致)
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

// 与 aclnnCat 的 IsNonContiguousSupport/IsNonConCases 条件对齐, 断点轴放宽为任意非尾轴(b < size-1):
// dtype 一致、单一断点轴且各 tensor 同位、尾轴 stride 为 1、断点轴 stride < 2^32,
// 且每个 tensor 至少满足尾轴大包/stride大包/总量小包其一; breakAxis 传出断点轴位置
static bool IsNonContiguousSupport(const op::FVector<const aclTensor*>& tensors, int64_t realDim, int64_t& breakAxis)
{
    if (!CheckSocAndNonConBasic(tensors)) {
        return false;
    }
    // stack 支持的 DT_COMPLEX128 不在底层 ConcatD AICore tiling 支持列表内, 回退常规路径
    op::DataType shape0Dtype = tensors[0]->GetDataType();
    if (shape0Dtype == DataType::DT_COMPLEX128) {
        return false;
    }
    if (realDim < 0) {
        return false;
    }
    if (!FindBreakAxis(tensors, realDim, breakAxis)) {
        return false;
    }
    // 插轴位 = 用户 dim, 几何准入与 aclnnCat 非连续门控(需求 2.3.5.3)完全对齐, 仅放行两类:
    // 1) breakAxis == dim-1: 插轴后断点恰为 concat 轴前一轴, 走存量 legacy 模板;
    // 2) dim == 0: 插轴后断点轴 b+1 > 0, 走 ConcatD dim=0 gather 泛化分支(输出布局即用户布局)。
    // 两类输出布局均 == 用户期望布局, 无需仿射重排。
    // 其余几何(b>=dim 且 dim>0 / b<dim-1)测试覆盖不足(dim>=4、rank5 等场景无用例背书),
    // 门控回退常规 Contiguous 路径
    if (breakAxis != realDim - 1 && realDim != 0) {
        return false;
    }
    auto dimNum = static_cast<int64_t>(tensors[0]->GetViewShape().GetDimNum());
    auto shape0DtypeSize = ge::GetSizeByDataType(shape0Dtype);
    auto coreNum = GetCurrentPlatformInfo().GetVectorCoreNum();
    // dim=0 族走 ConcatD gather 泛化分支: 单个连续段(断点轴后各维乘积)必须装得进本设备
    // gather UB 预算, 口径与底层 tiling 完全一致: tiling 在 DoTiling 中先按 ENABLE_DB 将
    // ubSize 减半(双缓冲), 再 (ubSize/2-1024)/dtypeSize/2(BUFFER_NUM), 且受 u16 索引上限
    // 65535 钳制。段长仅由 shape 决定(与 tensor 数无关), 随设备 UB 大小不同表现不同;
    // 超预算时 tiling 会拒绝且 l0op::ConcatD 不回传 nullptr、带病进执行图导致整个调用
    // 失败(ubFactorDim0=0 / INNER_NULLPTR), 无法事后回退, 必须在此前置拦截,
    // 回退常规 Contiguous+Pack 路径
    if (realDim == 0) {
        int64_t segElems = 1;
        for (int64_t j = breakAxis + 1; j < dimNum; j++) {
            segElems *= tensors[0]->GetViewShape().GetDim(j);
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
        if (tensors[i]->GetDataType() != shape0Dtype) {
            return false;
        }
        op::Shape shapeI = tensors[i]->GetViewShape();
        op::Strides stridesI = tensors[i]->GetViewStrides();
        // 断点轴 stride 上限校验须在 int64 域比较: 先 cast uint32 会截断高位,
        // stride ≥ 2^32 的视图(如 f32 ≥16GB 存储)会通过门控后在 compact tiling 的
        // uint32 strideListCompact 中被静默截断, 导致错误源地址
        if (stridesI[breakAxis] >= static_cast<int64_t>(MAX_UINT32_NUM)) {
            return false;
        }
        // 连续成员(无 stride 断点)在直通路径上是整块搬运, 不受小包约束:
        // 豁免, 避免混合输入中"块大行小"的连续成员误杀整表直通(与 aclnnCat 对齐)
        bool hasBreak = false;
        for (int64_t j = dimNum - DIM_TWO; j >= 0; j--) {
            if (j == 0 && shapeI.GetDim(0) == 1) {
                continue;
            }
            if (stridesI[j] != stridesI[j + 1] * shapeI.GetDim(j + 1)) {
                hasBreak = true;
                break;
            }
        }
        if (!hasBreak) {
            continue;
        }
        // 每个tensor都要校验，不满足尾轴大包；尾轴小包但stride大包；尾轴小包总体数据小包的场景，不支持非连续
        uint64_t lastAllData = 1;
        for (int64_t j = breakAxis + 1; j < dimNum; j++) {
            lastAllData *= shapeI.GetDim(j);
        }
        if (!(lastAllData * shape0DtypeSize >= SMALL_BAG) && !(stridesI[breakAxis] * shape0DtypeSize > SMALL_BAG) &&
            !(shapeI.GetShapeSize() * shape0DtypeSize < coreNum * SINGLE_CORE_PROCESS_SIZE)) {
            return false;
        }
    }
    return true;
}

// 构造插轴 view 保留 stride 信息, 插轴放在用户 dim 位(p = dim), 调用前门控已保证仅两类几何:
// - dim == 0: 插轴后断点轴 b+1 > 0, 直通 ConcatD dim=0 gather 泛化分支(参考 aclnnCat dim=0
//   非连续实现), concat 输出物理布局 == 用户期望布局(N 轴恰在用户 dim 位), 无需仿射重排;
// - dim == b+1: 断点恰为 dim-1, 走存量 legacy 非连续模板(与插轴 p=breakAxis+1 等价);
// 按 MAX_TENSOR_NUM 分批; 返回 false 表示底层 ConcatD 拒绝, 由调用方回退常规路径
static bool ProcessNonContiguous(const op::FVector<const aclTensor*>& tensors, int64_t dim, int64_t breakAxis,
                                 aclOpExecutor* executor, const aclTensor** out)
{
    (void)breakAxis; // 插轴位仅由用户 dim 决定, 断点位置已由门控保证可直通
    int64_t p = dim;
    op::FVector<const aclTensor*> tensorListA;
    for (uint64_t i = 0; i < tensors.size(); i++) {
        op::Shape shapeI = tensors[i]->GetViewShape();
        op::Shape viewShape = MakeStackViewShape(shapeI, p);
        op::Strides viewStrides = MakeStackViewStrides(shapeI, tensors[i]->GetViewStrides(), p);
        auto viewTensor = executor->CreateView(tensors[i], viewShape, tensors[i]->GetStorageShape(), viewStrides,
                                               tensors[i]->GetViewOffset());
        if (viewTensor == nullptr) {
            OP_LOGW("aclnnStack create non contiguous view failed, fallback to contiguous path.");
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
                auto concatTensor = l0op::ConcatD(tensorAllocList, p, executor);
                if (concatTensor == nullptr) {
                    OP_LOGW("aclnnStack non contiguous ConcatD rejected, fallback to contiguous path.");
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
                auto concatTensorTail = l0op::ConcatD(aclTensorListTail, p, executor);
                if (concatTensorTail == nullptr) {
                    OP_LOGW("aclnnStack non contiguous ConcatD rejected, fallback to contiguous path.");
                    return false;
                }
                tensorListB.emplace_back(concatTensorTail);
            }
            tensorListOnce.clear();
        }
        tensorListA = tensorListB;
    }
    if (tensorListA.empty()) {
        *out = nullptr;
        return true;
    }

    // 插轴位 = 用户 dim: concat 输出物理形状(连续) = 原shape 在 dim 位插 N, 与用户期望
    // 布局完全一致, 直接作为结果返回, 外层 Cast/ViewCopy 同 dtype 时为直连, 无重排开销
    *out = tensorListA.front();
    return true;
}

static aclnnStatus SplitToStack(const aclTensorList* tensors, int64_t dim, aclOpExecutor* executor,
                                const aclTensor** out)
{
    size_t maxInputs = 32;
    op::FVector<const aclTensor*> tensorListA;
    auto promoteType = (*tensors)[0]->GetDataType();
    bool firstLoop = EmplaceTensorList(tensors, tensorListA, promoteType);

    // 非连续输入: 满足条件时构造插轴view(插轴置于断点轴后一位)直通ConcatD, 免去Contiguous整理;
    // 底层ConcatD拒绝时回退常规路径
    op::FVector<const aclTensor*> tensorListNonEmpty;
    for (auto tensor : tensorListA) {
        if (!tensor->IsEmpty()) {
            tensorListNonEmpty.emplace_back(tensor);
        }
    }
    int64_t breakAxis = -1;
    if (IsNonContiguousSupport(tensorListNonEmpty, dim, breakAxis)) {
        if (ProcessNonContiguous(tensorListNonEmpty, dim, breakAxis, executor, out)) {
            return ACLNN_SUCCESS;
        }
    }
    // 注: 混合输入(连续+非连续)同样由上述门控放行——FindBreakAxis 对连续成员跳过投票;
    // 不再做"连续成员 Contiguous 归一化后二次重试"——归一化不改变门控可见的
    // shape/stride, 二次判定必然与首次一致(实测不可达), 已删除。

    while (tensorListA.size() > 1) {
        op::FVector<const aclTensor*> tensorListOnce;
        op::FVector<const aclTensor*> tensorLlistB;
        for (auto tensor : tensorListA) {
            if (tensor->IsEmpty()) {
                continue;
            }
            if (firstLoop) {
                auto contiguous = l0op::Contiguous(tensor, executor);
                CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto castOut = l0op::Cast(contiguous, promoteType, executor);
                CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListOnce.emplace_back(castOut);
            } else {
                tensorListOnce.emplace_back(tensor);
            }
            if (tensorListOnce.size() == maxInputs) {
                auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                const aclTensor* stackTensor = nullptr;
                if (firstLoop) {
                    stackTensor = l0op::Pack(tensorList, dim, promoteType, executor);
                } else {
                    stackTensor = l0op::ConcatD(tensorList, dim, promoteType, executor);
                }
                CHECK_RET(stackTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorLlistB.emplace_back(stackTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
            const aclTensor* stackTensorTail = nullptr;
            if (firstLoop) {
                stackTensorTail = l0op::Pack(aclTensorListTail, dim, promoteType, executor);
            } else {
                stackTensorTail = l0op::ConcatD(aclTensorListTail, dim, promoteType, executor);
            }
            CHECK_RET(stackTensorTail != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tensorLlistB.emplace_back(stackTensorTail);
            tensorListOnce.clear();
        }
        tensorListA = tensorLlistB;
        firstLoop = false;
    }
    *out = tensorListA.empty() ? nullptr : tensorListA.front();
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStackGetWorkspaceSize(const aclTensorList* tensors, int64_t dim, aclTensor* out,
                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnStack, DFX_IN(tensors, dim), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    int64_t realDim = dim;
    auto ret = CheckParams(tensors, &realDim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* castIn = nullptr;
    if (tensors->Size() == 1) {
        if ((*tensors)[0]->IsEmpty()) {
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
        auto contiguous = l0op::Contiguous((*tensors)[0], uniqueExecutor.get());
        CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        op::FVector<const aclTensor*> tensorList{contiguous};
        auto oneTensor = uniqueExecutor.get()->AllocTensorList(tensorList.data(), tensorList.size());
        castIn = l0op::Pack(oneTensor, realDim, (*tensors)[0]->GetDataType(), uniqueExecutor.get());
    } else {
        aclnnStatus retSplit = SplitToStack(tensors, realDim, uniqueExecutor.get(), &castIn);
        CHECK_RET(retSplit == ACLNN_SUCCESS, retSplit);
    }
    if (castIn == nullptr) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckShapeAndScalarSame(castIn, out), ACLNN_ERR_PARAM_INVALID);

    auto castOut = l0op::Cast(castIn, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStack(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStack);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
