/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_multinomial.h"
#include "multinomial_with_replacement.h"
#include "math/reduce_sum_op/op_host/op_api/reduce_sum_op.h"
#include "math/abs/op_api/abs.h"
#include "math/add/op_api/add.h"
#include "math/sub/op_api/sub.h"
#include "math/real_div/op_host/op_api/realdiv.h"
#include "math/log/op_api/log.h"
#include "math/arg_max_v2/op_host/op_api/argmax_v2.h"
#include "math/topk/op_host/op_api/topk.h"
#include "math/cumsum/op_host/op_api/cumsum.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "dsa_random_uniform.h"
#include "random/stateless_random_uniform_v2/op_api/stateless_random_uniform_v2.h"
#include "conversion/concat/op_api/concat.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "math/greater_equal/op_api/greater_equal.h"
#include "conversion/view_copy/op_host/op_api/view_copy.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << 24;
constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t DIM_NUM_TWO = 2;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t CPU_NPU_BOUNDARY = 5000;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static inline bool CheckSocVersionGe910B(void) {
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline bool CheckSocVersionIsSupportDSA(void) {
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList() {
  if (CheckSocVersionGe910B()) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static inline int64_t MakeWrapDim(int64_t dim, int64_t dimPostExpr) {
  // support dim=0
  if (dimPostExpr <= 0) {
    dimPostExpr = 1;
  }
  if (dim < 0) {
    dim += dimPostExpr;
  }
  return dim;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, {op::DataType::DT_INT64}, return false);
  return true;
}

static bool CheckShape(const aclTensor *self, int64_t numsamples, const aclTensor *out) {
  if (self->GetViewShape().GetDimNum() != DIM_NUM_ONE && self->GetViewShape().GetDimNum() != DIM_NUM_TWO) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of self only can be 1 or 2.");
    return false;
  }
  if (self->GetViewShape().GetDimNum() != out->GetViewShape().GetDimNum()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of self should be equal to dim of out.");
    return false;
  }
  auto dimNum = out->GetViewShape().GetDimNum();
  auto nCategories = out->GetViewShape().GetDim(dimNum - 1);
  if (nCategories != numsamples) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "excepted the size of out at last dim must be equal with numsamples %ld, but got %ld.", numsamples, nCategories);
    return false;
  }
  if (self->GetViewShape().GetDimNum() != DIM_NUM_ONE && self->GetViewShape().GetDim(DIM_ZERO) != out->GetViewShape().GetDim(DIM_ZERO)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "excepted the size of out at first dim %ld must be equal with the size of self at first dim %ld when dimNum > 1", 
            self->GetViewShape().GetDim(DIM_ZERO), out->GetViewShape().GetDim(DIM_ZERO));
    return false;
  }
  return true;
}

static bool CheckValueRange(const aclTensor *self, int64_t numsamples, bool replacement) {
  if (numsamples <= 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Numsamples must > 0.");
    return false;
  }
  auto dimNum = self->GetViewShape().GetDimNum();
  auto nCategories = self->GetViewShape().GetDim(dimNum - 1);
  if (!replacement && (numsamples > nCategories)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numsamples must <= shape.GetDim(dimNum - 1) without replacement");
    return false;
  }
  if (nCategories > FLOAT32_MAX_CONSECUTIVE_INT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "number of categories cannot exceed 2^24");
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t numsamples, bool replacement, const aclTensor *out) {
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(self, numsamples, out), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckValueRange(self, numsamples, replacement), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

const aclTensor *GetRandomUniformReplaceMent(const aclTensor *selfContiguous, int64_t numsamples,
                                  const int64_t *randomParams, aclOpExecutor *executor)
{
  const int64_t randAShape[] = {numsamples};
  auto randAShapeArray = executor->AllocIntArray(randAShape, 1);
  const aclTensor* randomUniform = nullptr;
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
    auto low = executor->AllocScalar(0.0f);
    auto high = executor->AllocScalar(1.0f);
    randomUniform = l0op::DSARandomUniform(randAShapeArray, *randomParams, *(randomParams + 1), low, high, executor);
  } else {
    int32_t alg = 1;
    op::Shape shape;
    op::ToShape(randAShapeArray->GetData(), randAShapeArray->Size(), shape);
    auto randAShapeTensor = executor->AllocTensor(shape, selfContiguous->GetDataType(), selfContiguous->GetViewFormat());
    CHECK_RET(randAShapeTensor != nullptr, nullptr);
    randomUniform = l0op::StatelessRandomUniformV2(randAShapeTensor, *randomParams, *(randomParams + 1), alg, executor);
  }
  CHECK_RET(randomUniform != nullptr, nullptr);
  return randomUniform;
}

const aclTensor *GetRandomUniformReplaceMentTensor(int64_t numsamples,
                                  const aclTensor *seed, const aclTensor *offset, aclOpExecutor *executor)
{
  // concat
  FVector<uint64_t> zeroVector = {static_cast<uint64_t>(0)};
  auto zeroTensor = executor->ConvertToTensor(zeroVector.data(), zeroVector.size(), offset->GetDataType());
  FVector<const aclTensor*> tensorListOnce;
  tensorListOnce.emplace_back(zeroTensor);
  tensorListOnce.emplace_back(offset);
  auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
  auto concatTensor = l0op::ConcatD(tensorList, 0, executor);

  // RandomUniform, shape = {1, ..., 1, numsamples}
  const int64_t randAShape[] = {numsamples};
  auto randAShapeArray = executor->AllocIntArray(randAShape, 1);
  auto low = executor->AllocScalar(0.0f);
  auto high = executor->AllocScalar(1.0f);
  auto randomUniform = l0op::DSARandomUniformTensor(randAShapeArray, seed, concatTensor, low, high, executor);
  CHECK_RET(randomUniform != nullptr, nullptr);

  return randomUniform;
}

const aclTensor *RunMultinomialReplaceMent(const aclTensor *selfContiguous, int64_t numsamples,
                                            const aclTensor *randomUniform, const aclTensor *out, aclOpExecutor *executor) {
  // trun weight into probability
  int64_t dimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
  int64_t lastDim = MakeWrapDim(-1, dimNum);
  const int64_t dim[] = {lastDim};
  auto dimArray = executor->AllocIntArray(dim, 1);
  auto sumSelf = l0op::ReduceSumOp(selfContiguous, dimArray, true, executor);
  CHECK_RET(sumSelf != nullptr, nullptr);

  auto divSumSelf = l0op::RealDiv(selfContiguous, sumSelf, executor);
  CHECK_RET(divSumSelf != nullptr, nullptr);

  // trun probability into point on [0, 1], shape = original_shape + {1}
  const aclTensor *dimTensor = nullptr;
  if (lastDim == 0 || lastDim > INT32_MAX) {
    dimTensor = executor->ConvertToTensor(&lastDim, 1, DataType::DT_INT64);
  } else {
    dimTensor = executor->ConvertToTensor(&lastDim, 1, DataType::DT_INT32);
  }
  CHECK_RET(dimTensor != nullptr, nullptr);
  auto accumSelf = l0op::Cumsum(divSumSelf, dimTensor, executor);
  CHECK_RET(accumSelf != nullptr, nullptr);

  auto unsqueezeAccum = l0op::UnsqueezeNd(accumSelf, -1, executor);
  CHECK_RET(unsqueezeAccum != nullptr, nullptr);

  auto castRandom = l0op::Cast(randomUniform, selfContiguous->GetDataType(), executor);
  CHECK_RET(castRandom != nullptr, nullptr);

  int64_t newShape[dimNum + 1];
  newShape[0] = numsamples;
  for (int64_t i = 0; i < dimNum; i++) {
    newShape[i] = 1;
  }
  newShape[dimNum] = numsamples;
  auto newShapeArray = executor->AllocIntArray(newShape, dimNum + 1);
  auto reshapeRandom = l0op::Reshape(castRandom, newShapeArray, executor);
  CHECK_RET(reshapeRandom != nullptr, nullptr);

  // caculate the point RandomUniform on interval unsqueezeAccum
  auto greaterEqual = l0op::GreaterEqual(reshapeRandom, unsqueezeAccum, executor);
  CHECK_RET(greaterEqual != nullptr, nullptr);

  auto castGreaterEqual = l0op::Cast(greaterEqual, out->GetDataType(), executor);
  CHECK_RET(castGreaterEqual != nullptr, nullptr);

  auto multinomialOut = l0op::ReduceSumOp(castGreaterEqual, dimArray, false, executor);
  CHECK_RET(multinomialOut != nullptr, nullptr);
  return multinomialOut;
}

const aclTensor *GetRandomUniformNoReplaceMent(const aclTensor *selfContiguous, const int64_t *randomParams,
                                               aclOpExecutor *executor)
{
  // exponentialOne = torch.exponential_(1)
  const aclTensor* randomUniform = nullptr;
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
    auto inputShape = op::ToShapeVector(selfContiguous->GetViewShape());
    auto inputShapeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    CHECK_RET(inputShapeArray != nullptr, nullptr);
    auto low = executor->AllocScalar(0.0f);
    auto high = executor->AllocScalar(1.0f);
    randomUniform = l0op::DSARandomUniform(inputShapeArray, *randomParams, *(randomParams + 1), low, high, executor);
  } else {
    int32_t alg = 1;
    auto statelessUniform = l0op::StatelessRandomUniformV2(selfContiguous, *randomParams, *(randomParams + 1), alg, executor);
    CHECK_RET(statelessUniform != nullptr, nullptr);
    randomUniform = l0op::Cast(statelessUniform, DataType::DT_FLOAT, executor);
  }
  CHECK_RET(randomUniform != nullptr, nullptr);

  return randomUniform;
}

const aclTensor *GetRandomUniformNoReplaceMentTensor(const aclTensor *selfContiguous, const aclTensor *seed,
                                               const aclTensor *offset, aclOpExecutor *executor)
{
  // concat
  FVector<uint64_t> zeroVector = {static_cast<uint64_t>(0)};
  auto zeroTensor = executor->ConvertToTensor(zeroVector.data(), zeroVector.size(), offset->GetDataType());
  FVector<const aclTensor*> tensorListOnce;
  tensorListOnce.emplace_back(zeroTensor);
  tensorListOnce.emplace_back(offset);
  auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
  auto concatTensor = l0op::ConcatD(tensorList, 0, executor);

  // exponentialOne = torch.exponential_(1)
  auto inputShape = op::ToShapeVector(selfContiguous->GetViewShape());
  auto inputShapeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
  CHECK_RET(inputShapeArray != nullptr, nullptr);
  auto low = executor->AllocScalar(0.0f);
  auto high = executor->AllocScalar(1.0f);
  auto randomUniform = l0op::DSARandomUniformTensor(inputShapeArray, seed, concatTensor, low, high, executor);
  CHECK_RET(randomUniform != nullptr, nullptr);

  return randomUniform;
}

const aclTensor *RunMultinomialNoReplaceMent(const aclTensor *selfContiguous, int64_t numsamples,
                                              const aclTensor *randomUniform, const aclTensor *out, aclOpExecutor *executor) {
  const float one = 1.0;
  const aclTensor *oneTensor = executor->ConvertToTensor(&one, 1, DataType::DT_FLOAT);
  auto subRandom = l0op::Sub(oneTensor, randomUniform, executor);
  CHECK_RET(subRandom != nullptr, nullptr);

  const float logBase = -1.0f;
  const float logScale = 1.0f;
  const float logShift = 0.0f;
  auto logRandom = l0op::Log(subRandom, logBase, logScale, logShift, executor);
  CHECK_RET(logRandom != nullptr, nullptr);

  auto absLog = l0op::Abs(logRandom, executor);
  CHECK_RET(absLog != nullptr, nullptr);

  // add positiveMin to prevent exponential = 0
  const float positiveMin = 1e-7;
  const aclTensor *positiveMinTensor = executor->ConvertToTensor(&positiveMin, 1, DataType::DT_FLOAT);
  auto exponential = l0op::Add(absLog, positiveMinTensor, executor);
  CHECK_RET(exponential != nullptr, nullptr);

  auto castExponential = l0op::Cast(exponential, selfContiguous->GetDataType(), executor);
  CHECK_RET(castExponential != nullptr, nullptr);

  // give selfContiguous random disturbance
  auto divExponential = l0op::RealDiv(selfContiguous, castExponential, executor);
  CHECK_RET(divExponential != nullptr, nullptr);

  int64_t dimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
  int64_t lastDim = MakeWrapDim(-1, dimNum);
  const aclTensor *multinomialOutInt32;
  if (numsamples == 1) {
    multinomialOutInt32 = l0op::ArgMaxV2(divExponential, lastDim, true, executor);
  } else {
    auto topkOut = l0op::Topk(divExponential, numsamples, lastDim, true, true, op::DataType::DT_INT32, executor);
    multinomialOutInt32 = std::get<1>(topkOut);
  }
  CHECK_RET(multinomialOutInt32 != nullptr, nullptr);

  auto multinomialOut = l0op::Cast(multinomialOutInt32, out->GetDataType(), executor);
  CHECK_RET(multinomialOut != nullptr, nullptr);
  return multinomialOut;
}

aclnnStatus aclnnMultinomialGetWorkspaceSize(const aclTensor *self, int64_t numsamples, bool replacement,
                                             int64_t seed, int64_t offset, aclTensor *out,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnMultinomial, DFX_IN(self, numsamples, replacement, seed, offset), DFX_OUT(out));

  auto ret = CheckParams(self, numsamples, replacement, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  const int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  int64_t selfSize = 1;
  for (int64_t i = 0; i < dimNum; i++) {
    selfSize *= self->GetViewShape().GetDim(i);
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  const aclTensor *multinomialOut;
  int64_t randomParams[2] = {seed, offset};
  if (!CheckSocVersionGe910B() || selfSize <= CPU_NPU_BOUNDARY) {
    multinomialOut = l0op::MultinomialWithReplacement(selfContiguous,
                                                      numsamples,
                                                      replacement,
                                                      seed,
                                                      offset,
                                                      uniqueExecutor.get());
  } else if (!replacement || numsamples == 1) {
    auto randomUniform = GetRandomUniformNoReplaceMent(selfContiguous, randomParams, uniqueExecutor.get());
    CHECK_RET(randomUniform != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    multinomialOut=RunMultinomialNoReplaceMent(selfContiguous,
                                                numsamples,
                                                randomUniform,
                                                out,
                                                uniqueExecutor.get());
  } else {
    // RandomUniform, shape = {1, ..., 1, numsamples}
    auto randomUniform = GetRandomUniformReplaceMent(selfContiguous, numsamples, randomParams, uniqueExecutor.get());
    CHECK_RET(randomUniform != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    multinomialOut=RunMultinomialReplaceMent(selfContiguous,
                                              numsamples,
                                              randomUniform,
                                              out,
                                              uniqueExecutor.get());
  }
  CHECK_RET(multinomialOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(multinomialOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultinomial(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMultinomial);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static const aclTensor *AddOffsetTensor(const aclTensor *offsetTensor, int64_t offset, aclOpExecutor *executor) {
    FVector<int64_t> tmpVector = {static_cast<int64_t>(offset)};
    auto offsetTmpTensor = executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), offsetTensor->GetDataType());
    CHECK_RET(offsetTmpTensor != nullptr, nullptr);
    auto offsetAddOut = l0op::Add(offsetTensor, offsetTmpTensor, executor);

    return offsetAddOut;
}

aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor *self, int64_t numsamples, bool replacement,
                                             const aclTensor *seedTensor, const aclTensor *offsetTensor, int64_t offset, aclTensor *out,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnMultinomialTensor, DFX_IN(self, numsamples, replacement, seedTensor, offsetTensor, offset), DFX_OUT(out));

  CHECK_RET(CheckSocVersionIsSupportDSA(), ACLNN_ERR_PARAM_INVALID);
  auto ret = CheckParams(self, numsamples, replacement, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  const int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  int64_t selfSize = 1;
  for (int64_t i = 0; i < dimNum; i++) {
    selfSize *= self->GetViewShape().GetDim(i);
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  auto offsetAddOut = AddOffsetTensor(offsetTensor, offset, uniqueExecutor.get());
  CHECK_RET(offsetAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor *multinomialOut;
  if (!CheckSocVersionGe910B() || selfSize <= CPU_NPU_BOUNDARY) {
    multinomialOut = l0op::MultinomialWithReplacementTensor(selfContiguous,
                                                      numsamples,
                                                      replacement,
                                                      seedTensor,
                                                      offsetAddOut,
                                                      uniqueExecutor.get());
  } else if (!replacement || numsamples == 1) {
    auto randomUniform = GetRandomUniformNoReplaceMentTensor(selfContiguous, seedTensor, offsetAddOut, uniqueExecutor.get());
    CHECK_RET(randomUniform != nullptr, ACLNN_ERR_INNER_NULLPTR);
    multinomialOut=RunMultinomialNoReplaceMent(selfContiguous,
                                                numsamples,
                                                randomUniform,
                                                out,
                                                uniqueExecutor.get());
  } else {
    // RandomUniform, shape = {1, ..., 1, numsamples}
    auto randomUniform = GetRandomUniformReplaceMentTensor(numsamples, seedTensor, offsetAddOut, uniqueExecutor.get());
    CHECK_RET(randomUniform != nullptr, ACLNN_ERR_INNER_NULLPTR);
    multinomialOut=RunMultinomialReplaceMent(selfContiguous,
                                              numsamples,
                                              randomUniform,
                                              out,
                                              uniqueExecutor.get());
  }
  CHECK_RET(multinomialOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(multinomialOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultinomialTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMultinomialTensor);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
