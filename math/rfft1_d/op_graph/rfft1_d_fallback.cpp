/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_graph/op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t rfftInputX = 0;
constexpr size_t rfftOutput = 0;

graphStatus Rfft1DExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto x_ge = host_api_ctx->GetInputTensor(rfftInputX);
    OP_CHECK_IF(x_ge == nullptr, OP_LOGE("aclnnfallback", "x_ge is null"), return GRAPH_FAILED);

    auto output_ge = host_api_ctx->GetOutputTensor(rfftOutput);
    OP_CHECK_IF(output_ge == nullptr, OP_LOGE("aclnnfallback", "output_ge is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const int64_t* n = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* dim = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* norm = attrs->GetAttrPointer<int64_t>(2);
    // execute opapi
    auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclRfft1D, x_ge, n, dim, norm, output_ge);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(Rfft1D).OpExecuteFunc(Rfft1DExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
