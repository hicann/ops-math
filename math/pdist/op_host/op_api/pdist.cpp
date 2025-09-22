/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pdist.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Pdist);
const aclTensor *Pdist(const aclTensor *self, float p,
                       aclOpExecutor *executor) {
  L0_DFX(Pdist, self, p);
  auto pdistOut = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
  CHECK_RET(pdistOut != nullptr, nullptr);
  INFER_SHAPE(Pdist, OP_INPUT(self), OP_OUTPUT(pdistOut), OP_ATTR(p));
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Pdist,
                                         OP_INPUT(self),
                                         OP_ATTR(p),
                                         OP_OUTPUT(pdistOut));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PdistAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return pdistOut;
}
}