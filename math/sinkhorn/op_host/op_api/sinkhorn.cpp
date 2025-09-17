/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sinkhorn.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sinkhorn);

const aclTensor* Sinkhorn(const aclTensor* cost, const aclScalar* tol, aclTensor* p, aclOpExecutor* executor)
{
    L0_DFX(Sinkhorn, cost, p);

    float fTol = 0.0001f;
    if (tol != nullptr) {
        fTol = tol->ToFloat();
    }

    ADD_TO_LAUNCHER_LIST_AICORE(Sinkhorn, OP_INPUT(cost), OP_OUTPUT(p), OP_ATTR(fTol));
    return p;
}

} // namespace l0op
