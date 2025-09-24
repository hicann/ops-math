/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file abs_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_ABS_DAG_H
#define CANN_CUSTOM_OPS_ABS_DAG_H
#include "op_kernel/atvoss/atp/dag.h"
#include "op_kernel/atvoss/atp/vec.h"
#include "op_kernel/atvoss/atp/placeholder.h"

using namespace AscendC;

template <typename U, typename T = float>
struct AbsDag {
    using OpCopyIn0 = AscendC::Bind<AscendC::Vec::CopyIn<U>, AscendC::Placeholder::In0<U>>;
    using OpCopyIn0Cast = AscendC::Bind<AscendC::Vec::Cast<T, U, 0>, OpCopyIn0>;
    using OpResult = AscendC::Bind<AscendC::Vec::Abs<T>, OpCopyIn0Cast>;
    using OpResultCast = AscendC::Bind<AscendC::Vec::Cast<U, T, 1>, OpResult>;
    using OpCopyOut = AscendC::Bind<AscendC::Vec::CopyOut<U>, AscendC::Placeholder::Out0<U>, OpResultCast>;

    using Outputs = AscendC::Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = AscendC::DAGSch<Outputs, void, MemCfg>;
};

#endif // CANN_CUSTOM_OPS_ABS_DAG_H