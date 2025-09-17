/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unfold_grad_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class UnfoldGrad : public OpDef {
public:
    explicit UnfoldGrad(const char *name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("input_sizes")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("grad_in")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("dim").Int();
        this->Attr("size").Int();
        this->Attr("step").Int();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(UnfoldGrad);
} // namespace ops
