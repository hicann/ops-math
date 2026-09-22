/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferDataType4ArgMaxV2(gert::InferDataTypeContext* context)
{
    OP_LOGD("Begin InferDataType4ArgMaxV2");
    ge::DataType outputDtype = ge::DT_INT32;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto attr_dtype_ptr = attrs->GetAttrPointer<int64_t>(0);

    if (attr_dtype_ptr != nullptr) {
        outputDtype = static_cast<ge::DataType>(*attr_dtype_ptr);
    } else {
        OP_LOGW(context->GetNodeName(), "get attr dtype failed.");
    }

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OP_LOGW(context->GetNodeName(), "Get platform_info unsuccessful.");
    } else {
        int64_t ubBlockSize = platform_info.ai_core_spec.ubblock_size;
        OP_LOGD(context->GetNodeName(), "UB block size is %ld.", ubBlockSize);
        static const int64_t nano_block_size = 16;
        if (ubBlockSize == nano_block_size) {
            outputDtype = ge::DT_INT16;
        }
    }
    context->SetOutputDataType(0, outputDtype);
    OP_LOGD("End InferDataType4ArgMaxV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ArgMaxV2).InferDataType(InferDataType4ArgMaxV2);
} // namespace ops
