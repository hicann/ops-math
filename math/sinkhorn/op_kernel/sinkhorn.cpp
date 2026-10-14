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
 * \file sinkhorn.cpp
 * \brief
 */
#include "sinkhorn.h"

using namespace AscendC;

// 输入 cost dtype 的 TilingKey 语义编码（与 Host 侧 tilingKey = dataType 保持一致）。
// opc 工具会按源码中的这些语义常量生成独立 kernel 实例（每个 key 编译一份二进制，
// 编译期注入 -DTILING_KEY_VAR=<key>），kernel 侧仅以命名常量参与分支判断，
// 不使用 TILING_KEY_IS 宏，也不直接书写裸的 TilingKey 数值。
#define SINKHORN_FLOAT_TILING_KEY 0
#define SINKHORN_FLOAT16_TILING_KEY 1
#define SINKHORN_BFLOAT16_TILING_KEY 27

template <typename IT>
__aicore__ inline void LaunchSinkhorn(GM_ADDR cost, GM_ADDR p, GM_ADDR usrWorkspace, SinkhornTilingData* tilingData)
{
    AscendC::KernelSinkhorn<float, IT> op;
    op.Init(cost, p, usrWorkspace, tilingData);
    op.Process();
}

extern "C" __global__ __aicore__ void sinkhorn(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // 这两句都不要删除，有些调用框架还不能适配，需要保留
    SetSysWorkspace(workspace);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针。

    if (TILING_KEY_VAR == (SINKHORN_FLOAT_TILING_KEY)) {
        LaunchSinkhorn<float>(cost, p, usrWorkspace, &tiling_data);
    } else if (TILING_KEY_VAR == (SINKHORN_FLOAT16_TILING_KEY)) {
        LaunchSinkhorn<half>(cost, p, usrWorkspace, &tiling_data);
    } else if (TILING_KEY_VAR == (SINKHORN_BFLOAT16_TILING_KEY)) {
        LaunchSinkhorn<bfloat16_t>(cost, p, usrWorkspace, &tiling_data);
    }
}
