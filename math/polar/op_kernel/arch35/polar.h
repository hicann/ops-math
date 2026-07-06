/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file polar_simd.h
 * \brief polar operator SIMD kernel entry for arch35 (ascend950)
 */
#ifndef POLAR_SIMD_H_
#define POLAR_SIMD_H_

#include "kernel_operator.h"
#include "atvoss/broadcast/broadcast_sch.h"
#include "polar_dag.h"
#include "polar_struct.h"

using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void polar(GM_ADDR abs, GM_ADDR angle, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    using OpDag = PolarOp::PolarBrcDag<complex64, float>::OpDag;
    BroadcastSch<schMode, OpDag> sch(tiling);
    sch.Process(abs, angle, out);
}

#endif // POLAR_SIMD_H_