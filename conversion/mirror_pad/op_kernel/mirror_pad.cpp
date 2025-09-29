/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file mirror_pad.cpp
 * \brief mirror_pad
 */
#include "./arch35/mirror_pad_reflect.h"
#include "./arch35/mirror_pad_symmetric.h"

using namespace MirrorPad;

#define ONE_BYTE_TILING_KEY 1
#define TWO_BYTE_TILING_KEY 2
#define FOUR_BYTE_TILING_KEY 4
#define EIGHT_BYTE_TILING_KEY 8

extern "C" __global__ __aicore__ void mirror_pad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    // reflect mode
    if (tilingData.padMode == 1) {
        if (TILING_KEY_IS(ONE_BYTE_TILING_KEY)) {
          KernelMirrorPadReflect<int8_t> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      } else if (TILING_KEY_IS(TWO_BYTE_TILING_KEY)) {
          KernelMirrorPadReflect<half> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      } else if (TILING_KEY_IS(FOUR_BYTE_TILING_KEY)) {
          KernelMirrorPadReflect<float> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      } else if (TILING_KEY_IS(EIGHT_BYTE_TILING_KEY)) {
          KernelMirrorPadReflect<int64_t> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      }
    } else {
      if (TILING_KEY_IS(ONE_BYTE_TILING_KEY)) {
          KernelMirrorPadSymmetric<int8_t> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      } else if (TILING_KEY_IS(TWO_BYTE_TILING_KEY)) {
          KernelMirrorPadSymmetric<half> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      } else if (TILING_KEY_IS(FOUR_BYTE_TILING_KEY)) {
          KernelMirrorPadSymmetric<float> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      } else if (TILING_KEY_IS(EIGHT_BYTE_TILING_KEY)) {
          KernelMirrorPadSymmetric<int64_t> op;
          op.Init(x, paddings, y, tilingData);
          op.Process();
      }
    }
}