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
 * \file onnx_common.h
 * \brief
 */

#ifndef MATH_COMMON_ONNX_COMMON_H
#define MATH_COMMON_ONNX_COMMON_H

#include <string>
#include <vector>
#include <map>

#include "stub_ops.h"
#include "register/register.h"
#include "graph/operator.h"
#include "math_onnx_plugin_util.h"
#include "graph/graph.h"
#include "base/err_msg.h"
#include "log/log.h"
#include "onnx/proto/ge_onnx.pb.h"

// All protobuf-free helpers that used to live here (GetOpName, Vec2Tensor, CreateScalar,
// DataTypeOnnx, onnx2om_dtype_map, GetOmDtypeFromOnnxDtype, ChangeFormatFromOnnx) have
// been moved to math_onnx_plugin_util.h, which this header includes. This header now only
// exists to pull in ge_onnx.pb.h (and the surrounding headers) for plugins still coupled
// to the protobuf NodeProto interface.

#endif //  MATH_COMMON_ONNX_COMMON_H
