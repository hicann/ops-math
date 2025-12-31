/**
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 70701343... cl compile
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
<<<<<<< HEAD
=======
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
>>>>>>> 0fad06aa... ut
=======
>>>>>>> 70701343... cl compile
 */

/*!
 * \file col2im_tiling.h
 * \brief Col2Im tiling function declarations
 */

#ifndef COL2IM_TILING_H
#define COL2IM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

/**
 * @brief Col2Im算子的编译信息结构
 */
struct Col2ImCompileInfo {
    int32_t H;
    int32_t W;
    
    // 卷积参数
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_val;
    int32_t padding_val;
    int32_t dilation_val;
};

/**
 * @brief Col2Im算子的Tiling函数
 * @param context Tiling上下文
 * @return ge::GRAPH_SUCCESS 成功，ge::GRAPH_FAILED 失败
 */
ge::graphStatus Col2ImTilingFunc(gert::TilingContext* context);

/**
 * @brief Col2Im算子的TilingParse函数
 * @param context TilingParse上下文
 * @return ge::GRAPH_SUCCESS 成功，ge::GRAPH_FAILED 失败
 */
ge::graphStatus TilingParseForCol2Im(gert::TilingParseContext* context);

} // namespace optiling

#endif // COL2IM_TILING_H