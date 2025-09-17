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
 * \file test_strided_slice_assign_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "op_log.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "common_unittest.h"
#include "runtime/diag_util.h"
#include "conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "selection_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class StridedSliceAssignV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceAssignV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceAssignV2Tiling TearDown" << std::endl;
    }
};

TEST_F(StridedSliceAssignV2Tiling, strided_slice_assing_v2_tiling_002)
{
    auto opParas = op::StridedSliceAssignV2("StridedSliceAssignV2");

    vector<int64_t> var_shape = { 4, 6, 8 };
    vector<int64_t> input_shape = { 2, 2, 4 };
    vector<int64_t> idx_shape = { 3 };

    vector<int64_t> begin_const_value = { 1, 2, 3 };
    vector<int64_t> end_const_value = { 4, 6, 7 };
    vector<int64_t> strides_const_value = { 2, 3, 1 };
    vector<int64_t> axes_const_value = { 0, 1, 2 };
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE(opParas, var, var_shape, DT_FLOAT16, FORMAT_ND, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, input_value, input_shape, DT_FLOAT16, FORMAT_ND, {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, begin, idx_shape, DT_INT64, FORMAT_ND, begin_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, end, idx_shape, DT_INT64, FORMAT_ND, end_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, strides, idx_shape, DT_INT64, FORMAT_ND, strides_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axes, idx_shape, DT_INT64, FORMAT_ND, axes_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, var_shape, DT_FLOAT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = { false, false, true, true, true, true };
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::StridedSliceAssignV2CompileInfo compileInfo;
    auto parse_ret = TilingParseTest("StridedSliceAssignV2", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

TEST_F(StridedSliceAssignV2Tiling, strided_slice_assing_v2_tiling_003)
{
    auto opParas = op::StridedSliceAssignV2("StridedSliceAssignV2");

    vector<int64_t> var_shape = { 4, 6, 8 };
    vector<int64_t> input_shape = { 0, 2, 4 };
    vector<int64_t> idx_shape = { 3 };

    vector<int32_t> begin_const_value = { 5, 2, 3 };
    vector<int32_t> end_const_value = { -100, 6, 7 };
    vector<int32_t> strides_const_value = { 2, 3, 1 };
    vector<int32_t> axes_const_value = { 0, 1, 2 };
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE(opParas, var, var_shape, DT_FLOAT16, FORMAT_ND, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, input_value, input_shape, DT_FLOAT16, FORMAT_ND, {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, begin, idx_shape, DT_INT32, FORMAT_ND, begin_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, end, idx_shape, DT_INT32, FORMAT_ND, end_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, strides, idx_shape, DT_INT32, FORMAT_ND, strides_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axes, idx_shape, DT_INT32, FORMAT_ND, axes_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, var_shape, DT_FLOAT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = { false, false, true, true, true, true };
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::StridedSliceAssignV2CompileInfo compileInfo;
    auto parse_ret = TilingParseTest("StridedSliceAssignV2", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}