/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <register/op_tiling.h>
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "random_ops.h"
#include "test_common.h"
#include "common_unittest.h"
#include "../../../op_host/lin_space_tiling.h"
#include "ut_op_util.h"
#include "array_ops.h"
#include "../../../fusion_pass/common/fp16_t.hpp"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;

class LinSpaceTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LinSpaceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LinSpaceTiling TearDown" << std::endl;
    }
};

TEST_F(LinSpaceTiling, lin_space_tiling_001)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {10000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT32, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT32, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_INT32, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000800 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 10000 0 256 0 32 0 320 0 80 0 0 0 64 0 0 0 0 0 1 0 320 0 1 0 80 0 601 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_002)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<float> start_const_value = {0};
    vector<float> stop_const_value = {8};
    vector<int64_t> num_const_value = {10000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_FLOAT, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_FLOAT, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT64, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {10000}, DT_FLOAT, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000800 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 10000 0 256 0 32 0 320 0 80 0 0 0 64 0 0 0 0 0 1 0 320 0 1 0 80 0 101 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_003)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int8_t> start_const_value = {0};
    vector<int8_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {101};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT8, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT8, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {101}, DT_INT8, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.080000 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 101 0 32 0 4 0 32 0 5 0 0 0 0 0 0 0 0 0 1 0 32 0 1 0 5 0 301 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_004)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<uint8_t> start_const_value = {0};
    vector<uint8_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {101};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_UINT8, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_UINT8, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {101}, DT_UINT8, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.080000 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 101 0 32 0 4 0 32 0 5 0 0 0 0 0 0 0 0 0 1 0 32 0 1 0 5 0 401 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_005)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int16_t> start_const_value = {0};
    vector<int16_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {101};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {101}, DT_INT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.080000 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 101 0 16 0 7 0 16 0 5 0 0 0 0 0 0 0 0 0 1 0 16 0 1 0 5 0 501 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_006)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<fe::fp16_t> start_const_value;
    vector<fe::fp16_t> stop_const_value;
    vector<int32_t> num_const_value = {101};
    fe::fp16_t start_value(0);
    fe::fp16_t stop_value(8.0);
    start_const_value.push_back(start_value);
    stop_const_value.push_back(stop_value);

    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_FLOAT16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_FLOAT16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {101}, DT_FLOAT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.080000 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 101 0 16 0 7 0 16 0 5 0 0 0 0 0 0 0 0 0 1 0 16 0 1 0 5 0 201 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_007)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};

    vector<int16_t> start_const_value = {0x3f80};  // 1.0, bf16:
    vector<int16_t> stop_const_value = {0x4120};   // 10.0, bf16:
    vector<int32_t> num_const_value = {101};

    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_BF16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_BF16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {101}, DT_BF16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "1.000000 10.000000 0.090000 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 101 0 16 0 7 0 16 0 5 0 0 0 0 0 0 0 0 0 1 0 16 0 1 0 5 0 701 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_201)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {640000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT32, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT32, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_INT32, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000013 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 5 0 3616 0 5 0 3616 0 602 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_202)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<float> start_const_value = {0};
    vector<float> stop_const_value = {8};
    vector<int64_t> num_const_value = {640000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_FLOAT, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_FLOAT, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT64, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_FLOAT, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000013 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 5 0 3616 0 5 0 3616 0 102 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_203)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int8_t> start_const_value = {0};
    vector<int8_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {640000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT8, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT8, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_INT8, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000013 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 2 0 3616 0 2 0 3616 0 302 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_204)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<uint8_t> start_const_value = {0};
    vector<uint8_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {640000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_UINT8, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_UINT8, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_UINT8, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000013 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 2 0 3616 0 2 0 3616 0 402 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_205)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int16_t> start_const_value = {0};
    vector<int16_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {640000};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_INT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000013 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 3 0 3616 0 3 0 3616 0 502 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_206)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<fe::fp16_t> start_const_value;
    vector<fe::fp16_t> stop_const_value;
    vector<int32_t> num_const_value = {640000};
    fe::fp16_t start_value(0);
    fe::fp16_t stop_value(8.0);
    start_const_value.push_back(start_value);
    stop_const_value.push_back(stop_value);

    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_FLOAT16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_FLOAT16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_FLOAT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "0.000000 8.000000 0.000013 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 3 0 3616 0 3 0 3616 0 202 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_207)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};

    vector<int16_t> start_const_value = {0x3f80};  // 1.0, bf16:
    vector<int16_t> stop_const_value = {0x4120};   // 10.0, bf16:
    vector<int32_t> num_const_value = {640000};

    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_BF16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_BF16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {640000}, DT_BF16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);
    // EXPECT_EQ(parse_ret, ge::GRAPH_SUCCESS);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);

    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<float>(raw_tiling_data->GetData(), 3 * 4), "1.000000 10.000000 0.000014 ");
    EXPECT_EQ(to_string<int32_t>(((int32_t*)raw_tiling_data->GetData() + 3), 4 * 28),
              "0 640000 0 256 0 32 0 20000 0 20000 0 39 0 3616 0 39 0 3616 0 3 0 3616 0 3 0 3616 0 702 ");
}

TEST_F(LinSpaceTiling, lin_space_tiling_208)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT32, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT32, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_INT32, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

TEST_F(LinSpaceTiling, lin_space_tiling_209)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_FLOAT, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_FLOAT, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_FLOAT, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

TEST_F(LinSpaceTiling, lin_space_tiling_210)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_FLOAT16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_FLOAT16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_FLOAT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

TEST_F(LinSpaceTiling, lin_space_tiling_211)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_INT8, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_INT8, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_INT8, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

TEST_F(LinSpaceTiling, lin_space_tiling_212)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_UINT8, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_UINT8, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_UINT8, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

TEST_F(LinSpaceTiling, lin_space_tiling_213)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_UINT16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_UINT16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_UINT16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
}

TEST_F(LinSpaceTiling, lin_space_tiling_214)
{
    auto opParas = op::LinSpace("LinSpace");

    vector<int64_t> start_shape = {1};
    vector<int64_t> stop_shape = {1};
    vector<int64_t> num_shape = {1};
    vector<int32_t> start_const_value = {0};
    vector<int32_t> stop_const_value = {8};
    vector<int32_t> num_const_value = {1};
    using namespace ut_util;
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, start, start_shape, DT_BF16, FORMAT_ND, start_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, stop, stop_shape, DT_BF16, FORMAT_ND, stop_const_value);
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num, num_shape, DT_INT32, FORMAT_ND, num_const_value);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, output, {8}, DT_BF16, FORMAT_ND, {});

    Runtime2TestParam rtparam;
    rtparam.input_const = {true, true, true};
    std::unique_ptr<uint8_t[]> tilingdata;
    optiling::LinSpaceCompileInfo compileInfo;
    auto parse_ret = TilingParseTest("LinSpace", "", &compileInfo);

    auto tiling_ret = TilingTest(opParas, &compileInfo, tilingdata, rtparam);
    EXPECT_EQ(tiling_ret, ge::GRAPH_SUCCESS);
}

template <typename T>
void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
                   std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({{data_size}, {data_size}},          // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                        gert::kFollowing,                    // placement
                        dtype,                               // dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

TEST_F(LinSpaceTiling, linspace_tiling_ascendc_0001)
{
    gert::StorageShape start_shape = {{1}, {1}};
    gert::StorageShape stop_shape = {{1}, {1}};
    gert::StorageShape num_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{5}, {5}};

    int32_t start_value[1] = {0};
    int32_t stop_value[1] = {8};
    int32_t num_value[1] = {5};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, ge::DT_INT32, start_value, 1, const_tensors);
    SetConstInput(1, ge::DT_INT32, stop_value, 1, const_tensors);
    SetConstInput(2, ge::DT_INT32, num_value, 1, const_tensors);

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::LinSpaceCompileInfo compile_info;

    std::string op_type("LinSpace");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .SetOpType("LinSpace")
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&start_shape, &stop_shape, &num_shape})
                      .OutputShapes({&out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(const_tensors)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(raw_tiling_data->GetDataSize(), 168);
    string tilingRes = to_string<int64_t>(raw_tiling_data->GetData(), 152);
    tilingRes += to_string<float>((static_cast<float*>(raw_tiling_data->GetData()) + 38), 12);
    EXPECT_EQ(tilingRes, "40 196608 5 1 16256 5 1 5 5 5 1 5 5 0 2 1 2 1 3 0.000000 8.000000 2.000000 ");
}

static void ExecuteTestCase(ge::DataType start_dtype, ge::DataType stop_dtype, ge::DataType num_dtype,
                            ge::DataType out_dtype, gert::StorageShape input_shape, gert::StorageShape out_shape,
                            std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors,
                            string compareStr = "")
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::LinSpaceCompileInfo compile_info;

    std::string op_type("LinSpace");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .SetOpType("LinSpace")
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &input_shape})
                      .OutputShapes({&out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, start_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, stop_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, num_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, out_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(const_tensors)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(raw_tiling_data->GetDataSize(), 168);
    string tilingRes = to_string<int64_t>(raw_tiling_data->GetData(), 152);
    tilingRes += to_string<float>((static_cast<float*>(raw_tiling_data->GetData()) + 38), 12);
    EXPECT_EQ(tilingRes, compareStr);
}

TEST_F(LinSpaceTiling, range_tiling_ascendc_0002)
{
    gert::StorageShape input_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{1031}, {1031}};

    float start_value[1] = {0};
    float stop_value[1] = {10000};
    int64_t num_value[1] = {1031};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, ge::DT_FLOAT, start_value, 1, const_tensors);
    SetConstInput(1, ge::DT_FLOAT, stop_value, 1, const_tensors);
    SetConstInput(2, ge::DT_INT64, num_value, 1, const_tensors);

    string compareStr = "40 196608 1031 9 16256 128 1 128 128 7 1 7 7 4 3 1 3 1 125 0.000000 10000.000000 9.708738 ";
    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape, out_shape, const_tensors, compareStr);
}

TEST_F(LinSpaceTiling, range_tiling_ascendc_0003)
{
    gert::StorageShape input_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{125}, {125}};

    fe::fp16_t start_value[1] = {0};
    float stop_value[1] = {1000};
    int64_t num_value[1] = {125};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, ge::DT_FLOAT16, start_value, 1, const_tensors);
    SetConstInput(1, ge::DT_FLOAT, stop_value, 1, const_tensors);
    SetConstInput(2, ge::DT_INT64, num_value, 1, const_tensors);

    string compareStr = "40 196608 125 1 16256 125 1 125 125 125 1 125 125 0 62 1 62 1 63 0.000000 1000.000000 8.064516 ";
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT32, input_shape, out_shape, const_tensors, compareStr);
}

TEST_F(LinSpaceTiling, range_tiling_ascendc_0004)
{
    gert::StorageShape input_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{0}, {0}};

    fe::fp16_t start_value[1] = {0};
    float stop_value[1] = {1000};
    int64_t num_value[1] = {0};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, ge::DT_FLOAT16, start_value, 1, const_tensors);
    SetConstInput(1, ge::DT_FLOAT, stop_value, 1, const_tensors);
    SetConstInput(2, ge::DT_INT64, num_value, 1, const_tensors);

    string compareStr = "40 196608 0 1 16256 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0.000000 1000.000000 0.000000 ";
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT32, input_shape, out_shape, const_tensors, compareStr);
}

TEST_F(LinSpaceTiling, range_tiling_ascendc_0005)
{
    gert::StorageShape input_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{1}, {1}};

    fe::fp16_t start_value[1] = {0};
    float stop_value[1] = {1000};
    int64_t num_value[1] = {1};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, ge::DT_FLOAT16, start_value, 1, const_tensors);
    SetConstInput(1, ge::DT_FLOAT, stop_value, 1, const_tensors);
    SetConstInput(2, ge::DT_INT64, num_value, 1, const_tensors);

    string compareStr = "40 196608 1 1 32512 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0.000000 0.000000 0.000000 ";
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT8, input_shape, out_shape, const_tensors, compareStr);
}