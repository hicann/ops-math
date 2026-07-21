/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace aicpu;

class TEST_TRANS_DATA_UT : public testing::Test {};

namespace {
std::shared_ptr<NodeDef> BuildTransDataNode(DataType dtype, Format src_fmt, Format dst_fmt,
                                            const std::vector<int64_t>& src_shape,
                                            const std::vector<int64_t>& dst_shape, void* src_data, void* dst_data,
                                            const std::string& src_format_str, const std::string& dst_format_str,
                                            int64_t groups = 1)
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "TransData", "TransData")
        .Input({"src", dtype, src_shape, src_data, src_fmt})
        .Output({"dst", dtype, dst_shape, dst_data, dst_fmt})
        .Attr("src_format", src_format_str)
        .Attr("dst_format", dst_format_str)
        .Attr("groups", groups);
    return node_def;
}
} // namespace

TEST_F(TEST_TRANS_DATA_UT, unsupported_output_format_returns_param_invalid)
{
    std::vector<int64_t> src_shape = {1, 2, 2, 4};
    std::vector<int64_t> dst_shape = {1, 2, 2, 4};
    std::vector<float> src(16, 1.0f);
    std::vector<float> dst(16, 0.0f);
    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NHWC, FORMAT_NHWC, src_shape, dst_shape, src.data(), dst.data(),
                                       "NHWC", "NHWC");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TRANS_DATA_UT, insufficient_dims_returns_param_invalid)
{
    std::vector<int64_t> src_shape = {2, 4};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(8, 1.0f);
    std::vector<float> dst(256, 0.0f);
    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TRANS_DATA_UT, groups_zero_returns_param_invalid)
{
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(16, 1.0f);
    std::vector<float> dst(256, 0.0f);
    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z", 0);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
