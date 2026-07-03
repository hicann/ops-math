/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch32 (non-RegBase: ascend310p / ascend910_93 / ascend910b) tiling UT. This file is compiled
// only for the arch32 builds (wired via TILING_DIR arch32 in the op_host tests CMakeLists), where
// IsRegbaseSocVersion() is false and the HistogramV2MembaseTiling template is selected. It covers
// the MemBase-specific tiling branches; MemBase GetShapeAttrsInfo does not validate inputs, so the
// input-validation failure cases live only in the arch35 (SIMT) UT.

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../../op_host/histogram_v2_tiling.h"

using namespace ge;
using namespace std;

namespace {
using AV = Ops::Math::AnyValue;
using TD = gert::TilingContextPara::TensorDescription;

class HistogramV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HistogramV2Tiling (arch32/MemBase) SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HistogramV2Tiling (arch32/MemBase) TearDown" << std::endl; }
};

static bool RunTiling(const gert::TilingContextPara& para, uint64_t& tilingKey)
{
    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    tilingKey = static_cast<uint64_t>(info.tilingKey);
    return ok;
}

static bool RunTiling(const gert::TilingContextPara& para)
{
    uint64_t key = 0;
    return RunTiling(para, key);
}

// Build a HistogramV2 tiling param with 3 inputs (x, min, max) + 1 output (y).
static gert::TilingContextPara MakePara(void* compileInfo, const std::vector<int64_t>& xDims, ge::DataType xDtype,
                                        const std::vector<int64_t>& mmDims, ge::DataType mmDtype,
                                        const std::vector<int64_t>& yDims, ge::DataType yDtype, int64_t bins)
{
    gert::StorageShape xShape;
    for (auto d : xDims) {
        xShape.MutableOriginShape().AppendDim(d);
        xShape.MutableStorageShape().AppendDim(d);
    }
    gert::StorageShape mmShape;
    for (auto d : mmDims) {
        mmShape.MutableOriginShape().AppendDim(d);
        mmShape.MutableStorageShape().AppendDim(d);
    }
    gert::StorageShape yShape;
    for (auto d : yDims) {
        yShape.MutableOriginShape().AppendDim(d);
        yShape.MutableStorageShape().AppendDim(d);
    }
    return gert::TilingContextPara("HistogramV2",
                                   {
                                       TD(xShape, xDtype, ge::FORMAT_ND),
                                       TD(mmShape, mmDtype, ge::FORMAT_ND),
                                       TD(mmShape, mmDtype, ge::FORMAT_ND),
                                   },
                                   {
                                       TD(yShape, yDtype, ge::FORMAT_ND),
                                   },
                                   {
                                       gert::TilingContextPara::OpAttr("bins", AV::CreateFrom<int64_t>(bins)),
                                   },
                                   compileInfo);
}

static optiling::HistogramV2CompileInfo MakeCompileInfo(int64_t coreNum = 64, NpuArch arch = NpuArch::DAV_2201)
{
    optiling::HistogramV2CompileInfo ci;
    ci.totalCoreNum = static_cast<int32_t>(coreNum);
    ci.ubSizePlatForm = 262144;
    ci.sysWorkspaceSize = 16 * 1024 * 1024;
    ci.npuArch = arch;
    return ci;
}

// ---------------------- MemBase valid cases ----------------------

TEST_F(HistogramV2Tiling, tiling_fp32_ub_full)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// dtype cases below exercise the MemBase SetTilingKeyMode dtype -> TilingKey dispatch
// (HISTOGRAM_V2_FP16 / INT32 / INT8 / UINT8 / INT16); int64 is covered separately below.
TEST_F(HistogramV2Tiling, tiling_fp16)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT16, {1}, ge::DT_FLOAT16, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2Tiling, tiling_int32)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT32, {1}, ge::DT_INT32, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2Tiling, tiling_int8)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT8, {1}, ge::DT_INT8, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2Tiling, tiling_uint8)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_UINT8, {1}, ge::DT_UINT8, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2Tiling, tiling_int16)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT16, {1}, ge::DT_INT16, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// int64 exercises the MemBase TilingDataInCore int64 branch (tileLength / 2).
TEST_F(HistogramV2Tiling, tiling_int64)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT64, {1}, ge::DT_INT64, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// npuArch DAV_2002 exercises the MemBase 310P branch (ubSelfLength_310P, userWorkspaceSize, ScheduleMode).
TEST_F(HistogramV2Tiling, tiling_membase_310p_branch)
{
    auto ci = MakeCompileInfo(64, NpuArch::DAV_2002);
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// totalLength < coreNum -> MemBase tailLength == 0 branch (coreNum reduced to 1).
TEST_F(HistogramV2Tiling, tiling_tail_length_zero)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {1}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

} // namespace
