/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "math/confusion_matrix/op_host/arch35/confusion_matrix_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "any_value.h"

using namespace std;

class ConfusionMatrixTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConfusionMatrixTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConfusionMatrixTilingTest TearDown" << std::endl; }
};

static gert::TilingContextPara::OpAttr MakeNumClassesAttr(int64_t numClasses)
{
    return gert::TilingContextPara::OpAttr("num_classes", Ops::Math::AnyValue::CreateFrom<int64_t>(numClasses));
}

static gert::TilingContextPara::OpAttr MakeDtypeAttr(const std::string& dtype)
{
    return gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<std::string>(dtype));
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_float32_with_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(5),
                                                  MakeDtypeAttr("float32"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 65536;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_float32_no_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(5),
                                                  MakeDtypeAttr("float32"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_int32_with_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 10}, {10, 10}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(10),
                                                  MakeDtypeAttr("int32"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 65792;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_int32_no_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 10}, {10, 10}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(10),
                                                  MakeDtypeAttr("int32"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_float16_with_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 5}, {5, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(5),
                                                  MakeDtypeAttr("float16"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 66048;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_int8_no_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 10}, {10, 10}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(10),
                                                  MakeDtypeAttr("int8"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 771;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_uint8_with_weights)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{100}, {100}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                  {{{100}, {100}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{10, 10}, {10, 10}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(10),
                                                  MakeDtypeAttr("uint8"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 66563;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionMatrixTilingTest, confusion_matrix_tiling_large_num_classes)
{
    optiling::AscendCConfusionMatrixCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ConfusionMatrix",
                                              {
                                                  {{{10000}, {10000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{10000}, {10000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{10000}, {10000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1000, 1000}, {1000, 1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  MakeNumClassesAttr(1000),
                                                  MakeDtypeAttr("float32"),
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 65538;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
