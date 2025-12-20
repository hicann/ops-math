/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include <iostream>
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/tensor.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tensor_data.h"
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_common.h"
using namespace ut_util;

class CumsumUTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "cumsum SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "cumsum TearDown" << std::endl;
  }
};

TEST_F(CumsumUTest, CUMSUM_INFERSHAPE) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Cumsum"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Cumsum")->infer_shape;
  gert::StorageShape x_shape = {{3, 2}, {3, 2}};
  gert::StorageShape axis_shape = {{}, {}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .InputShapes({&x_shape, &axis_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 2);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
}

TEST_F(CumsumUTest, cum_sum_test_0) {
  auto op_info = op::Cumsum("Cumsum");
  vector<int64_t> x_shape = {-2};
  vector<int64_t> axis_shape = {1};
  vector<int32_t> axis_value = {1};

  TENSOR_INPUT_WITH_SHAPE(op_info, x, x_shape, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op_info, axis, axis_shape, ge::DT_INT32, ge::FORMAT_ND, axis_value);

  auto ret = op_info.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op_info.GetOutputDesc("y");
  std::vector<int64_t> expected_output_y_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
  }
