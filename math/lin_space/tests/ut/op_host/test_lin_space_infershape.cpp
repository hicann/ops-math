/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include <iostream>
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/tensor.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tensor_data.h"

class linspace : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "linspace Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "linspace Proto Test TearDown" << std::endl;
  }

  template <typename T>
  gert::Tensor * ConstructInputConstTensor(std::unique_ptr<uint8_t[]>& input_tensor_holder,
                                           const T &const_value, ge::DataType const_dtype) {
    auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
    gert::Tensor tensor({{1}, {1}},                                 // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}},        // format
                        gert::kFollowing,                          // placement
                        const_dtype,                               //dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
    *tensor_data = const_value;
    std::cout<<" const_value:" << *tensor_data<< std::endl;
    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor;
  }
};

TEST_F(linspace, LinSpaceInferShapeCase0)
{
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace")->infer_shape;
  int32_t start = 1;
  int32_t stop = 10;
  int32_t num = 10;
  auto start_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t)]);
  auto start_tensor = ConstructInputConstTensor(start_input_tensor, start, ge::DT_INT32);
  auto stop_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t)]);
  auto stop_tensor = ConstructInputConstTensor(stop_input_tensor, stop, ge::DT_INT32);
  auto num_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t)]);
  auto num_tensor = ConstructInputConstTensor(num_input_tensor, num, ge::DT_INT32);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, stop_tensor, num_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 10);
}

TEST_F(linspace, LinSpaceInferShapeCase1)
{
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace")->infer_shape;
  float start = 1.0;
  float stop = 10.0;
  int64_t num = 100;
  auto start_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(float)]);
  auto start_tensor = ConstructInputConstTensor(start_input_tensor, start, ge::DT_FLOAT);
  auto stop_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(float)]);
  auto stop_tensor = ConstructInputConstTensor(stop_input_tensor, stop, ge::DT_FLOAT);
  auto num_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t)]);
  auto num_tensor = ConstructInputConstTensor(num_input_tensor, num, ge::DT_INT64);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, stop_tensor, num_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 100);
}

TEST_F(linspace, LinSpaceInferShapeCase2)
{
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace")->infer_shape;
  float start = 1.0;
  float stop = 10.0;
  bool num = true;
  auto start_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(float)]);
  auto start_tensor = ConstructInputConstTensor(start_input_tensor, start, ge::DT_FLOAT);
  auto stop_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(float)]);
  auto stop_tensor = ConstructInputConstTensor(stop_input_tensor, stop, ge::DT_FLOAT);
  auto num_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(bool)]);
  auto num_tensor = ConstructInputConstTensor(num_input_tensor, num, ge::DT_BOOL);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, stop_tensor, num_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 0);
}

TEST_F(linspace, LinSpaceInferDataTypeCase0) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LinSpace")->infer_datatype;
  ge::DataType input_ref = ge::DT_FLOAT;
  ge::DataType output_ref = ge::DT_FLOAT16;

  auto holder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputDataTypes({&input_ref, &input_ref, &input_ref})
                    .OutputDataTypes({&output_ref})
                    .Build();
  auto context = holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
