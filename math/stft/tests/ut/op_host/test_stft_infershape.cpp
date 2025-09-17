/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_STFT_proto.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "../../../../stft/op_graph/stft_proto.h"

class STFT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "STFT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "STFT TearDown" << std::endl;
    }
};

TEST_F(STFT, STFT_FP32_IN_1D_NO_WIN_ONESIDED_RETURN_REAL)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[2, 3, 2]");
}

TEST_F(STFT, STFT_CFP128_IN_1D_NO_WIN_TWOSIDED_RETURN_COMPLEX)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[3, 3]");
}

TEST_F(STFT, STFT_FP64_IN_2D_NO_WIN_ONESIDED_RETURN_REAL)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{1, 10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 2, 3, 2]");
}

TEST_F(STFT, STFT_CFP64_IN_2D_NO_WIN_TWOSIDED_RETURN_COMPLEX)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{1, 10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 3, 3]");
}

TEST_F(STFT, STFT_FP64_IN_CFP128_WIN_RETURN_COMPLEX)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[3, 3]");
}

TEST_F(STFT, STFT_FP64_IN_CFP128_WIN_RETURN_REAL)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[3, 3, 2]");
}

// exception instance
TEST_F(STFT, STFT_NO_N_FFT)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_INPUT_NOT_1D_2D)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{1, 1, 10}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOWS_NOT_1D)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{1, 3}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOWS_LEN_NOT_MATCH)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{10}, {}};
    gert::StorageShape window_shape = {{4}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferShapeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputShapes({&x_shape, &x_shape, &window_shape})
                              .OutputShapes({&outputShape})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_INPUT_WRONG_TYPE)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_FLOAT16;
    ge::DataType input_win_ref = ge::DT_COMPLEX128;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOW_WRONG_TYPE)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_FLOAT;
    ge::DataType input_win_ref = ge::DT_FLOAT16;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_DOUBLE_C128_INFER_DATATYPE_RETURN_DT_DOUBLE)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_DOUBLE;
    ge::DataType input_win_ref = ge::DT_COMPLEX128;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_DOUBLE);
}

TEST_F(STFT, STFT_FP32_C64_INFER_DATATYPE_RETURN_DT_FP32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_FLOAT;
    ge::DataType input_win_ref = ge::DT_COMPLEX64;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(STFT, STFT_FP32_C128_INFER_DATATYPE_RETURN_DT_DOUBLE)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_FLOAT;
    ge::DataType input_win_ref = ge::DT_COMPLEX128;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_DOUBLE);
}

TEST_F(STFT, STFT_DOUBLE_C128_INFER_DATATYPE_RETURN_C128)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_DOUBLE;
    ge::DataType input_win_ref = ge::DT_COMPLEX128;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX128);
}

TEST_F(STFT, STFT_FP32_C64_INFER_DATATYPE_RETURN_C64)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_FLOAT;
    ge::DataType input_win_ref = ge::DT_COMPLEX64;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX64);
}

TEST_F(STFT, STFT_FP32_C128_INFER_DATATYPE_RETURN_C128)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_x_ref = ge::DT_FLOAT;
    ge::DataType input_win_ref = ge::DT_COMPLEX128;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX128);
}

TEST_F(STFT, STFT_C64_INFER_DATATYPE_RETURN_C64)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_ref = ge::DT_COMPLEX64;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_ref, &input_ref, &input_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX64);
}

TEST_F(STFT, STFT_DOUBLE_INFER_DATATYPE_RETURN_DOUBLE)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_ref = ge::DT_DOUBLE;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_ref, &input_ref, &input_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_DOUBLE);
}

TEST_F(STFT, STFT_C128_INFER_DATATYPE_RETURN_C128)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_ref = ge::DT_COMPLEX128;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_ref, &input_ref, &input_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX128);
}

TEST_F(STFT, STFT_FP32_INFER_DATATYPE_RETURN_FP32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_ref, &input_ref, &input_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(STFT, STFT_FP32_INFER_DATATYPE_RETURN_COMPLEX64)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT16;

    gert::StorageShape outputShape = {{}, {}};
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .InputDataTypes({&input_ref, &input_ref, &input_ref})
                              .OutputDataTypes({&output_ref})
                              .NodeAttrs(
                                  {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
                                   {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
                                   {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
                                   {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
                              .Build();

    EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX64);
}