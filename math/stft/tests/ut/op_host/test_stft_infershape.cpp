/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stft_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

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

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(
    std::vector<std::vector<int64_t> > expectResults,
    const gert::StorageShape& xShape, const gert::StorageShape& windowShape, gert::StorageShape& outputShape,
    ge::DataType inputDtype, ge::DataType windowDtype, ge::DataType outputDtype, int64_t hopLength, int64_t winLength,
    bool normalized, bool onesided, bool returnComplex, int64_t nFft,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xShape, (gert::Tensor*)&xShape, (gert::Tensor*)&windowShape};
    std::vector<gert::StorageShape*> outputShapes = {&outputShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("STFT")
                             .NodeIoNum(3, 1)
                             .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, windowDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("hop_length", hopLength)
                             .Attr("win_length", winLength)
                             .Attr("normalized", normalized)
                             .Attr("onesided", onesided)
                             .Attr("return_complex", returnComplex)
                             .Attr("n_fft", nFft)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

static void ExeTestCaseNoNFft(
    const gert::StorageShape& xShape, const gert::StorageShape& windowShape, gert::StorageShape& outputShape,
    ge::DataType inputDtype, ge::DataType windowDtype, ge::DataType outputDtype, int64_t hopLength, int64_t winLength,
    bool normalized, bool onesided, bool returnComplex, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xShape, (gert::Tensor*)&xShape, (gert::Tensor*)&windowShape};
    std::vector<gert::StorageShape*> outputShapes = {&outputShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("STFT")
                             .NodeIoNum(3, 1)
                             .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, windowDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("hop_length", hopLength)
                             .Attr("win_length", winLength)
                             .Attr("normalized", normalized)
                             .Attr("onesided", onesided)
                             .Attr("return_complex", returnComplex)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(STFT, STFT_FP32_IN_1D_NO_WIN_ONESIDED_RETURN_REAL)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    std::vector<int64_t> expectResult = {2, 3, 2};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_CFP128_IN_1D_NO_WIN_TWOSIDED_RETURN_COMPLEX)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = true;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_COMPLEX64;
    std::vector<int64_t> expectResult = {3, 3};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_FP64_IN_2D_NO_WIN_ONESIDED_RETURN_REAL)
{
    gert::StorageShape xShape = {{1, 10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    std::vector<int64_t> expectResult = {1, 2, 3, 2};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_CFP64_IN_2D_NO_WIN_TWOSIDED_RETURN_COMPLEX)
{
    gert::StorageShape xShape = {{1, 10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = true;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_COMPLEX64;
    std::vector<int64_t> expectResult = {1, 3, 3};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_FP64_IN_CFP128_WIN_RETURN_COMPLEX)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = true;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_COMPLEX64;
    std::vector<int64_t> expectResult = {3, 3};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_FP64_IN_CFP128_WIN_RETURN_REAL)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    std::vector<int64_t> expectResult = {3, 3, 2};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

// exception instance
TEST_F(STFT, STFT_NO_N_FFT)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCaseNoNFft(
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_INPUT_NOT_1D_2D)
{
    gert::StorageShape xShape = {{1, 1, 10}, {}};
    gert::StorageShape windowShape = {{1}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCase({},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOWS_NOT_1D)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{1, 3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCase({},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOWS_LEN_NOT_MATCH)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{4}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCase({},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_FAILED);
}
