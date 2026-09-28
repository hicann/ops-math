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
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/bessel_i1e_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;

static std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

static uint32_t GetDataTypeSize(DataType dt)
{
    switch (dt) {
        case DT_FLOAT:
            return 4;
        case DT_FLOAT16:
            return 2;
        case DT_BF16:
            return 2;
        default:
            return 4;
    }
}

// ============================================================
// Reference implementation: i1e(x) = sign(x) * exp(-|x|) * I1(|x|)
// Cephes-style piecewise polynomial approximation (matches kernel).
// ============================================================
static float ReferenceI1e(float x)
{
    float ax = std::fabs(x);
    float sign = (x < 0.0f) ? -1.0f : 1.0f;

    if (ax == 0.0f) {
        return 0.0f;
    }
    if (std::isnan(x)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (std::isinf(x)) {
        return 0.0f;
    }

    // Mathematically correct reference: I1(x) via power series (double precision)
    // I1(x) = (x/2) * sum_{k=0}^{inf} (x^2/4)^k / (k! * (k+1)!)
    // i1e(x) = I1(x) * exp(-|x|)
    double axd = static_cast<double>(ax);
    double sum = 0.0;
    double term = 1.0;
    double x2_4 = axd * axd / 4.0;
    double fact_k = 1.0;
    double fact_k1 = 1.0;

    for (int k = 0; k < 200; k++) {
        if (k > 0) {
            fact_k *= static_cast<double>(k);
            fact_k1 *= static_cast<double>(k + 1);
        }
        term = std::pow(x2_4, k) / (fact_k * fact_k1);
        sum += term;
        if (term < 1e-17 * std::fabs(sum)) {
            break;
        }
    }

    double i1 = (axd / 2.0) * sum;
    double i1e_val = i1 * std::exp(-axd);

    return sign * static_cast<float>(i1e_val);
}

// Forward declaration for max error tracking
static float max_abs_err_cached = 0.0f;

// ============================================================
// Verification: check output shape, dtype, and element-wise values
// ============================================================
struct VerifyResult {
    bool shape_ok;
    bool dtype_ok;
    bool value_ok;
    size_t total_elements;
    size_t mismatch_count;
    float max_abs_err;
    float max_rel_err;
};

static VerifyResult VerifyOutput(const ge::Tensor& output, const std::vector<int64_t>& expected_shape,
                                 DataType expected_dtype, const std::vector<float>& input_data, float rtol, float atol)
{
    VerifyResult vr = {true, true, true, 0, 0, 0.0f, 0.0f};

    // --- Shape verification ---
    auto out_desc = output.GetTensorDesc();
    auto out_shape = out_desc.GetShape().GetDims();
    vr.shape_ok = (out_shape.size() == expected_shape.size());
    if (vr.shape_ok) {
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (out_shape[i] != expected_shape[i]) {
                vr.shape_ok = false;
                break;
            }
        }
    }
    vr.total_elements = out_desc.GetShape().GetShapeSize();

    // --- Dtype verification ---
    DataType out_dtype = out_desc.GetDataType();
    vr.dtype_ok = (out_dtype == expected_dtype);

    // --- Value verification (float32 path) ---
    if (expected_dtype == DT_FLOAT) {
        const float* out_data = reinterpret_cast<const float*>(output.GetData());
        for (int64_t i = 0; i < static_cast<int64_t>(vr.total_elements); ++i) {
            float expected = ReferenceI1e(input_data[i]);
            float actual = out_data[i];
            float abs_err = std::fabs(expected - actual);
            float rel_err = (std::fabs(expected) > 1e-30f) ? abs_err / std::fabs(expected) : abs_err;
            if (abs_err > max_abs_err_cached)
                max_abs_err_cached = abs_err;
            if (rel_err > vr.max_rel_err)
                vr.max_rel_err = rel_err;
            if (abs_err > atol && rel_err > rtol) {
                vr.mismatch_count++;
                if (vr.mismatch_count <= 5) {
                    printf("  MISMATCH[%lld]: input=%.6f expected=%.8f got=%.8f abs_err=%.2e rel_err=%.2e\n",
                           static_cast<long long>(i), input_data[i], expected, actual, abs_err, rel_err);
                }
            }
        }
        vr.max_abs_err = max_abs_err_cached;
        vr.value_ok = (vr.mismatch_count == 0);
    } else {
        // For FP16/BF16, verify against reference with wider tolerance
        const uint16_t* out_data = reinterpret_cast<const uint16_t*>(output.GetData());
        float fp16_rtol = (expected_dtype == DT_BF16) ? 0.01f : 0.002f;
        float fp16_atol = (expected_dtype == DT_BF16) ? 0.01f : 0.002f;
        for (int64_t i = 0; i < static_cast<int64_t>(vr.total_elements); ++i) {
            float expected = ReferenceI1e(input_data[i]);
            // Decode fp16/bf16 to float
            float actual = 0.0f;
            if (expected_dtype == DT_FLOAT16) {
                uint16_t h = out_data[i];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t frac = h & 0x3FF;
                if (exp == 0) {
                    actual = (frac == 0) ? 0.0f : std::ldexp(static_cast<float>(frac), -24);
                } else if (exp == 31) {
                    actual = (frac == 0) ? std::numeric_limits<float>::infinity() :
                                           std::numeric_limits<float>::quiet_NaN();
                } else {
                    actual = std::ldexp(static_cast<float>((1 << 10) | frac), static_cast<int>(exp) - 25);
                }
                if (sign)
                    actual = -actual;
            } else { // BF16
                uint16_t h = out_data[i];
                uint32_t bits = static_cast<uint32_t>(h) << 16;
                std::memcpy(&actual, &bits, 4);
            }
            float abs_err = std::fabs(expected - actual);
            float rel_err = (std::fabs(expected) > 1e-30f) ? abs_err / std::fabs(expected) : abs_err;
            if (abs_err > vr.max_abs_err)
                vr.max_abs_err = abs_err;
            if (rel_err > vr.max_rel_err)
                vr.max_rel_err = rel_err;
            if (abs_err > fp16_atol && rel_err > fp16_rtol) {
                vr.mismatch_count++;
                if (vr.mismatch_count <= 5) {
                    printf("  MISMATCH[%lld]: input=%.6f expected=%.8f got=%.8f abs_err=%.2e\n",
                           static_cast<long long>(i), input_data[i], expected, actual, abs_err);
                }
            }
        }
        vr.value_ok = (vr.mismatch_count == 0);
    }

    return vr;
}

static int32_t GenTestData(const std::vector<int64_t>& shapes, Tensor& tensor, TensorDesc& desc, DataType dtype,
                           std::vector<float>& raw_data)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (auto d : shapes)
        size *= d;

    raw_data.resize(size);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < size; ++i) {
        raw_data[i] = dist(gen);
    }

    size_t data_len = size * GetDataTypeSize(dtype);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr)
        return FAILED;

    if (dtype == DT_FLOAT) {
        std::memcpy(pData, raw_data.data(), data_len);
    } else if (dtype == DT_FLOAT16) {
        uint16_t* h = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            // Simple fp32→fp16 conversion (round-to-nearest-even approximation)
            uint32_t bits;
            std::memcpy(&bits, &raw_data[i], 4);
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t frac = (bits >> 13) & 0x3FF;
            if (exp <= 0) {
                h[i] = static_cast<uint16_t>(sign);
            } else if (exp >= 31) {
                h[i] = static_cast<uint16_t>(sign | 0x7C00);
            } else {
                h[i] = static_cast<uint16_t>(sign | (exp << 10) | frac);
            }
        }
    } else { // DT_BF16
        uint16_t* h = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &raw_data[i], 4);
            h[i] = static_cast<uint16_t>(bits >> 16);
        }
    }

    tensor = Tensor(desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

static int32_t WriteDataToFile(const std::string& bin_file, uint64_t data_size, uint8_t* data)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr)
        return FAILED;
    fwrite(data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOpInGraph(DataType inDtype, std::vector<int64_t> xShape, std::vector<ge::Tensor>& input,
                    std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph,
                    std::vector<float>& raw_data)
{
    Status ret = SUCCESS;
    auto bessel_op = op::BesselI1e("bessel_i1e_0");

    auto placeholder0 = op::Data("placeholder0").set_attr_index(0);
    TensorDesc placeholder0_desc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    placeholder0_desc.SetPlacement(ge::kPlacementHost);
    placeholder0_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder0;
    ret = GenTestData(xShape, tensor_placeholder0, placeholder0_desc, inDtype, raw_data);
    if (ret != SUCCESS) {
        printf("%s - ERROR: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder0.update_input_desc_x(placeholder0_desc);
    input.push_back(tensor_placeholder0);
    graph.AddOp(placeholder0);
    bessel_op.set_input_x(placeholder0);

    TensorDesc y_desc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    bessel_op.update_output_desc_y(y_desc);

    inputs.push_back(placeholder0);
    outputs.push_back(bessel_op);
    return SUCCESS;
}

// Run one test case: build graph, execute, verify
static int RunTestCase(ge::Session* session, DataType dtype, const std::vector<int64_t>& shape,
                       const std::string& dtype_name, uint32_t& graph_id_counter)
{
    printf("\n%s - INFO: === Test case: %s, shape=[", GetTime().c_str(), dtype_name.c_str());
    for (size_t i = 0; i < shape.size(); ++i) {
        printf("%lld%s", static_cast<long long>(shape[i]), (i < shape.size() - 1) ? "," : "");
    }
    printf("] ===\n");

    Graph graph("bessel_i1e_geir_test");
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    std::vector<float> raw_data;

    if (CreateOpInGraph(dtype, shape, input, inputs, outputs, graph, raw_data) != SUCCESS) {
        printf("%s - ERROR: CreateOpInGraph failed\n", GetTime().c_str());
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t gid = graph_id_counter++;
    Status ret = session->AddGraph(gid, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR: AddGraph failed (ret=%d)\n", GetTime().c_str(), static_cast<int>(ret));
        return FAILED;
    }

    printf("%s - INFO: Running graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(gid, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR: RunGraph failed (ret=%d)\n", GetTime().c_str(), static_cast<int>(ret));
        return FAILED;
    }
    if (output.empty()) {
        printf("%s - ERROR: No output tensor returned\n", GetTime().c_str());
        return FAILED;
    }

    // Write output to file
    std::string output_file = "./bessel_i1e_geir_output_" + dtype_name + "_" + std::to_string(gid) + ".bin";
    uint8_t* output_data = output[0].GetData();
    int64_t output_shape_size = output[0].GetTensorDesc().GetShape().GetShapeSize();
    uint64_t data_size = static_cast<uint64_t>(output_shape_size) *
                         GetDataTypeSize(output[0].GetTensorDesc().GetDataType());
    WriteDataToFile(output_file, data_size, output_data);
    printf("%s - INFO: Output written to %s (%lu bytes)\n", GetTime().c_str(), output_file.c_str(), data_size);

    // --- Verification ---
    float rtol = 1e-4f, atol = 1e-4f;
    if (dtype == DT_FLOAT16) {
        rtol = 2e-3f;
        atol = 2e-3f;
    }
    if (dtype == DT_BF16) {
        rtol = 1e-2f;
        atol = 1e-2f;
    }

    VerifyResult vr = VerifyOutput(output[0], shape, dtype, raw_data, rtol, atol);

    printf("%s - INFO: Verification results:\n", GetTime().c_str());
    printf("  Shape:  %s (expected=[", vr.shape_ok ? "PASS" : "FAIL");
    for (size_t i = 0; i < shape.size(); ++i) {
        printf("%lld%s", static_cast<long long>(shape[i]), (i < shape.size() - 1) ? "," : "");
    }
    printf("])\n");
    printf("  Dtype:  %s (expected=%s)\n", vr.dtype_ok ? "PASS" : "FAIL", dtype_name.c_str());
    printf("  Values: %s (%zu/%lld elements, %zu mismatches, max_abs=%.6e, max_rel=%.6e)\n",
           vr.value_ok ? "PASS" : "FAIL", vr.total_elements, static_cast<long long>(vr.total_elements),
           vr.mismatch_count, vr.max_abs_err, vr.max_rel_err);

    bool all_ok = vr.shape_ok && vr.dtype_ok && vr.value_ok;
    printf("  Overall: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? SUCCESS : FAILED;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO: Start to initialize GE\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR: GE initialize failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO: Creating session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR: Create session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graph_id_counter = 0;
    int total_pass = 0, total_fail = 0;

    // === Test matrix: 3 dtypes × multiple shapes ===
    struct TestCase {
        DataType dtype;
        const char* name;
        std::vector<int64_t> shape;
    };
    std::vector<TestCase> cases = {
        // float32
        {DT_FLOAT, "float32", {32, 4, 4, 4}},
        {DT_FLOAT, "float32", {128}},
        {DT_FLOAT, "float32", {7, 11}},
        {DT_FLOAT, "float32", {2, 3, 4}},
        // float16
        {DT_FLOAT16, "float16", {16, 8}},
        {DT_FLOAT16, "float16", {64}},
        // bfloat16
        {DT_BF16, "bfloat16", {8, 8}},
        {DT_BF16, "bfloat16", {32}},
    };

    for (const auto& tc : cases) {
        int result = RunTestCase(session, tc.dtype, tc.shape, tc.name, graph_id_counter);
        if (result == SUCCESS) {
            total_pass++;
        } else {
            total_fail++;
        }
    }

    printf("\n%s - INFO: === Summary: %d PASS, %d FAIL (total %zu cases) ===\n", GetTime().c_str(), total_pass,
           total_fail, cases.size());

    printf("%s - INFO: Finalizing\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR: GE finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO: Done\n", GetTime().c_str());
    return (total_fail == 0) ? SUCCESS : FAILED;
}
