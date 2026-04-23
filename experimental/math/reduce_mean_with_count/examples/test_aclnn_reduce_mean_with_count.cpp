/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_reduce_mean_with_count.cpp
 * @brief ACLNN two-stage invocation example for ReduceMeanWithCount
 *
 * This example demonstrates the complete ACLNN calling flow:
 *   1. Initialize ACL environment
 *   2. Create input / output tensors on device
 *   3. Call aclnnReduceMeanWithCountGetWorkspaceSize (stage 1)
 *   4. Call aclnnReduceMeanWithCount (stage 2)
 *   5. Synchronize stream and read results
 *   6. Verify precision against CPU golden
 *   7. Release resources
 *
 * Build:
 *   source /path/to/cann/set_env.sh
 *   g++ -std=c++17 -o test_aclnn test_aclnn_reduce_mean_with_count.cpp \
 *       -I${ASCEND_HOME_PATH}/include \
 *       -L${ASCEND_HOME_PATH}/lib64 \
 *       -lascendcl -lnnopbase -lopapi -lcust_opapi
 *
 * Run:
 *   ./test_aclnn
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

// ============================================================
// aclnnReduceMeanWithCount API declaration
// Provided by the custom operator package (libcust_opapi.so).
// ============================================================
#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnReduceMeanWithCountGetWorkspaceSize(
    const aclTensor *input,
    const aclIntArray *axis,
    bool keepdim,
    aclTensor *meanResult,
    aclTensor *countResult,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnReduceMeanWithCount(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

// ============================================================
// Error checking macros
// ============================================================
#define CHECK_ACL(expr)                                                       \
    do {                                                                      \
        auto _ret = (expr);                                                   \
        if (_ret != 0) {                                                      \
            printf("[ERROR] %s:%d  %s  returned %d\n",                        \
                   __FILE__, __LINE__, #expr, static_cast<int>(_ret));        \
            return -1;                                                        \
        }                                                                     \
    } while (0)

#define CHECK_ACLNN(expr)                                                     \
    do {                                                                      \
        aclnnStatus _ret = (expr);                                            \
        if (_ret != 0) {                                                      \
            const char* _emsg = aclGetRecentErrMsg();                         \
            printf("[ERROR] %s:%d  %s  returned %d\n",                        \
                   __FILE__, __LINE__, #expr, static_cast<int>(_ret));        \
            if (_emsg) printf("[ERROR_DETAIL] %s\n", _emsg);                  \
            return -1;                                                        \
        }                                                                     \
    } while (0)

// ============================================================
// Utility: compute contiguous strides from shape
// ============================================================
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static int64_t NumElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

// ============================================================
// Utility: compute output shape
// ============================================================
static std::vector<int64_t> ComputeOutputShape(
    const std::vector<int64_t>& inputShape,
    const std::vector<int64_t>& axis,
    bool keepdim)
{
    int rank = static_cast<int>(inputShape.size());
    std::vector<int> normalizedAxes;
    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) normalizedAxes.push_back(i);
    } else {
        for (auto a : axis) {
            int na = static_cast<int>(a);
            if (na < 0) na += rank;
            normalizedAxes.push_back(na);
        }
    }
    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    normalizedAxes.erase(std::unique(normalizedAxes.begin(), normalizedAxes.end()),
                         normalizedAxes.end());

    std::vector<int64_t> outShape;
    for (int i = 0; i < rank; ++i) {
        bool isReduceAxis = std::find(normalizedAxes.begin(), normalizedAxes.end(), i)
                            != normalizedAxes.end();
        if (isReduceAxis) {
            if (keepdim) outShape.push_back(1);
        } else {
            outShape.push_back(inputShape[i]);
        }
    }
    return outShape;
}

// ============================================================
// CPU Golden: compute expected mean and count
// ============================================================
static void CpuGoldenReduceMeanWithCount(
    const float* input,
    const std::vector<int64_t>& inputShape,
    const std::vector<int64_t>& axis,
    float* goldenMean,
    int64_t* goldenCount)
{
    int rank = static_cast<int>(inputShape.size());
    std::vector<int> normalizedAxes;
    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) normalizedAxes.push_back(i);
    } else {
        for (auto a : axis) {
            int na = static_cast<int>(a);
            if (na < 0) na += rank;
            normalizedAxes.push_back(na);
        }
    }
    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    normalizedAxes.erase(std::unique(normalizedAxes.begin(), normalizedAxes.end()),
                         normalizedAxes.end());

    // Compute count = product of reduce-axis sizes
    int64_t count = 1;
    for (int ax : normalizedAxes) {
        count *= inputShape[ax];
    }

    // Output shape (keepdim=true for indexing)
    std::vector<int64_t> outShapeKeep(rank);
    for (int i = 0; i < rank; ++i) {
        bool isReduce = std::find(normalizedAxes.begin(), normalizedAxes.end(), i)
                        != normalizedAxes.end();
        outShapeKeep[i] = isReduce ? 1 : inputShape[i];
    }

    int64_t outCount = NumElements(outShapeKeep);
    std::vector<int64_t> inStrides = ComputeStrides(inputShape);
    std::vector<int64_t> outStrides = ComputeStrides(outShapeKeep);

    for (int64_t i = 0; i < outCount; ++i) {
        goldenMean[i] = 0.0f;
        goldenCount[i] = count;
    }

    int64_t totalIn = NumElements(inputShape);
    for (int64_t flatIdx = 0; flatIdx < totalIn; ++flatIdx) {
        std::vector<int64_t> indices(rank);
        int64_t rem = flatIdx;
        for (int d = 0; d < rank; ++d) {
            indices[d] = rem / inStrides[d];
            rem %= inStrides[d];
        }

        int64_t outFlat = 0;
        for (int d = 0; d < rank; ++d) {
            bool isReduce = std::find(normalizedAxes.begin(), normalizedAxes.end(), d)
                            != normalizedAxes.end();
            int64_t idx = isReduce ? 0 : indices[d];
            outFlat += idx * outStrides[d];
        }

        goldenMean[outFlat] += input[flatIdx];
    }

    for (int64_t i = 0; i < outCount; ++i) {
        goldenMean[i] /= static_cast<float>(count);
    }
}

// ============================================================
// Precision check: MERE / MARE
// ============================================================
static bool CheckPrecision(
    const float* actual, const float* golden, int64_t count,
    double mereThreshold, double mareThreshold)
{
    if (count == 0) return true;

    double sumRelErr = 0.0;
    double maxRelErr = 0.0;

    for (int64_t i = 0; i < count; ++i) {
        double a = static_cast<double>(actual[i]);
        double g = static_cast<double>(golden[i]);

        if (std::isnan(g)) {
            if (!std::isnan(a)) {
                printf("  [FAIL] Index %ld: golden=NaN, actual=%f\n", (long)i, a);
                return false;
            }
            continue;
        }
        if (std::isinf(g)) {
            if (!std::isinf(a) || std::signbit(a) != std::signbit(g)) {
                printf("  [FAIL] Index %ld: golden=%f, actual=%f (inf mismatch)\n",
                       (long)i, g, a);
                return false;
            }
            continue;
        }

        double relErr = std::abs(a - g) / (std::abs(g) + 1e-7);
        sumRelErr += relErr;
        if (relErr > maxRelErr) maxRelErr = relErr;
    }

    double mere = sumRelErr / static_cast<double>(count);
    double mare = maxRelErr;

    printf("  MERE = %.10e  (threshold: %.10e)\n", mere, mereThreshold);
    printf("  MARE = %.10e  (threshold: %.10e)\n", mare, mareThreshold);

    if (mere >= mereThreshold || mare >= mareThreshold) {
        printf("  [FAIL] Precision check failed\n");
        return false;
    }

    printf("  [PASS] Precision check passed\n");
    return true;
}

// ============================================================
// FP32 <-> FP16 / BF16 software conversions (IEEE-754)
// Avoid dependency on compiler-specific _Float16 / __bf16 intrinsics.
// ============================================================
static uint16_t FloatToFp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {
        // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {
        // Overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t halfMant = mant >> shift;
        // Round to nearest even
        uint32_t roundBit = (mant >> (shift - 1)) & 1u;
        halfMant += roundBit;
        return static_cast<uint16_t>(sign | halfMant);
    }
    uint32_t halfMant = mant >> 13;
    uint32_t roundBit = (mant >> 12) & 1u;
    uint32_t stickyBits = mant & 0xFFFu;
    if (roundBit && (stickyBits || (halfMant & 1u))) {
        halfMant += 1;
        if (halfMant == 0x400u) { // mantissa overflow -> bump exponent
            halfMant = 0;
            exp += 1;
            if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

static float Fp16ToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Subnormal -> normalize
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            exp += 1;
            mant &= 0x3FFu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static uint16_t FloatToBf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    // Round-to-nearest-even
    uint32_t lsb = (x >> 16) & 1u;
    uint32_t rounding = 0x7FFFu + lsb;
    x += rounding;
    return static_cast<uint16_t>(x >> 16);
}

static float Bf16ToFloat(uint16_t b) {
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// ============================================================
// Main
// ============================================================
static int RunTestCase(
    aclDataType inputDtype,
    const char* dtypeName,
    const std::vector<int64_t>& inputShape,
    const std::vector<int64_t>& axis,
    bool keepdim,
    double mereThreshold,
    double mareThreshold,
    bool& outPass)
{
    printf("\n============================================================\n");
    printf("  Running test case: dtype=%s\n", dtypeName);
    printf("============================================================\n");

    std::vector<int64_t> outputShape = ComputeOutputShape(inputShape, axis, keepdim);
    int64_t inputCount = NumElements(inputShape);
    int64_t outputCount = outputShape.empty() ? 1 : NumElements(outputShape);

    printf("  Input shape: ["); for (size_t i = 0; i < inputShape.size(); ++i) {
        printf("%ld%s", (long)inputShape[i], i + 1 == inputShape.size() ? "" : ", ");
    } printf("]  (%ld elements)\n", (long)inputCount);
    printf("  Axis: ["); for (size_t i = 0; i < axis.size(); ++i) {
        printf("%ld%s", (long)axis[i], i + 1 == axis.size() ? "" : ", ");
    } printf("]\n");
    printf("  Keepdim: %s\n", keepdim ? "true" : "false");
    printf("  Output element count: %ld\n", (long)outputCount);

    // Generate random FP32 reference data on host (shared across dtypes for reproducibility).
    std::vector<float> hostInputF32(inputCount);
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < inputCount; ++i) {
            hostInputF32[i] = dist(gen);
        }
    }

    // For FP16/BF16 cases, round-trip the host input through the target precision so
    // that the CPU golden reflects the same quantization the device will see.
    if (inputDtype == ACL_FLOAT16) {
        for (int64_t i = 0; i < inputCount; ++i) {
            hostInputF32[i] = Fp16ToFloat(FloatToFp16(hostInputF32[i]));
        }
    } else if (inputDtype == ACL_BF16) {
        for (int64_t i = 0; i < inputCount; ++i) {
            hostInputF32[i] = Bf16ToFloat(FloatToBf16(hostInputF32[i]));
        }
    }

    // CPU golden
    std::vector<int64_t> outShapeKeep = ComputeOutputShape(inputShape, axis, true);
    int64_t outCountKeep = outShapeKeep.empty() ? 1 : NumElements(outShapeKeep);
    std::vector<float> goldenMean(outCountKeep);
    std::vector<int64_t> goldenCount(outCountKeep);
    CpuGoldenReduceMeanWithCount(hostInputF32.data(), inputShape, axis,
                                  goldenMean.data(), goldenCount.data());

    // Pack host input into the target dtype buffer
    size_t elemBytes = (inputDtype == ACL_FLOAT) ? sizeof(float) : sizeof(uint16_t);
    std::vector<uint8_t> hostInputBytes(static_cast<size_t>(inputCount) * elemBytes);
    if (inputDtype == ACL_FLOAT) {
        std::memcpy(hostInputBytes.data(), hostInputF32.data(),
                    static_cast<size_t>(inputCount) * sizeof(float));
    } else if (inputDtype == ACL_FLOAT16) {
        auto* dst = reinterpret_cast<uint16_t*>(hostInputBytes.data());
        for (int64_t i = 0; i < inputCount; ++i) dst[i] = FloatToFp16(hostInputF32[i]);
    } else { // ACL_BF16
        auto* dst = reinterpret_cast<uint16_t*>(hostInputBytes.data());
        for (int64_t i = 0; i < inputCount; ++i) dst[i] = FloatToBf16(hostInputF32[i]);
    }

    // Device allocations
    size_t inputBytes = static_cast<size_t>(inputCount) * elemBytes;
    size_t meanBytes = static_cast<size_t>(outputCount) * elemBytes;
    size_t countBytes = static_cast<size_t>(outputCount) * sizeof(int64_t);

    void* devInput = nullptr;
    void* devMean = nullptr;
    void* devCount = nullptr;

    CHECK_ACL(aclrtMalloc(&devInput, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devMean, meanBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devCount, countBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(devInput, inputBytes, hostInputBytes.data(), inputBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(devMean, meanBytes, 0, meanBytes));
    CHECK_ACL(aclrtMemset(devCount, countBytes, 0, countBytes));

    // Tensors
    std::vector<int64_t> inputStrides = ComputeStrides(inputShape);
    aclTensor* inputTensor = aclCreateTensor(
        inputShape.data(), inputShape.size(),
        inputDtype,
        inputStrides.data(), 0,
        ACL_FORMAT_ND,
        inputShape.data(), inputShape.size(),
        devInput);

    std::vector<int64_t> outputStrides = ComputeStrides(outputShape);
    std::vector<int64_t> outputStorageShape = outputShape;
    if (outputStorageShape.empty()) outputStorageShape.push_back(1);
    std::vector<int64_t> outputStridesForCreate = outputStrides;
    if (outputStridesForCreate.empty()) outputStridesForCreate.push_back(1);

    aclTensor* meanTensor = aclCreateTensor(
        outputShape.data(), outputShape.size(),
        inputDtype,
        outputStridesForCreate.data(), 0,
        ACL_FORMAT_ND,
        outputStorageShape.data(), outputStorageShape.size(),
        devMean);

    aclTensor* countTensor = aclCreateTensor(
        outputShape.data(), outputShape.size(),
        ACL_INT64,
        outputStridesForCreate.data(), 0,
        ACL_FORMAT_ND,
        outputStorageShape.data(), outputStorageShape.size(),
        devCount);

    aclIntArray* axisArr = aclCreateIntArray(axis.data(), axis.size());

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    printf("  Calling aclnnReduceMeanWithCountGetWorkspaceSize...\n");
    CHECK_ACLNN(aclnnReduceMeanWithCountGetWorkspaceSize(
        inputTensor, axisArr, keepdim,
        meanTensor, countTensor,
        &workspaceSize, &executor));
    printf("  WorkspaceSize: %lu bytes\n", (unsigned long)workspaceSize);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    printf("  Calling aclnnReduceMeanWithCount...\n");
    CHECK_ACLNN(aclnnReduceMeanWithCount(workspace, workspaceSize, executor, stream));

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Pull device output back and decode to FP32 for comparison
    std::vector<uint8_t> hostMeanBytes(meanBytes);
    std::vector<int64_t> hostCount(outputCount);
    CHECK_ACL(aclrtMemcpy(hostMeanBytes.data(), meanBytes, devMean, meanBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(hostCount.data(), countBytes, devCount, countBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostMean(outputCount);
    if (inputDtype == ACL_FLOAT) {
        std::memcpy(hostMean.data(), hostMeanBytes.data(),
                    static_cast<size_t>(outputCount) * sizeof(float));
    } else if (inputDtype == ACL_FLOAT16) {
        auto* src = reinterpret_cast<const uint16_t*>(hostMeanBytes.data());
        for (int64_t i = 0; i < outputCount; ++i) hostMean[i] = Fp16ToFloat(src[i]);
    } else { // ACL_BF16
        auto* src = reinterpret_cast<const uint16_t*>(hostMeanBytes.data());
        for (int64_t i = 0; i < outputCount; ++i) hostMean[i] = Bf16ToFloat(src[i]);
    }

    printf("  --- Precision Check (%s) ---\n", dtypeName);
    bool precPass = CheckPrecision(hostMean.data(), goldenMean.data(), outputCount,
                                    mereThreshold, mareThreshold);

    bool countPass = true;
    for (int64_t i = 0; i < outputCount; ++i) {
        if (hostCount[i] != goldenCount[i]) {
            printf("  [FAIL] count[%ld] actual=%ld golden=%ld\n",
                   (long)i, (long)hostCount[i], (long)goldenCount[i]);
            countPass = false;
        }
    }
    printf("  Count exact match: %s\n", countPass ? "PASS" : "FAIL");

    outPass = precPass && countPass;
    printf("  Case result (%s): %s\n", dtypeName, outPass ? "PASS" : "FAIL");

    // Cleanup
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(meanTensor);
    aclDestroyTensor(countTensor);
    aclDestroyIntArray(axisArr);
    aclrtFree(devInput);
    aclrtFree(devMean);
    aclrtFree(devCount);
    return 0;
}

int main() {
    printf("============================================================\n");
    printf("  aclnnReduceMeanWithCount ACLNN Example (FP32/FP16/BF16)\n");
    printf("============================================================\n");

    CHECK_ACL(aclInit(nullptr));
    int deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    // Shared case: input shape [2, 3, 4], axis=[1], keepdim=false -> output [2, 4]
    std::vector<int64_t> inputShape = {2, 3, 4};
    std::vector<int64_t> axis = {1};
    bool keepdim = false;

    // Precision thresholds per dtype.
    // FP32:  MERE < 2^-13,   MARE < 10 * 2^-13
    // FP16:  MERE < 2^-10,   MARE < 10 * 2^-10
    // BF16:  MERE < 2^-7,    MARE < 10 * 2^-7
    bool passFp32 = false, passFp16 = false, passBf16 = false;

    int rc = 0;
    rc = RunTestCase(ACL_FLOAT, "FLOAT32", inputShape, axis, keepdim,
                     std::pow(2.0, -13.0), 10.0 * std::pow(2.0, -13.0), passFp32);
    if (rc) { aclrtResetDevice(deviceId); aclFinalize(); return rc; }

    rc = RunTestCase(ACL_FLOAT16, "FLOAT16", inputShape, axis, keepdim,
                     std::pow(2.0, -10.0), 10.0 * std::pow(2.0, -10.0), passFp16);
    if (rc) { aclrtResetDevice(deviceId); aclFinalize(); return rc; }

    rc = RunTestCase(ACL_BF16, "BFLOAT16", inputShape, axis, keepdim,
                     std::pow(2.0, -7.0), 10.0 * std::pow(2.0, -7.0), passBf16);
    if (rc) { aclrtResetDevice(deviceId); aclFinalize(); return rc; }

    bool overall = passFp32 && passFp16 && passBf16;
    printf("\n============================================================\n");
    printf("  SUMMARY:\n");
    printf("    FLOAT32:  %s\n", passFp32 ? "PASS" : "FAIL");
    printf("    FLOAT16:  %s\n", passFp16 ? "PASS" : "FAIL");
    printf("    BFLOAT16: %s\n", passBf16 ? "PASS" : "FAIL");
    printf("  RESULT: %s\n", overall ? "PASS" : "FAIL");
    printf("============================================================\n");

    aclrtResetDevice(deviceId);
    aclFinalize();
    return overall ? 0 : 1;
}
