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
 * @file test_aclnn_asin_grad.cpp
 * @brief AsinGrad ACLNN call example
 *
 * Demonstrates the two-phase ACLNN call pattern for the AsinGrad operator:
 *   1. aclnnAsinGradGetWorkspaceSize  - compute workspace size, create executor
 *   2. aclnnAsinGrad                  - execute computation on NPU
 *
 * Formula: dx = dy / sqrt(1 - x*x)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_asin_grad.h"

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_ACL(expr)                                              \
    do {                                                             \
        auto _ret = (expr);                                          \
        if (_ret != ACL_SUCCESS) {                                   \
            LOG_PRINT("[ERROR] %s failed, ret = %d", #expr, _ret);   \
            return _ret;                                             \
        }                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// Utility: compute contiguous strides from shape
// ---------------------------------------------------------------------------
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// ---------------------------------------------------------------------------
// Utility: total number of elements
// ---------------------------------------------------------------------------
int64_t GetNumElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ---------------------------------------------------------------------------
// Create an aclTensor backed by device memory, copying hostData to device
// ---------------------------------------------------------------------------
int CreateAclTensor(const void* hostData,
                    size_t byteSize,
                    const std::vector<int64_t>& shape,
                    aclDataType dtype,
                    void** deviceAddr,
                    aclTensor** tensor) {
    auto ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(*deviceAddr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
                              *deviceAddr);
    return ACL_SUCCESS;
}

// ---------------------------------------------------------------------------
// CPU golden: dx = dy / sqrt(1 - x*x)   (double precision intermediate)
// ---------------------------------------------------------------------------
void ComputeGolden(const float* dy, const float* x, float* dx, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double xd = static_cast<double>(x[i]);
        double dyd = static_cast<double>(dy[i]);
        dx[i] = static_cast<float>(dyd / std::sqrt(1.0 - xd * xd));
    }
}

// ---------------------------------------------------------------------------
// Verify: MERE / MARE check (threshold = 2^-13 for fp32)
// ---------------------------------------------------------------------------
bool VerifyResults(const float* golden, const float* actual, size_t n) {
    const double threshold = 1.220703125e-4;  // 2^-13
    double mere_sum = 0.0;
    double mare = 0.0;
    size_t finite_count = 0;

    for (size_t i = 0; i < n; ++i) {
        double g = golden[i];
        double a = actual[i];

        if (std::isnan(g)) {
            if (!std::isnan(a)) {
                LOG_PRINT("[FAIL] NaN mismatch at [%zu]: golden=NaN, actual=%.6f", i, a);
                return false;
            }
            continue;
        }
        if (std::isinf(g)) {
            if (!std::isinf(a) || std::signbit(g) != std::signbit(a)) {
                LOG_PRINT("[FAIL] Inf mismatch at [%zu]: golden=%.2f, actual=%.2f", i, g, a);
                return false;
            }
            continue;
        }

        double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
        mere_sum += rel_err;
        if (rel_err > mare) mare = rel_err;
        finite_count++;
    }

    if (finite_count == 0) {
        LOG_PRINT("[PASS] All elements are special values, matched.");
        return true;
    }

    double mere = mere_sum / static_cast<double>(finite_count);
    bool pass = (mere < threshold) && (mare < 10.0 * threshold);

    if (pass) {
        LOG_PRINT("[PASS] MERE=%.2e, MARE=%.2e (threshold=%.2e, %zu finite elems)",
                  mere, mare, threshold, finite_count);
    } else {
        LOG_PRINT("[FAIL] MERE=%.2e, MARE=%.2e (threshold=%.2e)", mere, mare, threshold);
    }
    return pass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    LOG_PRINT("========================================");
    LOG_PRINT("AsinGrad ACLNN Call Example (fp32)");
    LOG_PRINT("========================================");

    // ----- 1. Initialize ACL -----
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    // ----- 2. Prepare input data -----
    // Shape: [4, 8] = 32 elements
    std::vector<int64_t> shape = {4, 8};
    int64_t numel = GetNumElements(shape);
    size_t byteSize = numel * sizeof(float);

    // dy: gradient values in [-1, 1]
    // x : forward input values in [-0.9, 0.9] (within asin domain)
    std::vector<float> dy_host(numel);
    std::vector<float> x_host(numel);
    for (int64_t i = 0; i < numel; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(numel - 1);
        dy_host[i] = -1.0f + 2.0f * t;           // [-1, 1]
        x_host[i]  = -0.9f + 1.8f * t;           // [-0.9, 0.9]
    }

    // Output buffer (zeros)
    std::vector<float> dx_host(numel, 0.0f);

    // ----- 3. Create aclTensors (host -> device) -----
    void* dy_dev = nullptr;
    void* x_dev  = nullptr;
    void* dx_dev = nullptr;
    aclTensor* dy_tensor = nullptr;
    aclTensor* x_tensor  = nullptr;
    aclTensor* dx_tensor = nullptr;

    CHECK_ACL(CreateAclTensor(dy_host.data(), byteSize, shape, ACL_FLOAT, &dy_dev, &dy_tensor));
    CHECK_ACL(CreateAclTensor(x_host.data(),  byteSize, shape, ACL_FLOAT, &x_dev,  &x_tensor));
    CHECK_ACL(CreateAclTensor(dx_host.data(), byteSize, shape, ACL_FLOAT, &dx_dev, &dx_tensor));

    // ----- 4. Phase 1: GetWorkspaceSize -----
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnAsinGradGetWorkspaceSize(dy_tensor, x_tensor, dx_tensor,
                                              &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetWorkspaceSize failed: %d", ret); return 1);
    LOG_PRINT("workspaceSize = %lu", workspaceSize);

    // ----- 5. Allocate workspace (if needed) -----
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // ----- 6. Phase 2: Execute -----
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    ret = aclnnAsinGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAsinGrad failed: %d", ret); return 1);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    LOG_PRINT("aclnnAsinGrad executed successfully.");

    // ----- 7. Copy result back to host -----
    std::vector<float> npu_output(numel);
    CHECK_ACL(aclrtMemcpy(npu_output.data(), byteSize, dx_dev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // ----- 8. Verify against CPU golden -----
    std::vector<float> golden(numel);
    ComputeGolden(dy_host.data(), x_host.data(), golden.data(), numel);

    LOG_PRINT("\nVerification (fp32, shape=[4,8], numel=%ld):", numel);
    bool passed = VerifyResults(golden.data(), npu_output.data(), numel);

    // Print first few results
    LOG_PRINT("\nSample outputs (first 8 elements):");
    LOG_PRINT("  %-8s %-12s %-12s %-12s", "Index", "dy", "x", "dx(NPU)");
    for (int i = 0; i < 8 && i < numel; ++i) {
        LOG_PRINT("  %-8d %-12.6f %-12.6f %-12.6f", i, dy_host[i], x_host[i], npu_output[i]);
    }

    // ----- 9. Cleanup -----
    aclrtDestroyStream(stream);
    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(dy_tensor); aclrtFree(dy_dev);
    aclDestroyTensor(x_tensor);  aclrtFree(x_dev);
    aclDestroyTensor(dx_tensor); aclrtFree(dx_dev);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n========================================");
    LOG_PRINT("Result: %s", passed ? "PASS" : "FAIL");
    LOG_PRINT("========================================");

    return passed ? 0 : 1;
}
