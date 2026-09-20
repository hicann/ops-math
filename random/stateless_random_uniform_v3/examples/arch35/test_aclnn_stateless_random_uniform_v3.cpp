/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GLIBCXX_USE_CXX11_ABI 0

#include "../../op_api/stateless_random_uniform_v3.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include <iostream>

using namespace op;

int main()
{
    int32_t deviceId = 0;
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice failed, ret: " << ret << std::endl;
        return -1;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtCreateStream failed, ret: " << ret << std::endl;
        return -1;
    }

    auto executor = CREATE_EXECUTOR();
    if (executor.get() == nullptr) {
        std::cerr << "CREATE_EXECUTOR failed" << std::endl;
        return -1;
    }

    auto self = executor->AllocTensor(op::Shape{1000}, DataType::DT_FLOAT);
    if (self == nullptr) {
        std::cerr << "AllocTensor failed" << std::endl;
        return -1;
    }

    uint64_t seed = 12345;
    uint64_t offset = 0;
    float from = 10.0f;
    float to = 20.0f;

    constexpr int32_t v3KernelMode = 0;
    auto result = l0op::StatelessRandomUniformV3(self, seed, offset, from, to, v3KernelMode, executor.get());
    if (result == nullptr) {
        std::cerr << "StatelessRandomUniformV3 failed" << std::endl;
        return -1;
    }

    uint64_t workspaceSize = executor->GetWorkspaceSize();

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "aclrtMalloc failed, ret: " << ret << std::endl;
            return -1;
        }
    }

    executor->UpdateTensorAddr(workspace, workspaceSize);
    executor->SetStream(stream);
    auto runRet = executor->Run();
    if (runRet != ACLNN_SUCCESS) {
        std::cerr << "Run failed: " << runRet << std::endl;
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtSynchronizeStream failed, ret: " << ret << std::endl;
        return -1;
    }

    std::cout << "StatelessRandomUniformV3 test passed!" << std::endl;

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    return 0;
}
