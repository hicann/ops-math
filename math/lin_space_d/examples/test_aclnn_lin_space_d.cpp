#include <iostream>
#include <vector>
#include <type_traits> 
#include "acl/acl.h"
#include "aclnn_lin_space_d.h"

#define CHECK_RET(cond, return_expr, message, ...) \
    do {                                           \
        if (!(cond)) {                             \
            LOG_PRINT("[ERROR] " message, ##__VA_ARGS__); \
            return return_expr;  \
        }                                          \
    } while (0)


#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout);                 \
    } while (0)

template <typename T>
aclDataType GetAclDataType() {
    if (std::is_same<T, int8_t>::value) {
        return ACL_INT8;
    } else if (std::is_same<T, uint8_t>::value) {
        return ACL_UINT8;
    } else if (std::is_same<T, int16_t>::value) {
        return ACL_INT16;
    } else if (std::is_same<T, aclFloat16>::value) {
        return ACL_FLOAT16;
    } else if (std::is_same<T, int32_t>::value) {
        return ACL_INT32;
    } else if (std::is_same<T, float>::value) {
        return ACL_FLOAT;
    } else if (std::is_same<T, int64_t>::value) {
        return ACL_INT64;
    } else if (std::is_same<T, double>::value) {
        return ACL_DOUBLE;
    } else {
        return ACL_BF16; 
    }
}

// 计算张量总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

// 初始化 AscendCL 环境
int Init(int32_t deviceId, aclrtStream* stream) {
    LOG_PRINT("[INFO] Start initializing AscendCL...\n");
    
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d\n", ret);
    
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d\n", ret);
    
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d\n", ret);
    
    LOG_PRINT("[INFO] AscendCL initialization success (deviceId=%d, stream=%p)\n", deviceId, *stream);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData,    
    const std::vector<int64_t>& shape, 
    void** deviceAddr,                 
    aclDataType dataType,              
    aclTensor** tensor                 
) {
    int64_t elementCount = shape.empty() ? 1 : GetShapeSize(shape);
    size_t memSize = elementCount * sizeof(T);
    const char* shapeDesc = shape.empty() ? "scalar(0D)" : std::to_string(shape[0]).c_str();
    LOG_PRINT("[INFO] Creating tensor: shape=%s, elementCount=%ld, memSize=%lu bytes\n",
              shapeDesc, elementCount, memSize);
    
    // 分配Device内存
    auto ret = aclrtMalloc(deviceAddr, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d, memSize=%lu\n", ret, memSize);
    
    // Host→Device拷贝
    ret = aclrtMemcpy(*deviceAddr, memSize, hostData.data(), memSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy (Host→Device) failed. ERROR: %d\n", ret);
    
    const int64_t* stridePtr = nullptr;
    std::vector<int64_t> strides;
    if (!shape.empty() && shape.size() > 1) {
        strides.resize(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        stridePtr = strides.data();
    } else {
        stridePtr = nullptr;  
    }
    
    // 创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(),                                  // 0维：viewDims为nullptr
        static_cast<uint64_t>(shape.size()),        // 0维：viewDimsNum=0
        dataType,
        stridePtr,                                       // 步长
        0,
        ACL_FORMAT_ND,
        shape.data(),                               // 0维：storageDims为nullptr
        static_cast<uint64_t>(shape.size()),     // 0维：storageDimsNum=0
        *deviceAddr
    );
    CHECK_RET(*tensor != nullptr, -1, "aclCreateTensor failed. tensor is nullptr\n");
    
    LOG_PRINT("[INFO] Tensor created success (deviceAddr=%p, tensor=%p)\n", *deviceAddr, *tensor);
    return 0;
}

int main() {
    // ===================== 初始化 AscendCL =====================
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, ret, "Init AscendCL failed. ERROR: %d\n", ret);

    // ===================== 构造输入参数 =====================
    float startValue = 0.0f;
    float endValue = 0.01f;
    const int64_t numValue = 8;

    // 自动推导数据类型
    const aclDataType startType = GetAclDataType<decltype(startValue)>();
    const aclDataType endType = GetAclDataType<decltype(endValue)>();
    const aclDataType outputType = ACL_FLOAT;
    CHECK_RET(startType != ACL_DT_UNDEFINED && endType != ACL_DT_UNDEFINED,
              -1, "Invalid data type (startType=%d, endType=%d)\n", startType, endType);
    const std::vector<int64_t> scalarShape = {1}; 
    const std::vector<int64_t> outShape = {numValue};

    // Host侧数据
    const std::vector<decltype(startValue)> startHostData = {startValue};
    const std::vector<decltype(endValue)> endHostData = {endValue};
    const std::vector<float> outHostData(numValue, 0.0f);

    // 分配独立Device地址
    void* startDeviceAddr = nullptr;
    void* endDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    // 创建张量
    aclTensor* startTensor = nullptr;
    ret = CreateAclTensor(startHostData, scalarShape, &startDeviceAddr, startType, &startTensor);
    CHECK_RET(ret == 0, ret, "Create startTensor failed. ERROR: %d\n", ret);

    aclTensor* endTensor = nullptr;
    ret = CreateAclTensor(endHostData, scalarShape, &endDeviceAddr, endType, &endTensor);
    CHECK_RET(ret == 0, ret, "Create endTensor failed. ERROR: %d\n", ret);

    aclTensor* outTensor = nullptr;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, outputType, &outTensor);
    CHECK_RET(ret == 0, ret, "Create outTensor failed. ERROR: %d\n", ret);

    // 创建steps数组
    const std::vector<int64_t> stepsVec = {numValue};
    aclIntArray* stepsArray = aclCreateIntArray(stepsVec.data(), stepsVec.size());
    CHECK_RET(stepsArray != nullptr, -1, "aclCreateIntArray failed. stepsArray is nullptr\n");
    LOG_PRINT("[DEBUG] stepsArray 创建成功，地址：%p\n", stepsArray);  // 打印 stepsArray 地址
    LOG_PRINT("  startTensor: %p\n", startTensor);    // 第1个参数：start 张量
    LOG_PRINT("  endTensor: %p\n", endTensor);        // 第2个参数：end 张量
    LOG_PRINT("  stepsArray: %p\n", stepsArray);      // 第3个参数：steps 数组
    LOG_PRINT("  outTensor: %p\n", outTensor);        // 第4个参数：out 张量
    // ===================== 调用算子 =====================
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnLinSpaceDGetWorkspaceSize(startTensor, endTensor, stepsArray, outTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("[ERROR] aclnnLinSpaceDGetWorkspaceSize failed: %s", errMsg ? errMsg : "nullptr");
    }
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclnnLinSpaceDGetWorkspaceSize failed. ERROR: %d\n", ret);
    LOG_PRINT("[INFO] Workspace size calculated: %lu bytes\n", workspaceSize);

    // 分配工作空间
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc workspace failed. ERROR: %d\n", ret);
        LOG_PRINT("[INFO] Workspace allocated: addr=%p, size=%lu bytes\n", workspaceAddr, workspaceSize);
    }

    // 执行算子
    ret = aclnnLinSpaceD(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclnnLinSpaceD failed. ERROR: %d\n", ret);
    LOG_PRINT("[INFO] Operator aclnnLinSpaceD executed success\n");

    // 同步流
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // ===================== 获取结果 =====================
    const int64_t outElementCount = GetShapeSize(outShape);
    std::vector<float> resultData(outElementCount, 0.0f);
    const size_t resultMemSize = outElementCount * sizeof(float);

    ret = aclrtMemcpy(
        resultData.data(), resultMemSize,
        outDeviceAddr, resultMemSize,
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    CHECK_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy (Device→Host) failed. ERROR: %d\n", ret);

    // 打印结果
    LOG_PRINT("\n[INFO] Operator output result (total %ld elements):\n", outElementCount);
    for (int64_t i = 0; i < outElementCount; ++i) {
        LOG_PRINT("result[%ld] = %.6f\n", i, resultData[i]);
    }

    // ===================== 释放资源 =====================
    LOG_PRINT("\n[INFO] Start releasing resources...\n");

    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);  
        LOG_PRINT("[INFO] aclOpExecutor destroyed\n");
    }

    if (stepsArray != nullptr) {
        aclDestroyIntArray(stepsArray);
        LOG_PRINT("[INFO] aclIntArray destroyed\n");
    }

    if (startTensor != nullptr) {
        aclDestroyTensor(startTensor);
        LOG_PRINT("[INFO] startTensor destroyed\n");
    }
    if (endTensor != nullptr) {
        aclDestroyTensor(endTensor);
        LOG_PRINT("[INFO] endTensor destroyed\n");
    }
    if (outTensor != nullptr) {
        aclDestroyTensor(outTensor);
        LOG_PRINT("[INFO] outTensor destroyed\n");
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        LOG_PRINT("[INFO] Workspace memory freed (addr=%p)\n", workspaceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
        LOG_PRINT("[INFO] outDeviceAddr freed (addr=%p)\n", outDeviceAddr);
    }
    if (endDeviceAddr != nullptr) {
        aclrtFree(endDeviceAddr);
        LOG_PRINT("[INFO] endDeviceAddr freed (addr=%p)\n", endDeviceAddr);
    }
    if (startDeviceAddr != nullptr) {
        aclrtFree(startDeviceAddr);
        LOG_PRINT("[INFO] startDeviceAddr freed (addr=%p)\n", startDeviceAddr);
    }

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        LOG_PRINT("[INFO] aclrtStream destroyed (stream=%p)\n", stream);
    }
    ret = aclrtResetDevice(deviceId);
    if (ret == ACL_SUCCESS) {
        LOG_PRINT("[INFO] Device %d reset success\n", deviceId);
    }
    ret = aclFinalize();
    if (ret == ACL_SUCCESS) {
        LOG_PRINT("[INFO] aclFinalize success\n");
    }

    LOG_PRINT("\n[INFO] All resources released. Program exit.\n");
    return 0;
}