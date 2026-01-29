# aclnnInplaceUniformTensor

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/random/dsa_random_uniform)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

算子功能：生成[from, to)区间内离散均匀分布的随机数，并将其填充到selfRef张量中。(该接口的BOOL类型已废弃，如需使用，建议采用aclnnBernoulli或者aclnnInplaceBernoulli接口)

## 函数原型

每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnInplaceUniformTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceUniformTensor”接口执行计算。

- `aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(const aclTensor* selfRef, double from, double to, const aclTensor* seedTensor, const aclTensor* offsetTensor, uint64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnInplaceUniformTensor(void* workspace, uint64_t workspace_size, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnInplaceUniformTensorGetWorkspaceSize

- **参数说明**：
  
  - selfRef（aclTensor*，计算输入|计算输出）：输入输出tensor，Device侧的aclTensor。支持[非连续的Tensor](./common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT16、INT8、UINT8、DOUBLE。
  - from（double，计算输入）：Host侧DOUBLE类型，进行离散均匀分布取值的左边界。from的值需要在selfRef的数据类型取值范围内，from的取值需要小于等于to。数据类型支持DOUBLE。
  - to（double，计算输入）：Host侧DOUBLE类型，进行离散均匀分布取值的右边界。to的值需要在selfRef的数据类型取值范围内。数据类型支持DOUBLE。
  - seedTensor（aclTensor*，计算输入）：Device侧的aclTensor，shape为[1]，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据类型支持UINT64。设置随机数生成器的种子值。
  - offsetTensor（aclTensor*，计算输入）：Device侧的aclTensor，shape为[1]，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据类型支持UINT64。与标量offset的累加结果作为随机数算子的偏移量。
  - offset（uint64_t，计算输入）：Host侧的整型。作为offsetTensor的累加量。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的selfRef是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. selfRef的数据类型不在支持的范围之内。
                                        2. selfRef的shape超过8维。
                                        3. from大于to。
  ```

## aclnnInplaceUniformTensor

- **参数说明**：
  
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnInplaceUniformTensorGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_uniform.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> selfRefShape = {2, 2};
    std::vector<int64_t> seedShape = {1};
    std::vector<int64_t> offsetShape = {1};
    void* selfRefDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    void* seedDeviceAddr = nullptr;
    aclTensor* seed = nullptr;
    void* offsetDeviceAddr = nullptr;
    aclTensor* offset = nullptr;
    int64_t offset2 = 102;
    std::vector<float> selfRefHostData = {0, 0, 0, 0};
    std::vector<int64_t> seedHostData = {0};
    std::vector<int64_t> offsetHostData = {392};

    // 创建selfRef aclTensor
    ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建seed aclTensor
    ret = CreateAclTensor(seedHostData, seedShape, &seedDeviceAddr, aclDataType::ACL_INT64, &seed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_INT64, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double from = 2.4;
    double to = 4.4;
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnInplaceUniform第一段接口
    ret = aclnnInplaceUniformTensorGetWorkspaceSize(selfRef, from, to, seed, offset, offset2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceUniformTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnInplaceUniformTensor第二段接口
    ret = aclnnInplaceUniformTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceUniformTensor failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(selfRefShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(selfRef);
    aclDestroyTensor(seed);
    aclDestroyTensor(offset);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfRefDeviceAddr);
    aclrtFree(seedDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```