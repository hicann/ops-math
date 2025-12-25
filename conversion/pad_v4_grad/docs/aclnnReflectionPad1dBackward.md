# aclnnReflectionPad1dBackward

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/conversion/pad_v4_grad)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：reflection_pad1d的反向传播，前向计算参考[[aclnnReflectionPad1d](aclnnReflectionPad1d.md)]。
- 示例：

  ```
  输入gradOutput([[1, 1, 1, 1, 1]])
  self([[0, 1, 2]])
  padding([1, 1])
  输出为([[1, 3, 1]])
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnReflectionPad1dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReflectionPad1dBackward”接口执行计算。

- `aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnReflectionPad1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnReflectionPad1dBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput(aclTensor*, 计算输入): 输入的梯度，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 数据类型与self一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，shape支持2-3维且维度需要与self和gradInput一致，shape需要与reflection_pad1d正向传播的output一致。
  - self(aclTensor*, 计算输入)：需要进行填充的tensor，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，shape支持2-3维且维度需要与gradOutput和gradInput一致，shape与gradInput一致。
  - padding(aclIntArray*, 计算输入)：填充范围，Device侧的aclIntArray数组，数据类型为INT64，长度为2。padding的两个数值都需小于self最后一维度的数值。
  - gradInput(aclTensor*, 计算输出)：计算得到的self的梯度，数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 数据类型与self一致，shape与self一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. gradOutput, self, padding, gradInput任何一个为空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. gradOutput、self、padding和gradInput的数据类型或数据格式不在支持的范围之内。
  				     2. gradOutput、self、padding和gradInput的输入shape在支持范围之外。
  				     3. self为空tensor且self后两个维度的值存在0。
  				     4. padding内的数值大于等于self的维度。
  				     5. gradOutput shape需要与reflection_pad1d正向传播的output一致。
  ```

## aclnnReflectionPad1dBackward

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnReflectionPad1dBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnReflectionPad1dBackward默认确定性实现。

当gradOutput中元素个数大于300\*1024\*1024有运行超时风险。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_reflection_pad1d_backward.h"
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
    std::vector<int64_t> gradOutputShape = {1, 4};
    std::vector<int64_t> selfShape = {1, 2};
    std::vector<int64_t> gradInputShape = {1, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutputHostData = {1, 1, 1, 1};
    std::vector<float> selfHostData = {1, 2};
    std::vector<int64_t> paddingData = {1, 1};
    std::vector<float> gradInputHostData = {0, 0};
    // 创建gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建padding aclIntArray
    padding = aclCreateIntArray(paddingData.data(), 2);
    CHECK_RET(padding != nullptr, return ret);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnReflectionPad1dBackward第一段接口
    ret = aclnnReflectionPad1dBackwardGetWorkspaceSize(gradOutput, self, padding, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReflectionPad1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnReflectionPad1dBackward第二段接口
    ret = aclnnReflectionPad1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReflectionPad1dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(self);
    aclDestroyIntArray(padding);
    aclDestroyTensor(gradInput);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
