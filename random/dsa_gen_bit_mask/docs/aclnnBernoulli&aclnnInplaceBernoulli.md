# aclnnBernoulli&aclnnInplaceBernoulli

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：从伯努利分布中提取二进制随机数（0 或 1），prob为生成二进制随机数的概率，输入的张量用于指定shape。

- 计算公式：

  $$
  out∼Bernoulli(prob)
  $$

   其中，当使用aclnnBernoulli时，公式中的prob对应第一段接口中的prob，公式中的out对应第一段接口中的out；当使用aclnnInplaceBernoulli时，公式中的prob对应第一段接口中的prob，公式中的out对应第一段接口中的selfRef。

## 函数原型

  - aclnnBernoulli和aclnnInplaceBernoulli实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
    - aclnnBernoulli：需新建一个输出张量对象存储计算结果。
    - aclnnInplaceBernoulli：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
  - 每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnBernoulliGetWorkspaceSize”或者“aclnnInplaceBernoulliGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBernoulli”或者“aclnnInplaceBernoulli”接口执行计算。

    - `aclnnStatus aclnnBernoulliGetWorkspaceSize(const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
    - `aclnnStatus aclnnBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`
    - `aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(const aclTensor* selfRef, const aclScalar* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)`
    - `aclnnStatus aclnnInplaceBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBernoulliGetWorkspaceSize

  - **参数说明：**
    - self（aclTensor*，计算输入）：用于指定输出out的shape，Device侧的aclTensor，数据类型需要与out一致，shape支持0-8维，且shape需要与out一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
    - prob（aclScalar*，计算输入）：公式中的prob，Host侧的aclScalar，满足$ 0≤prob≤1 $。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16。
    - seed（int64_t，计算输入）：Host侧的整型，设置随机数生成器的种子。
    - offset（int64_t，计算输入）：Host侧的整型，设置随机数偏移量。
    - out（aclTensor*，计算输出）：公式中的out，Device侧的aclTensor，数据类型需要与self一致，shape支持0-8维，且shape需要与self一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
    - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现如下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、prob或out是空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：1. self或out的数据类型和数据格式不在支持的范围之内。
                                          2. prob的数据类型不在支持的范围之内。
                                          3. self和out的数据类型不一致。
                                          4. prob不满足0≤prob≤1。
                                          5. self或out的维度大于8维。
                                          6. self和out的shape不一致。
    ```

## aclnnBernoulli

- **参数说明：**
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnBernoulliGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## aclnnInplaceBernoulliGetWorkspaceSize

  - **参数说明：**
    - selfRef（aclTensor*，计算输入/输出）：公式中的out，Device侧的aclTensor，shape支持0-8维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
    - prob（aclScalar*，计算输入）：公式中的prob，Host侧的aclScalar，满足$ 0≤prob≤1 $。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16。
    - seed（int64_t，计算输入）：Host侧的整型，设置随机数偏移量。
    - offset（int64_t，计算输入）：Host侧的整型，设置随机数偏移量。
    - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现如下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的selfRef或prob是空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef的数据类型和数据格式不在支持的范围之内。
                                          2. prob的数据类型不在支持的范围之内。
                                          3. prob不满足0≤prob≤1。
                                          4. selfRef的维度大于8维。
    ```

## aclnnInplaceBernoulli

  - **参数说明：**
    - workspace（void*，入参）：在Device侧申请的workspace内存地址。
    - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnInplaceBernoulliGetWorkspaceSize获取。
    - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream，入参）：指定执行任务的Stream。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnBernoulli&aclnnInplaceBernoulli默认确定性实现。


## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

aclnnBernoulli

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bernoulli.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* prob = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData(16, 0);
  int64_t seed = 0;
  int64_t offset = 0;
  float probValue = 1.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建prob aclScalar
  prob = aclCreateScalar(&probValue,aclDataType::ACL_FLOAT);
  CHECK_RET(prob != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBernoulli接口调用示例
  // 3. 调用CANN算子库API
  // 调用aclnnBernoulli第一段接口
  ret = aclnnBernoulliGetWorkspaceSize(self, prob, seed, offset, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBernoulliGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBernoulli第二段接口
  ret = aclnnBernoulli(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBernoulli failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(prob);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

aclnnInplaceBernoulli

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bernoulli.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 4};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* prob = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int64_t seed = 0;
  int64_t offset = 0;
  float probValue = 1.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建prob aclScalar
  prob = aclCreateScalar(&probValue,aclDataType::ACL_FLOAT);
  CHECK_RET(prob != nullptr, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnInplaceBernoulli接口调用示例
  // 3. 调用CANN算子库API
  // 调用aclnnInplaceBernoulli第一段接口
  ret = aclnnInplaceBernoulliGetWorkspaceSize(self, prob, seed, offset, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBernoulliGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceBernoulli第二段接口
  ret = aclnnInplaceBernoulli(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBernoulli failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(prob);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
