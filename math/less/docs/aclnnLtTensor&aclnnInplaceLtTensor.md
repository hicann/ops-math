# aclnnLtTensor&aclnnInplaceLtTensor

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/less)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 算子功能：判断输入self中的每个元素是否小于输入other中的元素，返回一个Bool类型的Tensor。

- 计算公式：

  $$
  out_i = (self_i < other_i)  ?  [True] : [False]
  $$

## 函数原型

- aclnnLtTensor和aclnnInplaceLtTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnLtTensor：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceLtTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLtTensorGetWorkspaceSize”或者“aclnnInplaceLtTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLtTensor”或者“aclnnInplaceLtTensor”接口执行计算。

  * `aclnnStatus aclnnLtTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnLtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceLtTensorGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceLtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLtTensorGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型需要与other满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），shape需要与other满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，shape维度不高于8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、DOUBLE、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、DOUBLE、BOOL。
  - other(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型需要与self满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），shape需要与self的shape满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，shape维度不高于8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、DOUBLE、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、DOUBLE、BOOL。
  - out(aclTensor \*, 计算输出)：Device侧的aclTensor，数据类型需要是BOOL可转换的数据类型[互转换关系](../../../docs/zh/context/互转换关系.md), shape与self、other广播之后的shape（参见[broadcast关系](../../../docs/zh/context/broadcast关系.md)）一致，shape维度不高于8维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、DOUBLE、BOOL、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、DOUBLE、BOOL、COMPLEX64、COMPLEX128。
  - workspaceSize(uint64_t \*, 出参)：返回用户需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self，other或out是空指针。
  161002（ACLNN_ERR_PARAM_INVALID）：1. self、other或out的数据类型不在支持的范围之内。
                                    2. self、other或out的维度大于8。
                                    3. self和other的数据类型无法进行推导。
                                    4. self和other的shape无法进行broadcast。
                                    5. out的shape与broadcast后的shape不一致。
  ```

## aclnnLtTensor

- **参数说明：**

  * workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnLtTensorGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceLtTensorGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor*,计算输入|计算输出)：输入输出tensor，即公式中的self与out。Device侧的aclTensor，输入数据类型需要与other满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），shape需要与other满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且broadcast后的shape需要与selfRef的shape一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL、BFLOAT16。
  - other(aclTensor*,计算输入)：Device侧的aclTensor，数据类型需要与selfRef满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且broadcast后的shape需要与selfRef的shape一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL、BFLOAT16。
  - workspaceSize(uint64_t \*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的selfRef、other是空指针时。
  161002(ACLNN_ERR_PARAM_INVALID): 1. selfRef和other的数据类型不在支持的范围之内。
                                   2. selfRef和other的数据类型无法进行推导。
                                   3. selfRef和other的shape无法做broadcast。
                                   4. selfRef和other做broadcast后的shape不等于selfRef的shape。
                                   5. selfRef、other的维度大于8。
  ```

## aclnnInplaceLtTensor

- **参数说明：**

  * workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceLtTensorGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnLtTensor&aclnnInplaceLtTensor默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnLtTensor示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_tensor.h"

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


struct LtTensorData {
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {5, 5, 5, 5, 5, 5, 5, 5};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
};

int CreateInputAndOutputTensors(LtTensorData& data) {
  auto ret = 0;
  
  // 创建self aclTensor
  ret = CreateAclTensor(data.selfHostData, data.selfShape, &data.selfDeviceAddr, aclDataType::ACL_DOUBLE, &data.self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(data.otherHostData, data.otherShape, &data.otherDeviceAddr, aclDataType::ACL_DOUBLE, &data.other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(data.outHostData, data.outShape, &data.outDeviceAddr, aclDataType::ACL_DOUBLE, &data.out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  return ret;
}

int ExecuteLtTensorComputation(aclrtStream stream, LtTensorData& data) {
  auto ret = 0;
  aclOpExecutor* executor;
  
  // 调用aclnnLtTensor第一段接口
  ret = aclnnLtTensorGetWorkspaceSize(data.self, data.other, data.out, &data.workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  data.workspaceAddr = nullptr;
  if (data.workspaceSize > 0) {
    ret = aclrtMalloc(&data.workspaceAddr, data.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnLtTensor第二段接口
  ret = aclnnLtTensor(data.workspaceAddr, data.workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtTensor failed. ERROR: %d\n", ret); return ret);
  
  // 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  return ret;
}

int ProcessAndPrintResults(const LtTensorData& data) {
  auto ret = 0;
  auto size = GetShapeSize(data.outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), data.outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }
  return ret;
}

void ReleaseResources(LtTensorData& data) {
  // 释放aclTensor和aclScalar
  aclDestroyTensor(data.self);
  aclDestroyTensor(data.other);
  aclDestroyTensor(data.out);

  // 释放device资源
  aclrtFree(data.selfDeviceAddr);
  aclrtFree(data.otherDeviceAddr);
  aclrtFree(data.outDeviceAddr);
  if (data.workspaceSize > 0) {
    aclrtFree(data.workspaceAddr);
  }
}

int ExecuteLtTensorOperator(aclrtStream stream) {
  LtTensorData data;
  
  // 创建输入和输出张量
  auto ret = CreateInputAndOutputTensors(data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 执行LtTensor算子操作
  ret = ExecuteLtTensorComputation(stream, data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 处理并打印结果
  ret = ProcessAndPrintResults(data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 释放资源
  ReleaseResources(data);
  
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // 执行InplaceLtScalar操作
  ret = ExecuteLtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // 重置设备和终结ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**aclnnInplaceLtTensor示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_tensor.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
  if (!(cond)) {                     \
    return_expr;                     \
  }                                  \
 } while(0)

#define LOG_PRINT(message, ...)   \
 do {                             \
  printf(message, ##__VA_ARGS__); \
 } while(0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法,资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template<typename T>
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

int ExecuteInplaceLtTensorOperator(aclrtStream stream) {
  auto ret = 0;
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> otherHostData = {1, 1, 1, 1, 0, 0, 0, 0};

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT32, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceLtTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnInplaceLtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtTensor failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  aclDestroyTensor(self);
  aclDestroyTensor(other);

  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // 执行InplaceLtScalar操作
  ret = ExecuteInplaceLtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // 重置设备和终结ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
