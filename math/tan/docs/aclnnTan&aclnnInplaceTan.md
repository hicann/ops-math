# aclnnTan&aclnnInplaceTan

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/tan)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |     ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品 </term>                             |   √     |
| <term>Atlas 训练系列产品</term>                              |   √     |


## 功能说明

- 算子功能：它计算输入张量`self`中每个元素的正切值，并将结果存储在输出张量`out`中。

- 计算公式：
  $$
  out[i] = tan(self[i])
  $$

- 示例：

  ```
  输入self：
  tensor([[1, 2],
          [3, 4]])
  输出out：
  tensor([[ 1.5574, -2.1850],
          [-0.1425,  1.1578]])
  ```

## 函数原型

- aclnnTan和aclnnInplaceTan实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnTan：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceTan：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。 

- 每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnTanGetWorkspaceSize”或者“aclnnInplaceTanGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnTan”或者“aclnnInplaceTan”接口执行计算。

  - `aclnnStatus aclnnTanGetWorkspaceSize(const aclTensor *self, aclTensor *out,uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnTan(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceTanGetWorkspaceSize(const aclTensor   *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceTan(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnTanGetWorkspaceSize

- **参数说明**：

  * self（aclTensor*, 计算输入）：公式中的输入`self`，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，shape维度不大于8, 且shape需要与out一致，和out的数据类型满足数据类型推导规则。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16。
  * out（aclTensor*, 计算输出）：公式中的`out`，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，且shape需要与self一致，和self的数据类型满足数据类型推导规则。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  * workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。
  
- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self，out是空指针时。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self的数据类型不在支持的范围之内，self不能转为out的数据类型。
                                        2. self和out的数据类型不满足数据类型推导规则。
                                        3. Self和Out的维度大于8。
                                        4. Self和Out的shape不一致。
  ```

## aclnnTan

- **参数说明**：

  * workspace（void*, 入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnTanGetWorkspaceSize获取。
  * executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## aclnnInplaceTanGetWorkspaceSize

- **参数说明**：

  - selfRef(aclTensor*, 计算输入|计算输出): 公式中的`self`，Device侧的aclTensor，支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)，[数据格式](./../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的 selfRef是空指针时。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. selfRef 的数据类型不在支持的范围之内。
  ```

## aclnnInplaceTan

- **参数说明**：

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnInplaceTanGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnTan&aclnnInplaceTan默认确定性实现。

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、INT32、INT64数据类型的输入数据范围为[-65504,65504]时满足精度要求，超过数值范围无法保证，请使用CPU进行计算。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、BFLOAT16、FLOAT16、INT32、INT64数据类型的输入数据范围为[-65504,65504]时满足精度要求，超过数值范围无法保证，请使用CPU进行计算。如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），场景为最后2轴合轴小于16，前面的轴合轴超大。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

**aclnnTan接口调用示例：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_tan.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnTan第一段接口
  ret = aclnnTanGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnTan第二段接口
  ret = aclnnTan(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTan failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
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

**aclnnInplaceTan接口调用示例：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_tan.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfRefShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  std::vector<float> selfRefHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  // 创建self aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceTan第一段接口
  ret = aclnnInplaceTanGetWorkspaceSize(selfRef, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceTan第二段接口
  ret = aclnnInplaceTan(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTan failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
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

  // 6. 释放aclTensor，需要根据具体API的接口定义
  aclDestroyTensor(selfRef);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfRefDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

