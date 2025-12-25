# aclnnFmodScalar&aclnnInplaceFmodScalar

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/mod)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：返回 self除以other的余数。

- 计算公式：

  对于入参self，和比较标量other，Fmod可以用如下数学公式表示：

  $$
  out_{i} = self_{i} - (other *\left \lfloor (self_{i}/other) \right \rfloor)
  $$

## 函数原型

- aclnnFmodScalar和aclnnInplaceFmodScalar实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnFmodScalar：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceFmodScalar：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnFmodScalarGetWorkspaceSize”或者”aclnnInplaceFmodScalarGetWorkspaceSize“接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFmodScalar”或者”aclnnInplaceFmodScalar“接口执行计算。

  - `aclnnStatus aclnnFmodScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnFmodScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceFmodScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnFmodScalarGetWorkspaceSize

- **参数说明：**

  * self(aclTensor*, 计算输入)：Device侧的aclTensor。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，数据维度不支持8维以上。
     * <term>昇腾910_95 AI处理器</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../../docs/context/TensorScalar互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
     * <term>Atlas 训练系列产品</term>：DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
  * other(aclScalar*, 计算输入)：Host侧的aclScalar，[数据格式](../../../docs/context/数据格式.md)支持ND。
     * <term>昇腾910_95 AI处理器</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与self的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../../docs/context/TensorScalar互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
     * <term>Atlas 训练系列产品</term>：DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
  * out(aclTensor \*, 计算输出)：Device侧的aclTensor，且数据类型需要是self与other推导之后可转换的数据类型。shape需要与self一致，[数据格式](../../../docs/context/数据格式.md)支持ND，数据维度不支持8维以上。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8
     * <term>Atlas 训练系列产品</term>：DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8
  * workspaceSize(uint64_t \*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、other或out是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. self和other的数据类型和维度不在支持的范围之内。
                                    2. self和other不满足数据类型推导规则。
                                    3. 推导出的数据类型和维度无法转换为指定输出out的类型和维度。
                                    4. self和out的shape不一致。
  ```

## aclnnFmodScalar

- **参数说明：**

  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnFmodScalarGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## aclnnInplaceFmodScalarGetWorkspaceSize

- **参数说明：**

  * selfRef(aclTensor \*，计算输入|计算输出)：输入输出tensor，Device侧的aclTensor。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，数据维度不支持8维以上。
     * <term>昇腾910_95 AI处理器</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../../docs/context/TensorScalar互推导关系.md)），且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/context/互转换关系.md)）。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/context/互转换关系.md)）。
     * <term>Atlas 训练系列产品</term>：DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/context/互转换关系.md)）。
  * other(aclScalar*,计算输入)：Host侧的aclScalar，[数据格式](../../../docs/context/数据格式.md)支持ND。
     * <term>昇腾910_95 AI处理器</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与selfRef的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](../../../docs/context/TensorScalar互推导关系.md)）。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与selfRef的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)）。
     * <term>Atlas 训练系列产品</term>：DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8，且数据类型与selfRef的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)）。
  * workspaceSize(uint64_t \*，出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的selfRef、other是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. selfRef和other的数据类型和维度不在支持的范围之内。
                                    2. selfRef和other不满足数据类型推导规则。
                                    3. 推导出的数据类型无法转换为selfRef的类型。
  ```

## aclnnInplaceFmodScalar

- **参数说明：**

  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceFmodScalarGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFmodScalar&aclnnInplaceFmodScalar默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

**aclnnFmodScalar示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fmod_scalar.h"

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
  // 1. （固定写法）device/stream初始化 参考acl API手册
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
  aclScalar* other = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float otherValue = 1.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnFmodScalar第一段接口
  ret = aclnnFmodScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFmodScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnFmodScalar第二段接口
  ret = aclnnFmodScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFmodScalar failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. 释放device资源
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

**aclnnInplaceFmodScalar示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fmod_scalar.h"

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
  // 1. （固定写法）device/stream初始化 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  float otherValue = 1.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceFmodScalar第一段接口
  ret = aclnnInplaceFmodScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFmodScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceFmodScalar第二段接口
  ret = aclnnInplaceFmodScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFmodScalar failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(other);

  // 7. 释放device资源
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

