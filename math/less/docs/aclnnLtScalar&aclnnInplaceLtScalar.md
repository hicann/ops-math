# aclnnLtScalar&aclnnInplaceLtScalar

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/less)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |




## 功能说明

- 算子功能：判断输入self中的每个元素是否小于输入other的值，返回一个Bool类型的Tensor。

- 计算公式：

  $$
  out_i = (self_i < other_i)  ?  [True] : [False]
  $$

## 函数原型

- aclnnLtScalar和aclnnInplaceLtScalar实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnLtScalar：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceLtScalar：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](./../../../docs/context/两段式接口.md)，必须先调用“aclnnLtScalarGetWorkspaceSize”或者“aclnnInplaceLtScalarGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLtScalar”或者“aclnnInplaceLtScalar"接口执行计算。

  - `aclnnStatus aclnnLtScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnLtScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceLtScalarGetWorkspaceSize(const aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceLtScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLtScalarGetWorkspaceSize

- **参数说明：**

  * self(aclTensor*, 计算输入)：Device侧的aclTensor，shape维度不高于8维，支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)，[数据格式](./../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL，且与other满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL，且与other满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL，且与other满足[TensorScalar互推导关系](./../../../docs/context/TensorScalar互推导关系.md)。
  * other(aclScalar*, 计算输入)：Host侧的aclScalar。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL，且与self满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL，且与self满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL，且与self满足[TensorScalar互推导关系](./../../../docs/context/TensorScalar互推导关系.md)。
  * out(aclTensor \*, 计算输出)：Device侧的aclTensor，shape与self一致，数据类型要求为BOOL可转换的数据类型[互转换关系](./../../../docs/context/互转换关系.md)，支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)，[数据格式](./../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128。
  * workspaceSize(uint64_t \*, 出参)：返回用户需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、other、out是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. self，other或out的数据类型不在支持的范围之内。
                                    2. self和other数据类型不满足数据类型推导规则。
                                    3. self和out的shape不同。
                                    4. self和out的维度大于8。
  ```

## aclnnLtScalar

- **参数说明：**

  * workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnLtScalarGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

## aclnnInplaceLtScalarGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor*,计算输入|计算输出)：输入输出tensor，即公式中的self与out。Device侧的aclTensor，shape维度不高于8维，支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)，[数据格式](./../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL，且与other满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16，且与other满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16，且与other满足[TensorScalar互推导关系](./../../../docs/context/TensorScalar互推导关系.md)。
  - other(aclScalar*,计算输入)：Host侧的aclScalar。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL，且与selfRef满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16，且与selfRef满足[互推导关系](./../../../docs/context/互推导关系.md)。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16，且与selfRef满足[TensorScalar互推导关系](./../../../docs/context/TensorScalar互推导关系.md)。
  - workspaceSize(uint64_t \*, 出参)：返回用户需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的selfRef、other是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. selfRef和other的数据类型不在支持的范围之内。
                                   2. selfRef和other的数据类型不满足数据类型推导规则。
                                   3. selfRef的维度大于8。
  ```

## aclnnInplaceLtScalar

- **参数说明：**

  * workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceLtScalarGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnLtScalar&aclnnInplaceLtScalar默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./../../../docs/context/编译与运行样例.md)。

**aclnnLtScalar示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_scalar.h"

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

int ExecuteLtScalarOperator(aclrtStream stream) {
  auto ret = 0;

  //构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  double otherValue = 1.2;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(other != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnLtScalar第一段接口
  ret = aclnnLtScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnLtScalar第二段接口
  ret = aclnnLtScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtScalar failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
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
  ret = ExecuteLtScalarOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // 重置设备和终结ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**aclnnInplaceLtScalar示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_scalar.h"

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

int ExecuteInplaceLtScalarOperator(aclrtStream stream) {
  auto ret = 0;
  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 1.2, 0.3, 4.1, 5, 1.6, 7};
  double otherValue = 1.2;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(other != nullptr, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceLtScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnInplaceLtScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtScalar failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyScalar(other);

  aclrtFree(selfDeviceAddr);
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
  ret = ExecuteInplaceLtScalarOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // 重置设备和终结ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```