# aclnnSubs&aclnnInplaceSubs

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/sub)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

+ 算子功能：完成减法计算，被减数按alpha进行缩放。
+ 计算公式：

  $$
  out_{i} = self_{i} - alpha \times other
  $$

  $$
  selfRef_{i} = selfRef_{i} - alpha \times other
  $$

## 函数原型

* aclnnSubs和aclnnInplaceSubs实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  * aclnnSubs：需新建一个输出张量对象存储计算结果。
  * aclnnInplaceSubs：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
* 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSubsGetWorkspaceSize”或者“aclnnInplaceSubsGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnSubs”或者“aclnnInplaceSubs”接口执行计算。
  * `aclnnStatus aclnnSubsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnSubs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceSubsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceSubs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSubsGetWorkspaceSize

* **参数说明**：
  * self（aclTensor*, 计算输入）：公式中的输入`self`，shape维度不高于8维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与other满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。

  * other（aclScalar*，计算输入）：公式中的输入`other`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与self满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与self满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且与self满足[互推导关系](../../../docs/zh/context/互推导关系.md)。

  * alpha（aclScalar*, 计算输入）：公式中的`alpha`，数据类型需要可转换成self与other推导后的数据类型。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64。

  * out(aclTensor*, 计算输出)：公式中的`out`，数据类型需要是self与other推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64。
SS
  * workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。

* **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的 self、other、alpha、out 是空指针时。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. self 的数据类型不在支持的范围之内。
                                        2. self 和 other 不满足数据类型推导规则。
                                        3. 推导出的数据类型无法转换为指定输出 out 的类型。
                                        4. alpha 无法转换为 self 和 other 推导后的数据类型。
                                        5. self 和 out 的 shape 不一致。
                                        6. self和out的维度大于8。
  ```

## aclnnSubs

* **参数说明**：
  + workspace（void*, 入参）：在Device侧申请的workspace内存地址。
  + workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnSubsGetWorkspaceSize获取。
  + executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。
  + stream（aclrtStream, 入参）：指定执行任务的Stream。

* **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceSubsGetWorkspaceSize

* **参数说明**：
  * selfRef（aclTensor*, 计算输入|计算输出）：公式中的输入`selfRef`，维度不超过8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与other满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)，且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)，且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)，且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。

  * other（aclScalar*，计算输入）：公式中的输入`other`，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与selfRef满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)，且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，且与selfRef满足[互推导关系](../../../docs/zh/context/互推导关系.md)，且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，且与selfRef满足[互推导关系](../../../docs/zh/context/互推导关系.md)，且需要是推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。

  * alpha（aclScalar*, 计算输入）：公式中的`alpha`，数据类型需要可转换成selfRef与other推导后的数据类型（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16。

  * workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。

* **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的selfRef、other、alpha是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. selfRef的数据类型不在支持的范围之内。
                                        2. selfRef和other不满足数据类型推导规则。
                                        3. 推导出的数据类型无法转换为selfRef的类型。
                                        4. alpha无法转换为selfRef和other推导后的数据类型。
  ```

## aclnnInplaceSubs

* **参数说明**：
  + workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  + workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceSubsGetWorkspaceSize获取。
  + executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  + stream(aclrtStream, 入参)：指定执行任务的Stream。

* **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSubs&aclnnInplaceSubs默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sub.h"

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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  float otherValue = 2.0f;
  float alphaValue = 1.2f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnSubs接口调用示例
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSubs第一段接口
  ret = aclnnSubsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSubs第二段接口
  ret = aclnnSubs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubs failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceSubs接口调用示例
  // 3. 调用CANN算子库API
  LOG_PRINT("\ntest aclnnInplaceSubs\n");
  // 调用aclnnInplaceSubs第一段接口
  ret = aclnnInplaceSubsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceSubs第二段接口
  ret = aclnnInplaceSubs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSubs failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 将device侧内存上的结果拷贝到host侧
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("%ld acos(%f) = %f\n", i, selfHostData[i], resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyScalar(alpha);
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
