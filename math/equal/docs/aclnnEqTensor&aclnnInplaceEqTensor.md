# aclnnEqTensor&aclnnInplaceEqTensor

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/equal)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：计算两个Tensor中的元素是否相等，返回一个Tensor，self=other的为True(1.)，否则为False(0.)。
- 计算表达式：

  $$
  out = (self_i == other_i)  ?  [True] : [False]
  $$

## 函数原型

- aclnnEqTensor和aclnnInplaceEqTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnEqTensor：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceEqTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnEqTensorGetWorkspaceSize”或者“aclnnInplaceEqTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEqTensor”或者“aclnnInplaceEqTensor”接口执行计算aclnnEqTensor。

  * `aclnnStatus aclnnEqTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnEqTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceEqTensorGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceEqTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnEqTensorGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型需要与other满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），shape需要与other满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，shape维度不高于8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128。
  - other(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型需要与self满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），shape需要与self的shape满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，shape维度不高于8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128。
  - out(aclTensor \*, 计算输出)：Device侧的aclTensor，数据类型需要是BOOL可转换的数据类型[互转换关系](../../../docs/zh/context/互转换关系.md), shape与self、other广播之后的shape（参见[broadcast关系](../../../docs/zh/context/broadcast关系.md)）一致，shape维度不高于8维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64、UINT32、UINT16。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
  * workspaceSize(uint64_t \*, 出参): 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、other、out是空指针时。
  161002(ACLNN_ERR_PARAM_INVALID): 1. self、other或out的数据类型不在支持的范围之内。
                                   2. self、other或out的维度大于8。
                                   3. self和other的数据类型无法进行推导。
                                   4. self和other的shape无法进行broadcast。
                                   5. out的shape与broadcast后的shape不一致。
  ```

## aclnnEqTensor

- **参数说明：**

  * workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnEqTensorGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceEqTensorGetWorkspaceSize

- **参数说明：**

  * selfRef（aclTensor*, 计算输入/输出）：输入输出tensor，即公式中的self与out。数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](./../../../docs/zh/context/互推导关系.md)），shape需要与other满足[broadcast关系](./../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](./../../../docs/zh/context/非连续的Tensor.md)，[数据格式](./../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128。
  * other（aclTensor*, 计算输入）：公式中的输入`other`，Device侧的aclTensor。数据类型与selfRef的数据类型需满足数据类型推导规则（参见[互推导关系](./../../../docs/zh/context/互推导关系.md)），shape需要与selfRef满足[broadcast关系](./../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](./../../../docs/zh/context/非连续的Tensor.md)，[数据格式](./../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128。
  * workspaceSize（uint64_t \*，出参）: 返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor \*\*，出参）: 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的selfRef和other是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef和other的数据类型不在支持的范围之内。
                                        2. selfRef和other的数据类型无法进行推导。
                                        3. selfRef和other的shape无法做broadcast。
                                        4. selfRef和other做broadcast后的shape不等于selfRef的shape。
                                        5. selfRef、other的维度大于8。
                                        6. selfRef和other有且仅有一个为空Tensor  
  ```

## aclnnInplaceEqTensor

- **参数说明：**

  * workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceEqTensorGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnEqTensor&aclnnInplaceEqTensor默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnEqTensor示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_tensor.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,
    void** selfDeviceAddr, void** otherDeviceAddr, void** outDeviceAddr, aclTensor** self, aclTensor** other,
    aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {0, 1, 9, 9, 9, 9, 9, 9};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(otherHostData, otherShape, otherDeviceAddr, aclDataType::ACL_DOUBLE, other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* other, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnEqTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnEqTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqTensor failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(char),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(
      selfShape, otherShape, outShape, &selfDeviceAddr, &otherDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 释放资源
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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

**aclnnInplaceEqTensor示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_tensor.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, void** selfDeviceAddr, void** otherDeviceAddr,
    aclTensor** self, aclTensor** other)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {1, 1, 3, 3, 5, 5, 7, 7};

  // 创建 self
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 other
  ret = CreateAclTensor(otherHostData, otherShape, otherDeviceAddr, aclDataType::ACL_DOUBLE, other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* other, void** workspaceAddrOut, uint64_t& workspaceSize, void* selfDeviceAddr,
    std::vector<int64_t>& selfShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // 获取 workspace 大小
  auto ret = aclnnInplaceEqTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 分配 workspace
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // 执行算子
  ret = aclnnInplaceEqTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqTensor failed. ERROR: %d\n", ret); return ret);

  // 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝输出
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(double),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;

  ret = CreateInputs(selfShape, otherShape, &selfDeviceAddr, &otherDeviceAddr, &self, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, &workspaceAddr, workspaceSize, selfDeviceAddr, selfShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 销毁
  aclDestroyTensor(self);
  aclDestroyTensor(other);

  // 释放
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```