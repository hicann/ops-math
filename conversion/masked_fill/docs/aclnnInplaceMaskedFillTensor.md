# aclnnInplaceMaskedFillTensor

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/conversion/masked_fill)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

算子功能：用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnInplaceMaskedFillTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceMaskedFillTensor”接口执行计算。

* `aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnInplaceMaskedFillTensor(void* workspace, uint64_t workspace_size, aclOpExecutor* executor, aclrtStream stream)`

## aclnnInplaceMaskedFillTensorGetWorkspaceSize

- **参数说明：**

  - selfRef（aclTensor \*，计算输入|计算输出）：输入输出tensor，Device侧的aclTensor。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持BOOL、INT8、INT32、INT64、FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持BOOL、INT8、INT32、INT64、FLOAT、FLOAT16、BFLOAT16。
  - mask（aclTensor*，计算输入）：Device侧的aclTensor，数据类型支持BOOL。且shape与selfRef满足[broadcast关系](../../../docs/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - value（aclTensor*，计算输入）: Device侧的aclTensor，仅支持0维，数据类型需要与selfRef的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)）。[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持BOOL、INT8、INT32、INT64、FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持BOOL、INT8、INT32、INT64、FLOAT、FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ````
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的selfRef、mask、value是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef和mask的数据类型和数据格式不在支持的范围之内。
                                        2. selfRef和mask的shape无法做broadcast。
                                        3. value的数据类型不能转为selfRef的数据类型。
  ````

## aclnnInplaceMaskedFillTensor

- **参数说明：**

  * workspace（void \*，入参）：在Device侧申请的workspace内存地址。
  * workspace_size（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnInplaceMaskedFillTensorGetWorkspaceSize获取。
  * executor（aclOpExecutor \*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnInplaceMaskedFillTensor默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_masked_fill_tensor.h"

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
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> maskShape = {4, 2};
  std::vector<int64_t> valueShape = {};
  void* selfRefDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* mask = nullptr;
  aclTensor* value = nullptr;
  std::vector<float> selfHostData = {1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1};
  std::vector<char> maskHostData = {0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<float> valueHostData = {3.3};

  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mask aclTensor
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceMaskedFillTensor第一段接口
  ret = aclnnInplaceMaskedFillTensorGetWorkspaceSize(selfRef, mask, value, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMaskedFillTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceMaskedFillTensor第二段接口
  ret = aclnnInplaceMaskedFillTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMaskedFillTensor failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(mask);
  aclDestroyTensor(value);

  // 7. 释放device 资源
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(maskDeviceAddr);
  aclrtFree(valueDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
