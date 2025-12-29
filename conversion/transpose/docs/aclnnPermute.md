# aclnnPermute

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/transpose_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

算子功能：对tensor的任意维度进行调换。如输入self是shape为[2, 3, 5]的tensor，dims为(2, 0, 1)，则输出是shape为[5, 2, 3]的tensor。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnPermuteGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPermute”接口执行计算。

- `aclnnStatus aclnnPermuteGetWorkspaceSize(const aclTensor* self, const aclIntArray* dims, aclTensor* out,uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnPermuteGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*,计算输入)：输入的tensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，维度最大不超过8维，dtype需要与out一致。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN。

  - dims(aclIntArray*,计算输入)：整型数组，代表原来tensor的维度，指定新的轴顺序。取值需在[-self的维度数量，self的维度数量-1]范围内。

  - out(aclTensor*，计算输出)：输出的tensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维度最大不超过8维，shape由dims和原self的shape共同决定，dtype需要与self一致。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN。

  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、dims或out是空指针。
161002 (ACLNN_ERR_PARAM_INVALID): 1. self和out的数据类型不在支持的范围之内。
                                  2. self和out的数据类型不一致。
                                  3. 输入self的维度超过8维。
                                  4. dims的取值不在[-self的维度数量，self的维度数量-1]的范围内。
```

## aclnnPermute

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnPermuteGetWorkspaceSize获取。

  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnPermute默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_permute.h"

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
  std::vector<int64_t> dimsData = {1, 0};
  std::vector<int64_t> outShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建dims aclIntArray
  aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnPermute第一段接口
  ret = aclnnPermuteGetWorkspaceSize(self, dims, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnPermute第二段接口
  ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermute failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(dims);
  aclDestroyTensor(out);

  // 7. 释放device 资源
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

