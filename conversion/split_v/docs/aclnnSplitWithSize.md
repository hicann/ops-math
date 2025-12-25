# aclnnSplitWithSize

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/conversion/split_v)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

| <term>Atlas 推理系列产品 </term>                             |    ×      |



## 功能说明

算子功能：将输入self沿dim轴切分至splitSize中每个元素的大小。

## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnSplitWithSizeGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSplitWithSize”接口执行计算。

- `aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor *self, const aclIntArray *splitSize, int64_t dim, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSplitWithSize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSplitWithSizeGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*,计算输入)：表示被split的输入tensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，当输出个数大于32时，不支持DOUBLE、COMPLEX128、COMPLEX64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，当输出个数大于32时，不支持DOUBLE、COMPLEX128、COMPLEX64。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16。

  - splitSize(aclIntArray*,计算输入)：表示需要split的各块大小，数据类型支持INT64和INT32。所有块的大小总和需要等于self在dim维度上的shape大小。

  - dim(int64_t,计算输入)：数据类型支持INT64，表示输入tensor被split的维度。

  - out(aclTensorList*，计算输出)：表示被split后的输出tensor的列表，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64，当输出个数大于32时，不支持DOUBLE、COMPLEX128、COMPLEX64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16，当输出个数大于32时，不支持DOUBLE、COMPLEX128、COMPLEX64。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

​   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、splitSize、out是空指针时。
161002 (ACLNN_ERR_PARAM_INVALID): 1. self和out的数据类型不在支持的范围之内。
                                  2. self的长度不在支持的范围之内。
                                  3. out中的tensor长度不在支持的范围之内时。
                                  4. dim的取值越界不在[-dimNum, dimNum -1],dimNum为self的维度大小。
                                  5. splitSize中各元素之和不等于被split维度的shape大小时。
```

## aclnnSplitWithSize

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnSplitWithSizeGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSplitWithSize默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```c++
#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_split_with_size.h"

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

void CheckResult(const std::vector<std::vector<int64_t>> &shapeList, const std::vector<void *> addrList) {
  for (size_t i = 0; i < shapeList.size(); i++) {
    auto size = GetShapeSize(shapeList[i]);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), addrList[i],
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return);
    for (int64_t j = 0; j < size; j++) {
      LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
    }
  }
}

int main() {
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {5, 2};
  std::vector<int64_t> shape1 = {1, 2};
  std::vector<int64_t> shape2 = {4, 2};
  int64_t splitValue[] = {1, 4};
  int64_t dim = 0;

  void* selfDeviceAddr = nullptr;
  void* shape1DeviceAddr = nullptr;
  void* shape2DeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* shape1Addr = nullptr;
  aclTensor* shape2Addr = nullptr;
  aclIntArray *splitSize = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> shape1HostData = {0, 5};
  std::vector<float> shape2HostData = {1, 2, 3, 4, 6, 7, 8, 9};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  splitSize = aclCreateIntArray(splitValue, 2);
  CHECK_RET(splitSize != nullptr, return ret);

  ret = CreateAclTensor(shape1HostData, shape1, &shape1DeviceAddr, aclDataType::ACL_FLOAT, &shape1Addr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(shape2HostData, shape2, &shape2DeviceAddr, aclDataType::ACL_FLOAT, &shape2Addr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensorList
  std::vector<aclTensor*> tmp = {shape1Addr, shape2Addr};
  aclTensorList* out = aclCreateTensorList(tmp.data(), tmp.size());
  CHECK_RET(out != nullptr, return ret);

  // 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  // 调用aclnnSplitWithSize第一段接口
  ret = aclnnSplitWithSizeGetWorkspaceSize(self, splitSize, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSplitWithSizeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSplitWithSize第二段接口
  ret = aclnnSplitWithSize(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSplitWithSize failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CheckResult({shape1, shape2}, {shape1DeviceAddr, shape2DeviceAddr});

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(splitSize);
  aclDestroyTensorList(out);
  aclDestroyTensor(shape1Addr);
  aclDestroyTensor(shape2Addr);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(shape1DeviceAddr);
  aclrtFree(shape2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

