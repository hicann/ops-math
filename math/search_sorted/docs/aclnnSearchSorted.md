# aclnnSearchSorted

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/search_sorted)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

算子功能：在一个已排序的张量（sortedSequence）中查找给定tensor值（self）应该插入的位置。返回与self相同大小的张量，其中每个元素表示给定值在原始张量中应该插入的位置。如果self为scalar类型，请参考文档[aclnnSearchSorteds](./aclnnSearchSorteds.md)

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnSearchSortedGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSearchSorted”接口执行计算。

- `aclnnStatus aclnnSearchSortedGetWorkspaceSize(const aclTensor *sortedSequence, const aclTensor *self, const bool outInt32, const bool right, const aclTensor *sorter, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSearchSorted(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSearchSortedGetWorkspaceSize

- **参数说明**：
  
  - sortedSequence（aclTensor*,计算输入）：Device侧的aclTensor，为已排序的张量，shape不超过8维，数据类型支持DOUBLE、FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - self（aclTensor*,计算输入）：Device侧的aclTensor，存储的是要插入的值，数据类型支持DOUBLE、FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64，且数据类型与sortedSequence的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。shape不超过8维。
  - outInt32（bool,计算输入）：Host侧的布尔型，表示指定输出Tensor是否为INT32类型。
  - right（bool,计算输入）：Host侧的布尔型，表示如果找到相同的值，是否返回右侧的位置。如果为False，则返回左侧的位置。
  - sorter（aclTensor*,计算输入）：Device侧的aclTensor，指定sortedSequence中元素顺序，取值范围为零到最内层维度的dim值减一，数据类型为INT64，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。shape与sortedSequence一致。
  - out（aclTensor*,计算输出）: Device侧的aclTensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，数据类型支持INT32、INT64，[数据格式](../../../docs/context/数据格式.md)支持ND。shape与self一致。
  - workspaceSize（uint64_t*,出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**,出参）：返回op执行器，包含了算子计算流程。
  
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的 sortedSequence、self、out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. sortedSequence、self 的数据类型不在支持的范围之内。
                                        2. out的数据类型与outInt32值含义相违背。
                                        3. sortedSequence与self数据类型不同时，不能做数据类型推导。
                                        4. 传入的sorter不是INT64类型。
                                        5. out的shape与self的shape不相同，sorter的shape与sortedSequence的shape不相同。
                                        6. 当sortedSequence维度大于一，self除最后一维外，其他维度不与sortedSequence对应维度不相等时。
  ```

## aclnnSearchSorted

- **参数说明**：
  
  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnSearchSortedGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSearchSorted默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_searchsorted.h"

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
  std::vector<int64_t> sortedSequenceShape = {4, 4};
  std::vector<int64_t> sorterShape = {4, 4};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* sortedSequenceDeviceAddr = nullptr;
  void* sorterDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* sortedSequence = nullptr;
  aclTensor* sorter = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1,3,6,8,10,11,14,16};
  std::vector<float> sortedSequenceHostData = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  std::vector<int64_t> sorterHostData = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sortedSequence aclTensor
  ret = CreateAclTensor(sortedSequenceHostData, sortedSequenceShape, &sortedSequenceDeviceAddr, aclDataType::ACL_FLOAT, &sortedSequence);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sorter aclTensor
  ret = CreateAclTensor(sorterHostData, sorterShape, &sorterDeviceAddr, aclDataType::ACL_INT64, &sorter);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  bool outInt32 = false;
  bool right = false;
  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSearchSorted第一段接口
  ret = aclnnSearchSortedGetWorkspaceSize(sortedSequence, self, outInt32, right, sorter, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSortedGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnSearchSorted第二段接口
  ret = aclnnSearchSorted(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSorted failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int64_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(sortedSequence);
  aclDestroyTensor(sorter);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(sortedSequenceDeviceAddr);
  aclrtFree(sorterDeviceAddr);
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
