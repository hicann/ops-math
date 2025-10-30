# aclnnBincount

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：计算非负整数数组中每个数的频率。minlength为输出tensor的最小size；当weights为空指针时，self中的self[i]每出现一次，则其频率加1，当weights不为空时，self[i]每出现一次，其频率加weights[i]，最后存放到out的self[i]+1位置上；因此out大小为(self的最大值+1)与minlength中的最大值。

- 计算公式：

  如果n是self在位置i上的值，如果指定了weights，则
  
  $$
  out[self_i] = out[self_i] + weights_i
  $$
  
  否则：
  
  $$
  out[self_i] = out[self_i] + 1
  $$

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnBincountGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBincount”接口执行计算。

* `aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor* self, const aclTensor* weights, int64_t minlength,aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBincountGetWorkspaceSize

- **参数说明：**

  * self(aclTensor*,计算输入)：Device侧的aclTensor，数据类型支持INT8、INT16、INT32、INT64、UINT8，且必须是非负整数，[数据格式](common/数据格式.md)支持1维ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
  * weights(aclTensor*,计算输入)：Device侧的aclTensor，self每个值的权重，可为空指针。数据类型支持FLOAT、FLOAT16、FLOAT64、INT8、INT16、INT32、INT64、UINT8、BOOL，[数据格式](common/数据格式.md)支持1维ND，且shape必须与self一致，支持[非连续的Tensor](common/非连续的Tensor.md)。
  * minlength(int64_t，计算输入)：Host侧的int型，指定输出tensor最小长度。参数保证输出out的最小长度。如果计算出来的self的最大值小于minlength，则out的长度为minlength，否则为self的最大值加1。
  * out(aclTensor \*，计算输出)：Device侧的aclTensor，数据类型支持INT32、INT64、FLOAT、DOUBLE，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持1维ND。out的长度为self的最大值加1和minlength二者取最大。
  * workspaceSize(uint64_t \*，出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self或out是空指针。
161002 (ACLNN_ERR_PARAM_INVALID)：1. self、out、weights的数据类型和数据格式不在支持的范围之内。
                                  2. 当weights 不为空时，self、weights shape不一致。
```

## aclnnBincount

- **参数说明：**

  * workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnBincountGetWorkspaceSize获取。
  * executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
无

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max.h"
#include "aclnnop/aclnn_bincount.h"

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
  // device/stream初始化，参考acl API手册
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 先调max计算self中的最大元素值，然后与minlength计算输出tensorsize
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> maxOutShape = {1};

  void* selfDeviceAddr = nullptr;
  void* maxOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* maxOut = nullptr;
  std::vector<int32_t> selfHostData = {8, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> maxOutHostData(1, 0);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建maxOut aclTensor
  ret = CreateAclTensor(maxOutHostData, maxOutShape, &maxOutDeviceAddr, aclDataType::ACL_INT32, &maxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 调用CANN算子库API
  uint64_t workspaceSizeMax = 0;
  aclOpExecutor* executorMax;
  // 调用aclnnMax第一段接口
  ret = aclnnMaxGetWorkspaceSize(self, maxOut, &workspaceSizeMax, &executorMax);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddrMax = nullptr;
  if (workspaceSizeMax > 0) {
    ret = aclrtMalloc(&workspaceAddrMax, workspaceSizeMax, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMax第二段接口
  ret = aclnnMax(workspaceAddrMax, workspaceSizeMax, executorMax, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMax failed. ERROR: %d\n", ret); return ret);

  // 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 获取输出的值，将device侧内存上的结果拷贝至host侧
  std::vector<int32_t> resultData(1, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), maxOutDeviceAddr,
                    sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  aclDestroyTensor(maxOut);

  // 调用bincount
  int64_t minlength = 0;
  int64_t outSize = (resultData[0] < minlength) ? minlength : resultData[0] + 1;
  std::vector<int64_t> weightsShape = {8};
  std::vector<int64_t> outShape = {outSize};

  void* weightsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* weights = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> weightsHostData = {1, 1, 1.1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(outSize, 0);
  // 创建weights aclTensor
  ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, aclDataType::ACL_FLOAT, &weights);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnBincount第一段接口
  ret = aclnnBincountGetWorkspaceSize(self, weights, minlength, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBincountGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBincount第二段接口
  ret = aclnnBincount(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBincount failed. ERROR: %d\n", ret); return ret);

  // 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(outShape);
  std::vector<float> bincountResultData(size, 0);
  ret = aclrtMemcpy(bincountResultData.data(), bincountResultData.size() * sizeof(bincountResultData[0]), outDeviceAddr,
                    size * sizeof(bincountResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, bincountResultData[i]);
  }
  // 释放aclTensor
  aclDestroyTensor(self);
  aclDestroyTensor(weights);
  aclDestroyTensor(out);

  // 释放资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSizeMax > 0) {
    aclrtFree(workspaceAddrMax);
  }

  aclrtFree(weightsDeviceAddr);
  aclrtFree(maxOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
