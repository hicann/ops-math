# aclnnGer

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/ger)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：实现self和vec2的外积。

- 计算公式：

  $$
  out= self^T * vec2
  $$

## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnGerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGer”接口执行计算。

- `aclnnStatus aclnnGerGetWorkspaceSize(const aclTensor* self, const aclTensor* vec2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnGer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnGerGetWorkspaceSize

- **参数说明：**
  
  - self(aclTensor\*, 计算输入)：公式中的输入`self`, Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128，self与vec2的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)）。shape支持1维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - vec2(aclTensor\*, 计算输入): 公式中的输入`vec2`, Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128，self与vec2的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)）。shape支持1维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - out(aclTensor\*, 计算输出)：公式中的输出`out`, Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128，且数据类型是self与vec2推导之后可转换的数据类型（参见[互转换关系](../../../docs/context/互转换关系.md)）。shape支持2维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*, 出参): 返回op执行器，包含了算子计算流程。
  
- **返回值: **

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self，vec2或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self和vec2的数据类型不在支持的范围之内。
                                        2. self和vec2的数据格式不在支持的范围之内。
                                        3. 推导出的数据类型无法转换为指定输出out的类型。
                                        4. self或vec2的shape不为1维。
                                        5. out的shape不为2维。
                                        6. out在0和1维度上的size大小与self、vec2的size大小不完全相同。
  ```

## aclnnGer

- **参数说明：**
  
  - workspace(void\*, 入参): 在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnGerGetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参): op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参): 指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGer默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ger.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  std::vector<int64_t> selfShape = {2};
  std::vector<int64_t> vec2Shape = {3};
  std::vector<int64_t> outShape = {2, 3};
  void* selfDeviceAddr = nullptr;
  void* vec2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* vec2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 2.1};
  std::vector<float> vec2HostData = {1, 2, 3.0};
  std::vector<float> outHostData = {0, 0.1, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建vec2 aclTensor
  ret = CreateAclTensor(vec2HostData, vec2Shape, &vec2DeviceAddr, aclDataType::ACL_FLOAT, &vec2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGer第一段接口
  ret = aclnnGerGetWorkspaceSize(self, vec2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGer第二段接口
  ret = aclnnGer(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGer failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(vec2);
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(vec2DeviceAddr);
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
