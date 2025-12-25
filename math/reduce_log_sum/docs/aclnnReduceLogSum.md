# aclnnReduceLogSum

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/reduce_log_sum)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

算子功能：返回给定维度中输入张量每行的和再取对数。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnReduceLogSumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReduceLogSum”接口执行计算。
* `aclnnStatus aclnnReduceLogSumGetWorkspaceSize(const aclTensor* data, const aclIntArray* axes, bool keepDims, bool noopWithEmptyAxes, aclTensor* reduce, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnReduceLogSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnReduceLogSumGetWorkspaceSize

- **参数说明**：

  * data（aclTensor*, 计算输入）：表示参与计算的目标张量，Device侧的aclTensor，shape支持0-8维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，数据类型支持FLOAT16、FLOAT32，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * axes（aclIntArray*, 计算输入）：指定计算维度，Host侧的aclIntArray，数据类型支持INT64，取值范围为[-self.dim(), self.dim()-1]。
  * keepDims（bool, 计算输入）：指定是否在输出张量中保留输入张量的维度，Host侧的BOOL值。
  * noopWithEmptyAxes（bool, 计算输入）：指定axes为空时的行为：false即对所有轴进行计算；true即不进行计算，输出张量等于输入张量，Host侧的BOOL值。
  * reduce（aclTensor*, 计算输出）：表示计算后的结果，Device侧的aclTensor，shape支持0-8维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，数据类型支持FLOAT16、FLOAT32，需与data一致，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的data、axes或reduce是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. data或reduce的数据类型不在支持的范围之内。
                                   2. reduce shape与实际不匹配。
                                   3. axes数组中的维度超出输入tensor的维度范围。
                                   4. axes数组中的轴重复。
```
## aclnnReduceLogSum

- **参数说明**：

  * workspace（void*, 入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnReduceLogSumGetWorkspaceSize获取。
  * executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnReduceLogSum默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_log_sum.h"

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
  // 1. （固定写法）device/context/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> dataShape = {4, 2};
  std::vector<int64_t> outShape = {2};
  void* dataDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* data = nullptr;
  aclIntArray* axes = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> dataHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0};
  std::vector<int64_t> axesData = {0};
  bool keepDims = false;
  bool noopWithEmptyAxes = false;
  // 创建data aclTensor
  ret = CreateAclTensor(dataHostData, dataShape, &dataDeviceAddr, aclDataType::ACL_FLOAT, &data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建axes aclIntArray
  axes = aclCreateIntArray(axesData.data(), 1);
  CHECK_RET(axes != nullptr, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnReduceLogSum第一段接口
  ret = aclnnReduceLogSumGetWorkspaceSize(data, axes, keepDims, noopWithEmptyAxes, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceLogSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnReduceLogSum第二段接口
  ret = aclnnReduceLogSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceLogSum failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(data);
  aclDestroyIntArray(axes);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(dataDeviceAddr);
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
