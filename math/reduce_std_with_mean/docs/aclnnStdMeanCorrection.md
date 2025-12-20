# aclnnStdMeanCorrection

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/reduce_std_with_mean)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

| <term>Atlas 推理系列产品 </term>                             |   ×     |
| <term>Atlas 训练系列产品</term>                              |   √     |


## 功能说明

- 算子功能：计算样本标准差和均值。
- 计算公式：
  假设 dim 为 $i$，则对该维度进行计算。$N$为该维度的 shape。取 $self_{i}$，求出该维度上的平均值 $\bar{x_{i}}$。

  $$
  \left\{
  \begin{array} {rcl}
  meanOut& &= \bar{x_{i}}\\
  stdOut& &= \sqrt{\frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2}
  \end{array}
  \right.
  $$

  当`keepdim = true`时，reduce后保留该维度，且输出shape中该维度值为1；当`keepdim = false`时，不保留。

## 函数原型
每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnStdMeanCorrectionGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnStdMeanCorrection”接口执行计算。

- `aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim, aclTensor* stdOut, aclTensor* meanOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnStdMeanCorrection(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnStdMeanCorrectionGetWorkspaceSize

- **参数说明**

  - self（aclTensor\*, 计算输入）：公式中的`self`，Device侧的aclTensor，支持[非连续的Tensor](./common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16。
  - dim（aclIntArray\*, 计算输入）：公式中的`dim`，Host侧的aclIntArray，表示参与计算的维度，取值范围为[-self.dim(), self.dim()-1]，且其中的数据不能相同，支持的数据类型为INT64。当dim为nullptr或[]时，视为计算所有维度。
  - correction（int64_t, 计算输入）：公式中的$\delta N$值，Host侧的整型，修正值。
  - keepdim（bool, 计算输入）：公式中`keepdim`，Host侧的布尔型，是否在输出张量中保留输入张量的维度。
  - stdOut（aclTensor\*, 计算输出）：公式中`stdOut`，Device侧的aclTensor，支持[非连续的Tensor](./common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16。
  - meanOut（aclTensor\*, 计算输出）：公式中`meanOut`，Device侧的aclTensor，支持[非连续的Tensor](./common/非连续的Tensor.md)，[数据格式](./common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_INVALID）：1. 传入的 self、stdOut、meanOut是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、stdOut、meanOut数据类型不在支持的范围之内。
                                        2. dim 数组中的维度超出 self 的维度范围。
                                        3. dim 数组中元素重复。
                                        4. stdOut的shape出现如下情况会出错：
                                          keepdim为true时，stdOut.shape != self.shape(指定维度dim设置为1的形状)；
                                          keepdim为false时，stdOut.shape != self.shape(去除指定维度dim后的形状)。
                                        5. meanOut的shape出现如下情况会出错：
                                          keepdim为true时，meanOut.shape != self.shape(指定维度dim设置为1的形状)；
                                          keepdim为false时，meanOut.shape != self.shape(去除指定维度dim后的形状)。
  ```

## aclnnStdMeanCorrection

- **参数说明**

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnStdMeanCorrectionGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnStdMeanCorrection默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](./common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_std_mean_correction.h"

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
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> stdOutShape = {2, 4};
  std::vector<int64_t> meanOutShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* stdOutDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* stdOut = nullptr;
  aclTensor* meanOut = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24};
  std::vector<float> stdOutHostData = {1, 2, 3, 4, 5, 6, 7, 8.0};
  std::vector<float> meanOutHostData = {1, 2, 3, 4, 5, 6, 7, 8.0};
  std::vector<int64_t> dimData = {1};
  int64_t correction = 1;
  bool keepdim = false;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建stdOut aclTensor
  ret = CreateAclTensor(stdOutHostData, stdOutShape, &stdOutDeviceAddr, aclDataType::ACL_FLOAT, &stdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建meanOut aclTensor
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const aclIntArray *dim = aclCreateIntArray(dimData.data(), dimData.size());
  CHECK_RET(dim != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnStdMeanCorrection第一段接口
  ret = aclnnStdMeanCorrectionGetWorkspaceSize(self, dim, correction, keepdim, stdOut, meanOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStdMeanCorrectionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnStdMeanCorrection第二段接口
  ret = aclnnStdMeanCorrection(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStdMeanCorrection failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(stdOutShape);
  std::vector<float> stdResultData(size, 0);
  ret = aclrtMemcpy(stdResultData.data(), stdResultData.size() * sizeof(stdResultData[0]), stdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("stdResultData[%ld] is: %f\n", i, stdResultData[i]);
  }

  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(stdOut);
  aclDestroyTensor(meanOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(stdOutDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

