# aclnnVarMean

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/reduce_var)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品 </term>                             |   ×     |
| <term>Atlas 训练系列产品</term>                              |   √     |


## 功能说明

- 算子功能：返回输入Tensor指定维度的值求得的均值及方差。
- 计算公式：假设 dim 为 $i$，则对该维度进行计算。$N$为该维度的 shape。取 $self_{i}$，求出该维度上的平均值 $meanOut = \bar{self_{i}}$。
  方差计算公式如下：

  $$
  varOut = \frac{1}{max(0, N - correction)}\sum_{j=0}^{N-1}(self_{ij}-\bar{self_{i}})^2
  $$

  当`keepdim = true`时，reduce后保留该维度，且输出 shape 中该维度值为1；当 `keepdim = false`时，不保留该维度。
  当dim为nullptr或[]时，视为计算所有维度。

## 函数原型
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnVarMeanGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnVarMean”接口执行计算。

  - `aclnnStatus aclnnVarMeanGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim, aclTensor* varOut, aclTensor* meanOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnVarMean(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnVarMeanGetWorkspaceSize

- **参数说明：**

  - self（aclTensor*, 计算输入）：公式中的输入`self`，shape支持0到8维，self与meanOut的数据类型满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），self与varOut的数据类型满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - dim（aclIntArray*，入参）：公式中的`dim`，Host侧的aclIntArray，表示参与计算的维度，取值范围为[-self.dim(), self.dim()-1]，且其中的数据不能相同，支持的数据类型为INT64。当dim为nullptr或[]时，视为计算所有维度。
  - correction（int64_t，入参）：公式中的输入`correction`，修正值，数据类型为int64_t。
  - keepdim（bool，入参）：reduce轴的维度是否保留。数据类型为bool。
  - meanOut（aclTensor*, 计算输出）：均值的计算结果，self与meanOut的数据类型满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - varOut（aclTensor*, 计算输出）：公式中的输出`varOut`，方差的计算结果，self与varOut的数据类型满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

	aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、meanOut、varOut是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、meanOut、varOut的数据类型不在支持的范围之内。
                                        2. self的shape超过8维。
                                        3. dim的数值不合法（dim中的数据指向同一个维度、dim超出self的维度范围）。
                                        4. self与meanOut、varOut的shape不满足计算公式中的推导规则。
  ```

## aclnnVarMean

- **参数说明：**

  - workspace（void*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnVarMeanGetWorkspaceSize获取。
  - executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值：**

	aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnVarMean默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_var_mean.h"

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
  std::vector<int64_t> selfShape = {2,4};
  std::vector<int64_t> outShape = {2,1};
  void* selfDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* varDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dim = nullptr;
  aclTensor* var = nullptr;
  aclTensor* mean = nullptr;
  std::vector<float> selfHostData = {0.0, 1.1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> dimData = {1};
  int64_t correction = 1;
  bool keepdim = true;
  std::vector<float> varHostData = {0.0, 0};
  std::vector<float> meanHostData = {0.0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建dim aclIntArray
  dim = aclCreateIntArray(dimData.data(), dimData.size());
  CHECK_RET(dim != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(varHostData, outShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  ret = CreateAclTensor(meanHostData, outShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnVarMean第一段接口
  ret = aclnnVarMeanGetWorkspaceSize(self, dim, correction, keepdim, var, mean, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnVarMeanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnVarMean第二段接口
  ret = aclnnVarMean(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnVarMean failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> meanData(size, 0);
  ret = aclrtMemcpy(meanData.data(), meanData.size() * sizeof(meanData[0]), meanDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResult[%ld] is: %f\n", i, meanData[i]);
  }
  std::vector<float> varData(size, 0);
  ret = aclrtMemcpy(varData.data(), varData.size() * sizeof(varData[0]), varDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("varResult[%ld] is: %f\n", i, varData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(dim);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(meanDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
