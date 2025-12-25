# aclnnGroupedBiasAddGrad

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/grouped_bias_add_grad)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：分组偏置加法（GroupedBiasAdd）的反向计算。本接口的扩展接口是[aclnnGroupedBiasAddGradV2](./aclnnGroupedBiasAddGradV2.md)。
- 计算公式：<br>
(1) 有可选输入groupIdxOptional时：

$$
out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
$$

&emsp;&emsp;其中，gradY共2维，H表示gradY最后一维的大小，G表示groupIdxOptional第0维的大小，即groupIdxOptional有G个数，groupIdxOptional(j)表示第j个数的大小，计算后out为2维，shape为(G, H)。<br>
&emsp;&emsp;(2) 无可选输入groupIdxOptional时：

$$
out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
$$

&emsp;&emsp;其中，gradY共3维，G, C, H依次表示gradY第0-2维的大小，计算后out为2维，shape为(G, H)。
- 示例：<br>
(1) 有可选输入groupIdxOptional时：<br>
  gradY的shape为(1000, 30)，groupIdxOptional为(400, 600, 1000)，将gradY分为3组，每组累加的行数依次为400、200、400，计算后out的shape为(3, 30)。<br>
(2) 无可选输入groupIdxOptional时：<br>
  gradY的shape为(10, 100, 30)，将gradY分为10组，每组累加的行数均为100，计算后out的shape为(10, 30)。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnGroupedBiasAddGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedBiasAddGrad”接口执行计算。

- `aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *groupIdxOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGroupedBiasAddGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGroupedBiasAddGradGetWorkspaceSize

- **参数说明：**

  * gradY（aclTensor\*，计算输入）: 必选参数，反向传播梯度，公式中的gradY，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。有可选输入groupIdxOptional时，shape仅支持2维，无可选输入groupIdxOptional时，shape仅支持3维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * groupIdxOptional（aclTensor\*，计算输入）: 可选参数，每个分组结束位置，公式中的groupIdxOptional，Device侧的aclTensor，数据类型支持INT32，INT64，shape仅支持1维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * out（aclTensor\*，计算输出）: bias的梯度，公式中的out，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据类型必须与gradY的数据类型一致，shape仅支持2维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * workspaceSize（uint64\_t\*，出参）: 返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）: 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：传入的gradY、out是空指针时。
  161002(ACLNN_ERR_PARAM_INVALID): 1. gradY或out的数据类型/维度不在支持的范围内。
                                   2. gradY、groupIdxOptional、out的维度关系不匹配。
                                   3. group组数超过2048。
  ```

## aclnnGroupedBiasAddGrad

- **参数说明：**

  * workspace（void\*，入参）: 在Device侧申请的workspace内存地址。
  * workspaceSize（uint64\_t，入参）: 在Device侧申请的workspace大小，由第一段接口aclnnGroupedBiasAddGradGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）: op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）: 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnGroupedBiasAddGrad默认确定性实现。
- groupIdxOptional最大支持2048个数。
- 有可选输入groupIdxOptional时，需要保证Tensor数据是递增排列，且最后一个数值需要等于gradY第0维的大小。
- 有可选输入groupIdxOptional时，需要保证Tensor数值不超过INT32最大值，并且是非负数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_bias_add_grad.h"

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
  std::vector<int64_t> gradYShape = {40, 10};
  std::vector<int64_t> groupIdxShape = {4};
  std::vector<int64_t> outShape = {4, 10};
  void* gradYDeviceAddr = nullptr;
  void* groupIdxDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradY = nullptr;
  aclTensor* groupIdx = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradYHostData(400, 1.0);

  std::vector<int32_t> groupIdxHostData = {5, 15, 30, 40};
  std::vector<float> outHostData(40, 0.0);

  // 创建gradY aclTensor
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建groupIdxOptional aclTensor
  ret = CreateAclTensor(groupIdxHostData, groupIdxShape, &groupIdxDeviceAddr, aclDataType::ACL_INT32, &groupIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupedBiasAddGrad第一段接口
  ret = aclnnGroupedBiasAddGradGetWorkspaceSize(gradY, groupIdx, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupedBiasAddGrad第二段接口
  ret = aclnnGroupedBiasAddGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGrad failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(gradY);
  aclDestroyTensor(groupIdx);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(groupIdxDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
