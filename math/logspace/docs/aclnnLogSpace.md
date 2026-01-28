# aclnnLogSpace

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/logspace)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |   ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 推理系列产品</term>                             |     ×    |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 算子功能：创建一个大小为$\text{steps}$的一维张量，其值在$\text{base}^\text{start}$到$\text{base}^\text{end}$上对数尺度上均匀间隔，包含端点，以$\text{base}$为底。

- 计算公式：


$$
\text{result} = \left(\text{base}^\text{start},\ \text{base}^{\left(\text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1}\right)},\ \ldots,\ \text{base}^{\left(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1}\right)},\ \text{base}^\text{end}\right)
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLogSpaceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLogSpace”接口执行计算。

- `aclnnStatus aclnnLogSpaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, double base, const aclTensor *result, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLogSpace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLogSpaceGetWorkspaceSize

- **参数说明：**

  * start(aclScalar *，计算输入)：获取值的范围的起始位置，公式中的start，Host侧的aclScalar。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT16、BFLOAT16、FLOAT
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、BFLOAT16、FLOAT

  * end(aclScalar *，计算输入)：获取值的范围的结束位置，公式中的end，Host侧的aclScalar。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT16、BFLOAT16、FLOAT
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、BFLOAT16、FLOAT

  * steps(int64_t，计算输入)：获取值的步长，数据类型支持INT64，需要满足steps大于等于0。
  * base(double，计算输入)：对数空间的底数，默认为10.0，数据类型支持double。
  * result(aclTensor *，计算输出)：输出的对数间隔序列张量，公式中的result，Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，且out的元素个数需要与steps一致。不支持空Tensor。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT16、BFLOAT16、FLOAT
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、BFLOAT16、FLOAT

  * workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的start、end、steps或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. out的数据类型不在支持的范围之内。
                                        2. steps小于0。
                                        
  ```

## aclnnLogSpace

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnLogSpaceGetWorkspaceSize获取。

  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  * stream(aclrtStream, 入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnLogSpace默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include <math.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_logspace.h"

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
  // 固定写法，初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateOutputAclTensor(
    const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(),1);
    for (int64_t i = shape.size() - 2; i >=0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
  // 1. （固定写法）device/stream初始化，参考acl API文档
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* outDeviceAddr = nullptr;
  aclScalar* start = nullptr;
  aclScalar* end = nullptr;
  aclTensor* out = nullptr;

  float startValue = 0.0f;  //起始指数
  float endValue = 2.0f;    //结束指数
  int64_t steps = 5;    //步数
  float base = 10.0;    //底数

  std::vector<int64_t> shape = {steps};

  // 创建start aclScalar
  start = aclCreateScalar(&startValue, aclDataType::ACL_FLOAT);
  CHECK_RET(start != nullptr, LOG_PRINT("create start scalar failed\n"); return -1);
  // 创建end aclScalar
  end = aclCreateScalar(&endValue, aclDataType::ACL_FLOAT);
  CHECK_RET(end != nullptr, LOG_PRINT("create end scalar failed\n"); return -1);
  // 创建out aclTensor
  ret = CreateOutputAclTensor<float>(shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create output tensor failed\n"); return ret);
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnLogSpace第一段接口
  ret = aclnnLogSpaceGetWorkspaceSize(start, end, steps, base, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSpaceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnLogSpace第二段接口
  ret = aclnnLogSpace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSpace failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar
  aclDestroyScalar(start);
  aclDestroyScalar(end);
  aclDestroyTensor(out);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(outDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

