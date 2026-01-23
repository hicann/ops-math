# aclnnHardtanh&aclnnInplaceHardtanh

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/clip_by_value_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 算子功能: 将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于clipValueMax则限制为clipValueMax，若元素小于clipValueMin则限制为clipValueMin，否则等于元素本身。
- 计算公式：

  $$
  HardTanh(x) = \left\{\begin{matrix}\begin{array}{l} 
  clipValueMax, \ if\ x>clipValueMax \\
  clipValueMin, \ if\ x<clipValueMin \\ 
  x, \ otherwise \\\end{array}\end{matrix}\right.\begin{array}{l}\end{array}
  $$
  
## 函数原型
- aclnnHardtanh和aclnnInplaceHardtanh实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnHardtanh：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceHardtanh：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnHardtanhGetWorkspaceSize”或者“aclnnInplaceHardtanhGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnHardtanh”或者“aclnnInplaceHardtanh”接口执行计算。

  - `aclnnStatus aclnnHardtanhGetWorkspaceSize(const aclTensor *self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnHardtanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceHardtanhGetWorkspaceSize(aclTensor *selfRef, const aclScalar* clipValueMin, const aclScalar* clipValueMax, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceHardtanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnHardtanhGetWorkspaceSize
- **参数说明**：
  - self(aclTensor*,计算输入): 公式中的x，Device侧的aclTensor，且数据类型和out保持一致，shape和out保持一致，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - clipValueMin(aclScalar*,计算输入): Host侧的aclScalar，下界，数据类型需要可转化成self的数据类型，且clipValueMin和clipValueMax同时存在时需要保持一致。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - clipValueMax(aclScalar*,计算输入): Host侧的aclScalar，上界，数据类型需要可转化成self的数据类型，且clipValueMin和clipValueMax同时存在时需要保持一致。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - out(aclTensor\*，计算输出)：Device侧的aclTensor，且数据类型和self保持一致，shape和self保持一致，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、out、clipValueMax、clipValueMin其中一个为空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、out的数据类型不在支持的范围之内。
                                     2. self的数据类型与输出out的类型不一致。
                                     3. self的shape与输出out的shape不一致。
                                     4. clipValueMin的数值大于clipValueMax。
```

## aclnnHardtanh
- **参数说明**：
  - workspace(void\*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnHardtanhGetWorkspaceSize获取。
  - executor(aclOpExecutor\*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceHardtanhGetWorkspaceSize
- **参数说明**：
  - selfRef(aclTensor\*，计算输入): Device侧的aclTensor。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - clipValueMin(aclScalar*,计算输入): Host侧的aclScalar，下界，数据类型需要可转化成selfRef的数据类型，且clipValueMin和clipValueMax同时存在时需要保持一致。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - clipValueMax(aclScalar*,计算输入): Host侧的aclScalar，上界，数据类型需要可转化成selfRef的数据类型，且clipValueMin和clipValueMax同时存在时需要保持一致。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BFLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64
  - workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的selfRef、clipValueMax、clipValueMin其中一个为空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef的数据类型不在支持的范围之内。
                                      2. clipValueMin的数值大于clipValueMax。
```

## aclnnInplaceHardtanh
- **参数说明**：
  - workspace(void\*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceHardtanhGetWorkspaceSize获取。
  - executor(aclOpExecutor\*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_hardtanh.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
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
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* clipValueMin = nullptr;
  aclScalar* clipValueMax = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float clipValueMinValue = 1.2f;
  float clipValueMaxValue = 2.4f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建clipValueMin aclScalar
  clipValueMin = aclCreateScalar(&clipValueMinValue, aclDataType::ACL_FLOAT);
  CHECK_RET(clipValueMin != nullptr, return ret);
  // 创建clipValueMax aclScalar
  clipValueMax = aclCreateScalar(&clipValueMaxValue, aclDataType::ACL_FLOAT);
  CHECK_RET(clipValueMax != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnHardtanh接口调用示例
  // 3. 调用CANN算子库API
  // 调用aclnnHardtanh第一段接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnHardtanhGetWorkspaceSize(self, clipValueMin, clipValueMax, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardtanhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnHardtanh第二段接口
  ret = aclnnHardtanh(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardtanh failed. ERROR: %d\n", ret); return ret);
  
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
 
  // aclnnInplaceHardtanh接口调用示例
  // 3. 调用CANN算子库API
  LOG_PRINT("\ntest aclnnInplaceHardtanh\n");
  // 调用aclnnInplaceHardtanh第一段接口
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  ret = aclnnInplaceHardtanhGetWorkspaceSize(self, clipValueMin, clipValueMax, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceHardtanhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceHardtanh第二段接口
  ret = aclnnInplaceHardtanh(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceHardtanh failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  size = GetShapeSize(outShape);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(clipValueMin);
  aclDestroyScalar(clipValueMax);
  aclDestroyTensor(out);
 
  // 7. 释放device资源，需要根据具体API的接口定义修改
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
