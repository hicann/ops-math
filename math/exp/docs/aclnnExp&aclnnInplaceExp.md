# aclnnExp&aclnnInplaceExp

## 支持的产品型号
- <term>Atlas 推理系列产品</term>。
- <term>Atlas 训练系列产品</term>。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。

## 功能说明

- 算子功能：返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。
- 计算公式：

$$
out_{i} = e^{self_{i}}
$$

## 函数原型

- aclnnExp和aclnnInplaceExp实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnExp：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceExp：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnExpGetWorkspaceSize” 或者 “aclnnInplaceExpGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小，再调用 “aclnnExp” 或者 “aclnnInplaceExp” 接口执行计算。

  - `aclnnStatus aclnnExpGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceExpGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnExpGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*,计算输入)：Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BOOL、INT64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BOOL、INT64、BFLOAT16。    
  - out(aclTensor *，计算输出)：Device侧的aclTensor，数据类型需要是self可转换的数据类型。当self为BOOL或INT64时，out需要是浮点类型。其shape需要与self一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。  
  - workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现如下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 参数self、out是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. 参数self的数据类型不在支持的范围内。
                                 2. 参数out的数据类型不是self可转换的。
                                 3. 参数self、out的shape不一致。
                                 4. 参数self、out的维度大于8。
```

## aclnnExp

- **参数说明：**

  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnExpGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## aclnnInplaceExpGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor *，计算输入)：Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。   
  - workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现如下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 参数selfRef是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. 参数selfRef的数据类型不在支持的范围内。
                                 2. 参数selfRef的维度大于8。
```

## aclnnInplaceExp

- **参数说明：**

  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceExpGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

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
#include "aclnnop/aclnn_exp.h"

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
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnExp第一段接口
  ret = aclnnExpGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnExp第二段接口
  ret = aclnnExp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExp failed. ERROR: %d\n", ret); return ret);
  
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // 调用aclnnInplaceExp第一段接口
  ret = aclnnInplaceExpGetWorkspaceSize(self, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceExpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceExp第二段接口
  ret = aclnnInplaceExp(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceExp failed. ERROR: %d\n", ret); return ret);

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

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr,
                    inplaceSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("inplaceResult[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
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
