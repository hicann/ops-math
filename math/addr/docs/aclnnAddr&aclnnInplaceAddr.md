# aclnnAddr&aclnnInplaceAddr

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/addr)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |



## 功能说明

- 算子功能：求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出

- 计算公式：
  $$
  \text{out} = \beta\ \text{self} + \alpha\ (\text{vec1} \otimes\text{vec2})
  $$

## 函数原型
- aclnnAddr和aclnnInplaceAddr实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnAddr：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceAddr：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAddrGetWorkspaceSize”或者“aclnnInplaceAddrGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnAddr”或者“aclnnInplaceAddr”接口执行计算。

  * `aclnnStatus aclnnAddrGetWorkspaceSize(const aclTensor *self, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *betaOptional, const aclScalar *alphaOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnAddr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  * `aclnnStatus aclnnInplaceAddrGetWorkspaceSize(aclTensor *selfRef, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *betaOptional, const aclScalar *alphaOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceAddr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnAddrGetWorkspaceSize

- **参数说明：**
  
  - self(aclTensor*, 计算输入): 外积扩展矩阵，Device侧的aclTensor，shape维度不能超过2，并且需要与vec1、vec2满足[broadcast关系](../../../docs/zh/context/broadcast关系.md), 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - vec1(aclTensor*, 计算输入): 外积入参第一向量，一维向量，Device侧的aclTensor，shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md), 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - vec2(aclTensor*, 计算输入): 外积入参第二向量，一维向量，Device侧的aclTensor，shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md), 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - betaOptional(aclScalar*, 计算输入): 外积扩展矩阵比例因子，即公式中的β，host侧的aclScalar，如果betaOptional为bool类型，则self/vec1/vec2的数据类型只能是bool；如果self/vec1/vec2为整型，则betaOptional、alphaOptional不能为浮点型；[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - alphaOptional(aclScalar*, 计算输入): 外积比例因子，即公式中的α，host侧的aclScalar，如果alphaOptional为bool类型，则self/vec1/vec2的数据类型只能是bool；如果self/vec1/vec2为整型，则betaOptional、alphaOptional不能为浮点型；[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - out(aclTensor\*, 计算输出): 输出结果，Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor\*\*, 出参): 返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的tensor或out是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. self、vec1和vec2的数据类型和数据格式不在支持的范围之内。
                                    2. vec1和vec2维度不为1，self维度超过2。
                                    3. self不能扩展成为vec1和vec2的外积结果形状。
                                    4. beta或者alpha为bool类型时，self、vec1、vec2数据类型非bool类型
                                    5. self、vec1、vec2类型都为整型或bool或“整型+bool”时，beta或alpha为浮点型
  ```

## aclnnAddr

- **参数说明：**
  
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnAddrGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## aclnnInplaceAddrGetWorkspaceSize

- **参数说明：**
  
  - selfRef(aclTensor\*, 计算输入|计算输出): 外积扩展矩阵及输出矩阵，Device侧的aclTensor，shape维度为2，不支持空Tensor，并且需要与vec1、vec2满足[broadcast关系](../../../docs/zh/context/broadcast关系.md), 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - vec1(aclTensor*, 计算输入)：外积入参第一向量，一维向量，Device侧的aclTensor，shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md), 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - vec2(aclTensor*, 计算输入): 外积入参第二向量，一维向量，Device侧的aclTensor，shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md), 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - betaOptional(aclScalar*, 计算输入): 外积扩展矩阵比例因子，即公式中的β，host侧的aclScalar，如果betaOptional为bool类型，则self/vec1/vec2的数据类型只能是bool；如果self/vec1/vec2为整型，则betaOptional、alphaOptional不能为浮点型；[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - alphaOptional(aclScalar*, 计算输入): 外积比例因子，即公式中的α，host侧的aclScalar，如果alphaOptional为bool类型，则self/vec1/vec2的数据类型只能是bool；如果self/vec1/vec2为整型，则betaOptional、alphaOptional不能为浮点型；[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor\*\*, 出参): 返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的tensor是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. selfRef、vec1和vec2的数据类型和数据格式不在支持的范围之内。
                                    2. vec1和vec2维度不为1，selfRef维度不为2。
                                    3. selfRef不能扩展成为vec1和vec2的外积结果形状。
                                    4. beta或者alpha为bool类型时，selfRef、vec1、vec2数据类型非bool类型
                                    5. selfRef、vec1、vec2类型都为整型或bool或“整型+bool”时，beta或alpha为浮点型
  ```

## aclnnInplaceAddr

- **参数说明：**
  
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnInplaceAddrGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnAddr&aclnnInplaceAddr默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addr.h"

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
  // （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {3, 2};
  std::vector<int64_t> vec1Shape = {3};
  std::vector<int64_t> vec2Shape = {2};
  std::vector<int64_t> outShape = {3, 2};

  void* inputDeviceAddr = nullptr;
  void* vec1DeviceAddr = nullptr;
  void* vec2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* vec1 = nullptr;
  aclTensor* vec2 = nullptr;
  aclScalar* beta = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> inputHostData = {6, 0};
  std::vector<float> vec1HostData = {1, 2, 3};
  std::vector<float> vec2HostData = {4, 5};
  std::vector<float> outHostData = {6, 0};
  float betaValue = 1.5f;
  float alphaValue = 1.5f;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vec1HostData, vec1Shape, &vec1DeviceAddr, aclDataType::ACL_FLOAT, &vec1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vec2HostData, vec2Shape, &vec2DeviceAddr, aclDataType::ACL_FLOAT, &vec2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建beta和alpha scalar值
  beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);

 
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
   
   // aclnnAddr接口调用示例
   // 调用aclnnAddr第一段接口
  ret = aclnnAddrGetWorkspaceSize(input, vec1, vec2, beta, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnAddr第二段接口
  ret = aclnnAddr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddr failed. ERROR: %d\n", ret); return ret);

  // （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceAddr接口调用示例
  // 调用aclnnInplaceAddr第一段接口
  ret = aclnnInplaceAddrGetWorkspaceSize(input, vec1, vec2, beta, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnInplaceAddr第二段接口
  ret = aclnnInplaceAddr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddr failed. ERROR: %d\n", ret); return ret);

  // （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 释放aclTensor和aclScalar
  aclDestroyTensor(input);
  aclDestroyTensor(vec1);
  aclDestroyTensor(vec2);
  aclDestroyTensor(out);

  // 释放device 资源
  aclrtFree(inputDeviceAddr);
  aclrtFree(vec1DeviceAddr);
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
