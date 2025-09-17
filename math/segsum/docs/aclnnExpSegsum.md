# aclnnExpSegsum

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。
- 计算公式（以4D输入为例）：

  1. 输入self由（N1,N2,N3,N4）升维成（N1,N2,N3,N4,1）。
  2. 进行广播得到（N1,N2,N3,N4,N4）。
  3. 生成（N4,N4）类型为bool的三角矩阵A，上三角为True，下三角为False，对角线为True。
  4. 用0填充输入self里面与矩阵A中值为True的位置相对应的元素。
    
    $$
    self_i=
    \begin{cases}self_i,\quad A_i==False
    \\0, \quad A_i==True
    \end{cases}
    $$

  5. 以self的倒数第二维进行cumsum累加。从维度视角来看的某个元素（其它维度下标不变，当前维度下标依次递增），$selfTemp\_{i}$是输出张量中对应位置的元素。

     $$
     selfTemp_{i} = self_{1} + self_{2} + self_{3} + ...... + self_{i}
     $$

  6. 生成（N4,N4）类型为bool的三角矩阵B，上三角为True，下三角为False，对角线为False。
  7. 用-inf填充selfTemp里面与矩阵B中值为True的位置相对应的元素。
    
     $$
     out_i=
     \begin{cases}selfTemp_i,\quad B_i==False
     \\-inf, \quad B_i==True
     \end{cases}
     $$
  8. 计算selfTemp里面每个元素的指数。
    
     $$
     out_i=e^{selfTemp_i}
     $$

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnExpSegsumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnExpSegsum”接口执行计算。

- `aclnnStatus aclnnExpSegsumGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnExpSegsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnExpSegsumGetWorkspaceSize

- **参数说明**：

  - self（aclTensor*，计算输入）：Device侧的aclTensor。对应公式中的`self`，进行分段和计算的输入。数据类型支持FLOAT、BFLOAT16、FLOAT16。支持[非连续的Tensor](common/非连续的Tensor.md)，支持空Tensor。[数据格式](common/数据格式.md)支持ND。输入维度支持3D或4D。尾轴过大时输出占用空间过大，例如：输入尾轴为N时，输出占用内存是输入占用内存的N倍。

  - out（aclTensor\*，计算输出）：Device侧的aclTensor。对应公式中的`out`，完成分段和计算后的输出。数据类型支持FLOAT、BFLOAT16、FLOAT16。支持[非连续的Tensor](common/非连续的Tensor.md)，支持空Tensor。[数据格式](common/数据格式.md)支持ND。输出维度必须比输入维度大1，支持4D或5D。当输入self为3D时，输出前3维的维度大小与self的保持一致，最后1维的维度大小与第3维保持一致。当输入self为4D时，输出前4维的维度大小与self的保持一致，最后1维的维度大小与第4维保持一致。数据类型与输入self的保持一致。

  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、out的数据类型不在支持的范围之内。
                                        2. self、out的shape不满足参数要求。
  ```

## aclnnExpSegsum

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnExpSegsumGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_segsum.h"

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
  // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 1, 1, 4};
  std::vector<int64_t> outShape = {1, 1, 1, 4, 4};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<int64_t> outputSizeHostData = {16};
  std::vector<float> outHostData(16, 0);

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnExpSegsum第一段接口
  ret = aclnnExpSegsumGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnExpSegsum第二段接口
  ret = aclnnExpSegsum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsum failed. ERROR: %d\n", ret); return ret);
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