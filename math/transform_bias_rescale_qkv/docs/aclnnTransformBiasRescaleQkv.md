# aclnnTransformBiasRescaleQkv

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

- 算子功能：
  `TransformBiasRescaleQkv` 算子是一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口。它用于调整这些向量的偏置（Bias）和缩放（Rescale）因子，以优化注意力计算过程。

- 计算公式：  
  逐个元素计算过程见公式：

  $$
   \left\{
	\begin{array}{ll}
		q_o=(q_i+q_{bias})/\sqrt{dim\_per\_head} \\
		k_o=k_i+k_{bias} \\
        v_o=v_i+v_{bias} 
	\end{array}\right.
  $$

  公式中：
  - dim_per_head为每个注意力头的维度。
  - $q_o$、$k_o$、$v_o$分别为查询（Query）、键（Key）、值（Value）向量的输出元素。
  - $q_i$、$k_i$、$v_i$分别为查询（Query）、键（Key）、值（Value）向量的输入元素。
  - $q_{bias}$、$k_{bias}$、$v_{bias}$分别为查询（Query）、键（Key）、值（Value）向量的输入元素偏移。

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnTransformBiasRescaleQkvGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnTransformBiasRescaleQkv”接口执行计算。

- `aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(const aclTensor *qkv,
    const aclTensor *qkvBias,
    int64_t numHeads,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)`
- `aclnnStatus aclnnTransformBiasRescaleQkv(void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)`

## aclnnTransformBiasRescaleQkvGetWorkspaceSize

- **参数说明**：

  - qkv（aclTensor\*，计算输入）：输入的张量，公式中的$q_i$、$k_i$、$v_i$。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor，[数据格式](common/数据格式.md)支持ND。shape为{B,T,3 * num_heads * dim_per_head}三维张量。B为批量大小，T为序列长度，num_heads为注意力头数，dim_per_head为每个注意力头的维度。
  - qkvBias（aclTensor\*，计算输入）：输入的张量，公式中的$q_{bias}$、$k_{bias}$、$v_{bias}$。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor，[数据格式](common/数据格式.md)支持ND。shape为{3 * num_heads * dim_per_head}一维张量。
  - numHeads（int64_t，计算输入）：输入的头数，数据类型支持Int64，取值大于0。
  - qOut（aclTensor\*，计算输出）：输出张量，公式中的$q_o$。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。shape为{B,num_heads,T,dim_per_head}四维张量。
  - kOut（aclTensor\*，计算输出）：输出张量，公式中的$k_o$。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。shape为{B,num_heads,T,dim_per_head}四维张量。
  - vOut（aclTensor\*，计算输出）：输出张量，公式中的$v_o$。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。shape为{B,num_heads,T,dim_per_head}四维张量。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 传入的输入和输出是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. qkv和qkvBias的数据类型和数据格式不在支持的范围之内。
                                        2. qkv和qkvBias数据类型不一致。
                                        3. qkv和qkvBias的shape不满足参数说明的要求。
                                         
  ```
## aclnnTransformBiasRescaleQkv

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnTransformBiasRescaleQkvGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

  - 输入qkv、qkvBias和输出qOut、kOut、vOut的数据类型需要保持一致。
  - 输入值为nan，输出也为nan，输入是inf，输出也是inf。
  - 输入是-inf，输出也是-inf。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_transform_bias_rescale_qkv.h"

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
    LOG_PRINT("outPut result[%ld] is: %f\n", i, resultData[i]);
  }
}

void PrintInResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("input[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
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
// qkv
int64_t B = 3;
int64_t T = 4;
int64_t n = 3;
int64_t d = 16;
std::vector<int64_t> qkvShape = {B, T, 3 * n * d};
int qkvCount = B * T * 3 * n * d;
std::vector<float> qkvHostData(qkvCount, 1);

for (int i = 0; i < qkvCount; ++i) {
    qkvHostData[i] = i * 1.0;
}

void* qkvDeviceAddr = nullptr;
aclTensor* qkv = nullptr;
// 创建input aclTensor
ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT, &qkv);
CHECK_RET(ret == ACL_SUCCESS, return ret);

// qkvBias
std::vector<int64_t> qkvBiasShape = {3 * n * d};
std::vector<float> qkvBiasHostData(3 * n * d, 0.5);

void* qkvBiasDeviceAddr = nullptr;
aclTensor* qkvBias = nullptr;
// 创建input aclTensor
ret = CreateAclTensor(qkvBiasHostData, qkvBiasShape, &qkvBiasDeviceAddr, aclDataType::ACL_FLOAT, &qkvBias);
CHECK_RET(ret == ACL_SUCCESS, return ret);

std::vector<int64_t> outShape = {B, n, T, d};
std::vector<float> outHostData(qkvCount / 3, 1);
aclTensor* outQ = nullptr;
aclTensor* outK = nullptr;
aclTensor* outV = nullptr;
void* outQDeviceAddr = nullptr;
void* outKDeviceAddr = nullptr;
void* outVDeviceAddr = nullptr;

// 创建out aclTensor
ret = CreateAclTensor(outHostData, outShape, &outQDeviceAddr, aclDataType::ACL_FLOAT, &outQ);
ret = CreateAclTensor(outHostData, outShape, &outKDeviceAddr, aclDataType::ACL_FLOAT, &outK);
ret = CreateAclTensor(outHostData, outShape, &outVDeviceAddr, aclDataType::ACL_FLOAT, &outV);

CHECK_RET(ret == ACL_SUCCESS, return ret);

// 3. 调用CANN算子库API，需要修改为具体的Api名称
uint64_t workspaceSize = 16 * 1024 * 1024;
aclOpExecutor* executor;

// LOG_PRINT("qkv input=====");
// PrintInResult(qkvShape, &qkvDeviceAddr);

// LOG_PRINT("qkvBias input=====");
// PrintInResult(qkvBiasShape, &qkvBiasDeviceAddr);

// 调用aclnnTransformBiasRescaleQkv第一段接口
ret = aclnnTransformBiasRescaleQkvGetWorkspaceSize(
qkv,
qkvBias,
n,
outQ,
outK,
outV,
&workspaceSize,
&executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkvGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

// 根据第一段接口计算出的workspaceSize申请device内存
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
}

// 调用aclnnTransformBiasRescaleQkv第二段接口
ret = aclnnTransformBiasRescaleQkv(
workspaceAddr,
workspaceSize,
executor,
stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkv failed. ERROR: %d\n", ret); return ret);

// 4. （固定写法）同步等待任务执行结束
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
LOG_PRINT("q output=====");
PrintOutResult(outShape, &outQDeviceAddr);

LOG_PRINT("k output=====");
PrintOutResult(outShape, &outKDeviceAddr);


LOG_PRINT("v output=====");
PrintOutResult(outShape, &outVDeviceAddr);

// 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
aclDestroyTensor(qkv);
aclDestroyTensor(qkvBias);
aclDestroyTensor(outQ);
aclDestroyTensor(outK);
aclDestroyTensor(outV);

// 7.释放device资源，需要根据具体API的接口定义修改
aclrtFree(qkvDeviceAddr);
aclrtFree(qkvBiasDeviceAddr);

aclrtFree(outQDeviceAddr);
aclrtFree(outKDeviceAddr);
aclrtFree(outVDeviceAddr);

if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();

return 0;
}
```