# aclnnIsInTensorScalar

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

算子功能：检查element中的元素是否等于testElement。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnIsInTensorScalarGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIsInTensorScalar”接口执行计算。

+ `aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor *element, const aclScalar *testElement,bool assumeUnique, bool invert, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
+ `aclnnStatus aclnnIsInTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnIsInTensorScalarGetWorkspaceSize
- **参数说明：**

  * element(aclTensor*, 计算输入)：shape维度不高于8维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE，且与testElement满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
  * testElement(aclScalar*, 计算输入)：与element满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE。
    * <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE，且与testElement满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
  * assumeUnique(bool, 计算输入): 若为True，则假定element和testElement中元素唯一，用于加快计算速度。
  * invert(bool, 计算输入): 表示输出结果是否需要反转。
  * out(aclTensor*, 计算输出)：数据类型支持BOOL，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，shape需要与element的shape相同，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * workspaceSize(uint64_t*, 出参)：返回用户需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. 传入的 element、testElement、out是空指针时。
  161002 ACLNN_ERR_PARAM_INVALID: 1. element、testElement的数据类型不在支持的范围之内。
                                  1. element和testElement无法做数据类型推导。
                                  2. element和testElement推导后的数据类型不在支持范围内。
                                  3. out的数据类型不是bool。
                                  5、element、out的维度大于8维。
                                  4. out的shape与element的shape不一致。
  ```

## aclnnIsInTensorScalar
- **参数说明：**

  + workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  + workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnIsInTensorScalarGetWorkspaceSize获取。
  + executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  + stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnIsInTensorScalar默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isin_tensor_scalar.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& elementShape, std::vector<int64_t>& outShape, void** elementDeviceAddr, void** outDeviceAddr,
    aclTensor** element, aclTensor** out, aclScalar** testElement, bool& assumeUnique, bool& invert)
{
  std::vector<double> elementHostData = {0, 1, 2, 3, 2};
  std::vector<char> outHostData = {5, 0};
  double testElementValue = 2;

  // 创建 testElement scalar
  *testElement = aclCreateScalar(&testElementValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*testElement != nullptr, return ACL_ERROR_INVALID_PARAM);

  // 创建 element tensor
  auto ret = CreateAclTensor(elementHostData, elementShape, elementDeviceAddr, aclDataType::ACL_DOUBLE, element);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 out tensor
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* element, aclScalar* testElement, bool assumeUnique, bool invert, aclTensor* out, void** workspaceAddrOut,
    uint64_t& workspaceSize, void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // 第一段接口
  auto ret =
      aclnnIsInTensorScalarGetWorkspaceSize(element, testElement, assumeUnique, invert, out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 分配 workspace
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // 第二段接口
  ret = aclnnIsInTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInTensorScalar failed. ERROR: %d\n", ret); return ret);

  // 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝输出
  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> elementShape = {5};
  std::vector<int64_t> outShape = {5};
  void* elementDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* element = nullptr;
  aclScalar* testElement = nullptr;
  aclTensor* out = nullptr;

  bool assumeUnique = false;
  bool invert = false;

  ret = CreateInputs(
      elementShape, outShape, &elementDeviceAddr, &outDeviceAddr, &element, &out, &testElement, assumeUnique, invert);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      element, testElement, assumeUnique, invert, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 释放
  aclDestroyScalar(testElement);
  aclDestroyTensor(element);
  aclDestroyTensor(out);

  aclrtFree(elementDeviceAddr);
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

