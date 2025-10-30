# aclnnIsInTensorScalar

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：检查element中的元素是否等于testElement。

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnIsInTensorScalarGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIsInTensorScalar”接口执行计算。

+ `aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor *element, const aclScalar *testElement,bool assumeUnique, bool invert, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
+ `aclnnStatus aclnnIsInTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnIsInTensorScalarGetWorkspaceSize
- **参数说明：**

  * element(aclTensor*, 计算输入)：shape维度不高于8维。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE，且与testElement满足[TensorScalar互推导关系](common/TensorScalar互推导关系.md)。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE，且与testElement满足[互推导关系](common/互推导关系.md)。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE，且与testElement满足[互推导关系](common/互推导关系.md)。
  * testElement(aclScalar*, 计算输入)：与element满足[TensorScalar互推导关系](common/TensorScalar互推导关系.md)。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE。
  * assumeUnique(bool, 计算输入): 若为True，则假定element和testElement中元素唯一，用于加快计算速度。
  * invert(bool, 计算输入): 表示输出结果是否需要反转。
  * out(aclTensor*, 计算输出)：数据类型支持BOOL，支持[非连续的Tensor](common/非连续的Tensor.md)，shape需要与element的shape相同，[数据格式](common/数据格式.md)支持ND。
  * workspaceSize(uint64_t*, 出参)：返回用户需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. 传入的 element、testElement、out是空指针时。
  161002 ACLNN_ERR_PARAM_INVALID: 1. element、testElement的数据类型不在支持的范围之内。
                                  2. element和testElement无法做数据类型推导。
                                  3. element和testElement推导后的数据类型不在支持范围内。
                                  4. out的数据类型不是bool。
                                  5、element、out的维度大于8维。
                                  6. out的shape与element的shape不一致。
  ```

## aclnnIsInTensorScalar
- **参数说明：**

  + workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  + workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnIsInTensorScalarGetWorkspaceSize获取。
  + executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  + stream(aclrtStream, 入参)：指定执行任务的Stream。

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
  std::vector<int64_t> elementShape = {5};
  std::vector<int64_t> outShape = {5};
  void* elementDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* element = nullptr;
  aclScalar* testElement = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> elementHostData = {0, 1, 2, 3, 2};
  std::vector<char> outHostData = {5, 0};
  float testElementValue = 2;
  bool assumeUnique = false;
  bool invert = false;

  // 创建testElement aclScalar
  testElement = aclCreateScalar(&testElementValue, aclDataType::ACL_FLOAT);
  CHECK_RET(testElement != nullptr, return ret);
  // 创建element aclTensor
  ret = CreateAclTensor(elementHostData, elementShape, &elementDeviceAddr, aclDataType::ACL_FLOAT, &element);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BOOL, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnIsInTensorScalar第一段接口
  ret = aclnnIsInTensorScalarGetWorkspaceSize(element, testElement, assumeUnique, invert, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnIsInTensorScalar第二段接口
  ret = aclnnIsInTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInTensorScalar failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyScalar(testElement);
  aclDestroyTensor(element);
  aclDestroyTensor(out);

  // 7. 释放device资源， 需要根据具体API的接口定义修改
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

