# aclnnArange

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：从start起始到end结束按照step的间隔获取值，并保存到输出的1维张量，其中数据范围为[start,end)。

- 计算公式：

$$
out_{i+1} = out_i+step
$$

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnArangeGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnArange”接口执行计算。

- `aclnnStatus aclnnArangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnArange(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnArangeGetWorkspaceSize

- **参数说明**：

  - start(aclScalar,计算输入)：Host侧的aclScalar，获取值的范围的起始位置。[数据格式](common/数据格式.md)支持ND。需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。bool类型独立进行运算（具体见约束说明）。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
  - end(aclScalar,计算输入)：Host侧的aclScalar，获取值的范围的结束位置。[数据格式](common/数据格式.md)支持ND。需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。bool类型独立进行运算（具体见约束说明）。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
  - step(aclScalar,计算输入)：Host侧的aclScalar，获取值的步长。[数据格式](common/数据格式.md)支持ND。需要满足step不等于0。bool类型独立进行运算，只支持true。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
  - out(aclTensor，计算输出)：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、INT32、INT64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、INT32、INT64、BFLOAT16。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、BFLOAT16、INT32、INT64。
  - workspaceSize(uint64_t\*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的start、end、step或out是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. start、end、step或out的数据类型不在支持的范围之内。
                                   2. start、end、step不满足range的运算逻辑，即在step大于0时输入的start大于end，或者step小于0时输入的start小于end。
  ```

## aclnnArange

- **参数说明**：

  - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnArangeGetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

Warning：输入数据类型为float时，受限于数据类型本身的精度误差，对out的输出大小计算请采用float。如果用户采用double计算输出, double结果可能小于float结果，此时Tiling侧会进行校验告警。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include <math.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_arange.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. device/stream初始化，参考acl API手册
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  void* outDeviceAddr = nullptr;
  aclScalar* start = nullptr;
  aclScalar* end = nullptr;
  aclScalar* step = nullptr;
  aclTensor* out = nullptr;
  float startValue = 1.0f;
  float endValue = 5.0f;
  float stepValue = 1.0f;
  double size_arange = ceil(static_cast<double>(endValue - startValue) / stepValue);
  int64_t size_value = static_cast<int64_t>(size_arange);
  std::vector<int64_t> outShape = {size_value};
  std::vector<float> outHostData(size_value, 0);
  
  // 创建start aclScalar
  start = aclCreateScalar(&startValue, aclDataType::ACL_FLOAT);
  CHECK_RET(start != nullptr, return ret);
  // 创建end aclScalar
  end = aclCreateScalar(&endValue, aclDataType::ACL_FLOAT);
  CHECK_RET(end != nullptr, return ret);
  // 创建step aclScalar
  step = aclCreateScalar(&stepValue, aclDataType::ACL_FLOAT);
  CHECK_RET(step != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnArange第一段接口
  ret = aclnnArangeGetWorkspaceSize(start, end, step, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArangeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnArange第二段接口
  ret = aclnnArange(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArange failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(outShape);
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
  aclDestroyScalar(step);
  aclDestroyTensor(out);
  
  // 7. 释放device资源，需要根据具体API的接口定义修改
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
