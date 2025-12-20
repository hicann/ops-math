# aclnnRemainderScalarTensor

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/floor_mod)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明
- 算子功能: 将scalar self进行broadcast成和tensor other一样shape的tensor以后，其中的每个元素都转换为除以other的对应元素以后得到的余数。该结果与除数other同符号，并且该结果的绝对值是小于other的绝对值。
  实际计算remainder(self, other) 等效于以下公式：

  $$
  out_i = self - floor(self / other_i) * other_i
  $$

- 示例：

```
self = 5.0   # float
other = tensor([[-1, -2],
                [-3, -4]]).type(int32)
result = remainder(self, other)

# result的值
# tensor([[ 0., -1.],
#         [-1., -3.]])  float

# 对于元素other中的-4来说，计算结果为 5 - floor(5 / -4) * -4 = -3
# 可以看到，最终结果-3的绝对值小于原来的-4的绝对值。
```

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnRemainderScalarTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRemainderScalarTensor”接口执行计算。

* `aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnRemainderScalarTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRemainderScalarTensorGetWorkspaceSize

- **参数说明：**

  * self(aclScalar*, 计算输入)：公式中的输入`self`，Host侧的aclScalar。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与other的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](common/TensorScalar互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
    - <term>Atlas 训练系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE。数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。

  * other(aclTensor*, 计算输入)：公式中的输入`other`, Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>昇腾910_95 AI处理器</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与self的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](common/TensorScalar互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
    - <term>Atlas 训练系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE。数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。

  * out(aclTensor \*, 计算输出)：公式中的输出`out`，Device侧的aclTensor。shape需要与other一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。
    - <term>Atlas 训练系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE。

  * workspaceSize(uint64_t \*，出参)：返回需要在Device侧申请的workspace大小。

  * executor(aclOpExecutor \*\*，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR)： 1. 传入的self、other、out是空指针。
161002 (ACLNN_ERR_PARAM_INVALID)： 1. other、out的shape不一样。
                                   2. self和other无法做数据类型推导。
                                   3. self和other推导出的数据类型不属于支持的数据类型。
                                   4. self和other推导出的数据类型无法转换为指定输出out的类型。
                                   5. other、out的维度数大于8维。
```

## aclnnRemainderScalarTensor

- **参数说明：**

  * workspace(void \*，入参)：在Device侧申请的workspace内存地址。

  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnRemainderScalarTensorGetWorkspaceSize获取。

  * executor(aclOpExecutor \*，入参)：op执行器，包含了算子计算流程。

  * stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRemainderScalarTensor默认确定性实现。

- 当self的数据类型为INT32时，优先保障在范围[-2^24, 2^24]内的功能和精度；
- 当other为0，且self的数据类型为整型时，out的结果为self。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_remainder.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> outShape = {3, 3};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t Self = 3;

  // 创建self aclScalar
  self = aclCreateScalar(&Self, aclDataType::ACL_INT64);
  CHECK_RET(self != nullptr, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRemainderScalarTensor第一段接口
  ret = aclnnRemainderScalarTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRemainderScalarTensor第二段接口
  ret = aclnnRemainderScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderScalarTensor failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(otherDeviceAddr);
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

