# aclnnStridedSliceAssignV2

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/conversion/strided_slice_assign_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

算子功能：StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用`aclnnStridedSliceAssignV2GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnStridedSliceAssignV2`接口执行计算。

- `aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(aclTensor *varRef, const aclTensor *inputValue, const aclIntArray *begin, const aclIntArray *end, const aclIntArray *strides, const aclIntArray *axesOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnStridedSliceAssignV2( void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnStridedSliceAssignV2GetWorkspaceSize

- **参数说明：**

  * varRef(aclTensor*，计算输入|计算输出)：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * inputValue(aclTensor*,计算输入)：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8，且数据类型需与varRef保持一致，shape需要与varRef计算得出的切片shape保持一致，综合约束请见[约束说明](#约束说明)。[数据格式](../../../docs/context/数据格式.md)支持ND。
  * begin(aclIntArray*,计算输入)：切片位置的起始索引，Host侧的aclIntArray。数据类型支持INT64，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * end(aclIntArray*,计算输入)：切片位置的终止索引，Host侧的aclIntArray。数据类型支持INT64，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * strides(aclIntArray*,计算输入)：切片的步长，Host侧的aclIntArray。数据类型支持INT64。strides必须为正数，varRef最后一维对应的strides取值必须为1。[数据格式](../../../docs/context/数据格式.md)支持ND。
  * axesOptional(aclIntArray*,计算输入)：可选参数，切片的轴，Host侧的aclIntArray。数据类型支持INT64，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、out或dim是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：输入和输出的数据类型不在支持的范围之内。
  ```

## aclnnStridedSliceAssignV2

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnStridedSliceAssignV2GetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnStridedSliceAssignV2默认确定性实现。
inputValue的shape第i维的计算公式为：$inputValueShape[i] = \lceil\frac{end[i] - begin[i]}{strides[i]} \rceil$，其中$\lceil x\rceil$ 表示对 $x$向上取整。$end$ 和 $begin$ 为经过特殊值调整后的取值，调整方式为：当 $end[i] < 0$ 时，$end[i]=varShape[i] + end[i]$ ，若仍有$end[i] < 0$，则 $end[i] = 0$ ，当 $end[i] > varShape[i]$ 时， $end[i] = varShape[i]$。$begin$ 同理。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_strided_slice_assign_v2.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> varRefShape = {4, 3};
  std::vector<int64_t> inputValueShape = {2, 2};
  std::vector<int64_t> sliceShape = {2};

  void* varRefDeviceAddr = nullptr;
  void* inputValueDeviceAddr = nullptr;

  aclTensor* varRef = nullptr;
  aclTensor* inputValue = nullptr;

  std::vector<float> varRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> inputValueHostData = {-1, -2, -3, -4};
  std::vector<int64_t> beginData = {1, 0};
  std::vector<int64_t> endData = {4, 2};
  std::vector<int64_t> stridesData = {2, 1};
  std::vector<int64_t> axesData = {0, 1};

  // 创建varRef aclTensor
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建inputValue aclTensor
  ret = CreateAclTensor(inputValueHostData, inputValueShape, &inputValueDeviceAddr, aclDataType::ACL_FLOAT, &inputValue);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建begin aclIntArray
  aclIntArray *begin = aclCreateIntArray(beginData.data(), beginData.size());
  // 创建end aclIntArray
  aclIntArray *end = aclCreateIntArray(endData.data(), endData.size());
  // 创建strides aclIntArray
  aclIntArray *strides = aclCreateIntArray(stridesData.data(), stridesData.size());
  // 创建axes aclTensor
  aclIntArray *axes = aclCreateIntArray(axesData.data(), axesData.size());

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnStridedSliceAssignV2第一段接口
  ret = aclnnStridedSliceAssignV2GetWorkspaceSize(varRef, inputValue, begin, end, strides, axes, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceAssignV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnStridedSliceAssignV2第二段接口
  ret = aclnnStridedSliceAssignV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceAssignV2 failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(varRef);
  aclDestroyTensor(inputValue);
  aclDestroyIntArray(begin);
  aclDestroyIntArray(end);
  aclDestroyIntArray(strides);
  aclDestroyIntArray(axes);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(varRefDeviceAddr);
  aclrtFree(inputValueDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```