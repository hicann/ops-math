# aclnnSlice

## 产品支持情况

- <term>Atlas 训练系列产品</term>。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。

## 功能说明

算子功能：在指定维度$dim$上，根据给定的范围$[start, end]$和步长$step$，从输入张量$self$中提取子张量$out$。
$start$和$end$可以取$[0, self.shape[dim]]$以外的值，取值后根据以下公式转换为合法值，假设self.shape[dim] = N：

$$
start = \begin{cases}
&0, & if\;start < -N \\
&N, & if\;start >= N\\
&(start+N) \% N, & otherwise
\end{cases}
$$

$$
end = \begin{cases}
&N, & if\; end >= N\\
&start, & else\;if\;  (end+N)\%N < start \\
&(end+N)\%N, & otherwise\\
\end{cases}
$$

$out.shape$与$self.shape$只有dim轴上不一致，其他轴一致:

$$
out.shape[dim] = ⌊\frac{end - start + step - 1}{step}⌋
$$

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnSliceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSlice”接口执行计算。

- `aclnnStatus aclnnSliceGetWorkspaceSize(const aclTensor* self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSlice(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnSliceGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的张量$self$，Device侧的aclTensor。shape支持1到8维，self与out的数据类型一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT8、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT8、UINT8、BOOL、BFLOAT16。

  - dim(int64_t, 计算输入)：公式中的参数$dim$，指定的维度，Host侧的整型数值，取值范围为[-self.dim(), self.dim() - 1]。

  - start(int64_t, 计算输入)：公式中的参数$start$，切片位置的起始索引，Host侧的整型数值。

  - end(int64_t, 计算输入)：公式中的参数$end$，切片位置的终止索引，Host侧的整型数值。

  - step(int64_t, 计算输入)：公式中的参数$step$，指定切片的步长，Host侧的整型数值，取值范围为大于0的整数。

  - out(aclTensor*, 计算输出)：公式中的张量$out$，Device侧的aclTensor。shape支持1到8维，shape满足计算公式中的推导规则，self与out的数据类型一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT8、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT8、UINT8、BOOL、BFLOAT16。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self和out的数据类型不在支持的范围之内。
                                        2. self的shape维度为0或大于8。
                                        3. dim的取值不在[-self.dim(), self.dim() - 1]。
                                        4. step小于等于0。
                                        5. self和out的数据类型不一致。
                                        6. out的shape不满足计算公式中的推导规则。
  ```

## aclnnSlice

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnSliceGetWorkspaceSize获取。

  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_slice.h"

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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0, 0, 0};
  int64_t dim = 0;
  int64_t start = 1;
  int64_t end = 3;
  int64_t step = 1;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSlice第一段接口
  ret = aclnnSliceGetWorkspaceSize(self, dim, start, end, step, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSlice第二段接口
  ret = aclnnSlice(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlice failed. ERROR: %d\n", ret); return ret);

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

