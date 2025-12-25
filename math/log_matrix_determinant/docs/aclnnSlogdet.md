# aclnnSlogdet

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：计算输入self的行列式的符号和自然对数。

- 计算公式：

  $$
  signOut = sign(det(self))     \\
  logOut = log(abs(det(self)))
  $$

  其中det表示行列式计算，abs表示绝对值计算。 如果$det(self)$的结果是0，则$logOut = -inf$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSlogdetGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSlogdet”接口执行计算。

- `aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *self, aclTensor *signOut, aclTensor *logOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSlogdet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSlogdetGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*,计算输入): 公式中的`self`，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128。
    shape满足(\*, n, n)形式，其中`*`表示0或更多维度的batch, n表示任意正整数。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - signOut(aclTensor *，计算输出): 公式中的`signOut`，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128且需要和self满足推导关系，
  `self`为COMPLEX类型，不支持`signOut`为非COMPLEX类型。shape与self的batch一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  - logOut(aclTensor *，计算输出): 公式中的`logOut`，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128且需要和self满足推导关系，
  `self`为COMPLEX类型，不支持`logOut`为非COMPLEX类型。shape与self的batch一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。

  * executor(aclOpExecutor **，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self, signOut或logOut是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. self, signOut或logOut的数据类型和数据格式不在支持的范围之内。
                                 2. self的shape不满足约束。
                                 3. signOut和logOut的shape不满足约束。
```

## aclnnSlogdet

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnSlogdetGetWorkspaceSize获取。

  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSlogdet默认确定性实现。

输入数据中不支持存在溢出值Inf/Nan。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_slogdet.h"

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
  std::vector<int64_t> selfShape = {3, 2, 2};
  std::vector<int64_t> signOutShape = {3};
  std::vector<int64_t> logOutShape = {3};
  void* selfDeviceAddr = nullptr;
  void* signOutDeviceAddr = nullptr;
  void* logOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* signOut = nullptr;
  aclTensor* logOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> signOutHostData = {0, 0, 0};
  std::vector<float> logOutHostData = {0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建signOut aclTensor
  ret = CreateAclTensor(signOutHostData, signOutShape, &signOutDeviceAddr, aclDataType::ACL_FLOAT, &signOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logOut aclTensor
  ret = CreateAclTensor(logOutHostData, logOutShape, &logOutDeviceAddr, aclDataType::ACL_FLOAT, &logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSlogdet第一段接口
  ret = aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSlogdet第二段接口
  ret = aclnnSlogdet(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(signOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), signOutDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("signout result[%ld] is: %f\n", i, resultData[i]);
  }

  size = GetShapeSize(logOutShape);
  std::vector<float> logResultData(size, 0);
  ret = aclrtMemcpy(logResultData.data(), logResultData.size() * sizeof(logResultData[0]), logOutDeviceAddr,
                    size * sizeof(logResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("logout result[%ld] is: %f\n", i, logResultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(signOut);
  aclDestroyTensor(logOut);
  // 7. 释放device资源，需要根据具体API的接口定义参数
  aclrtFree(selfDeviceAddr);
  aclrtFree(signOutDeviceAddr);
  aclrtFree(logOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
