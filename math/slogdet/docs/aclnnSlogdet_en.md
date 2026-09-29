# aclnnSlogdet

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/slogdet)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     ×     |
| <term>Atlas inference products</term>                            |   √     |
| <term>Atlas training products</term>                             |   √     |

## Function

- Description: Computes the symbol and natural logarithm of the determinant of input `self`.

- Formula:

  $$
  signOut = sign(det(self))     \\
  logOut = log(abs(det(self)))
  $$

  `det` indicates determinant computation and `abs` indicates absolute value computation. If the result of `$det(self)$` is `0`, `$logOut` equals `-inf$`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSlogdetGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSlogdet` is called to perform computation.

- `aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *self, aclTensor *signOut, aclTensor *logOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSlogdet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSlogdetGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): `self` in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.
    The shape must be in the (\*, n, n) format. `*` indicates the batch size of zero or more dimensions, and n indicates any positive integer. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `signOut` (aclTensor *, compute output): `signOu`t in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128 and must meet the type deduction relationship with `self`.
  `self` is of the COMPLEX type. `signOut` cannot be of a non-COMPLEX type. The shape is the same as batch of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `logOut` (aclTensor *, compute output): `logOut` in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128 and must meet the deduction relationship with `self`.
  `self` is of the COMPLEX type. `logOut` cannot be of a non-COMPLEX type. The shape is the same as batch of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.

  * `executor` (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, signOut, or logOut is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self, signOut, or logOut is not supported.
                                 2. The shape of self does not meet constraints.
                                 3. The shapes of signOut and logOut do not meet constraints.
```

## aclnnSlogdet

- **Parameters:**

  - `workspace` (void *, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnSlogdetGetWorkspaceSize`.

  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnSlogdet` defaults to a deterministic implementation.

The input data does not support the overflow value (Inf or NaN).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  // (Boilerplate) Initialize resources.
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& signOutShape, std::vector<int64_t>& logOutShape,
    void** selfDeviceAddr, void** signOutDeviceAddr, void** logOutDeviceAddr, aclTensor** self, aclTensor** signOut,
    aclTensor** logOut)
{
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> signOutHostData = {0, 0, 0};
  std::vector<float> logOutHostData = {0, 0, 0};

  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a signOut aclTensor.
  ret = CreateAclTensor(signOutHostData, signOutShape, signOutDeviceAddr, aclDataType::ACL_FLOAT, signOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a logOut aclTensor.
  ret = CreateAclTensor(logOutHostData, logOutShape, logOutDeviceAddr, aclDataType::ACL_FLOAT, logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* signOut, aclTensor* logOut, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* signOutDeviceAddr, void* logOutDeviceAddr, std::vector<int64_t>& signOutShape,
    std::vector<int64_t>& logOutShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSlogdet.
  auto ret = aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the second-phase API of aclnnSlogdet.
  ret = aclnnSlogdet(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

  // Synchronize
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy signOut.
  auto sizeSign = GetShapeSize(signOutShape);
  std::vector<float> resultData(sizeSign, 0);
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), signOutDeviceAddr, sizeSign * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < sizeSign; i++) {
    LOG_PRINT("signout result[%ld] is: %f\n", i, resultData[i]);
  }

  // Copy logOut.
  auto sizeLog = GetShapeSize(logOutShape);
  std::vector<float> logResultData(sizeLog, 0);
  ret = aclrtMemcpy(
      logResultData.data(), logResultData.size() * sizeof(logResultData[0]), logOutDeviceAddr,
      sizeLog * sizeof(logResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < sizeLog; i++) {
    LOG_PRINT("logout result[%ld] is: %f\n", i, logResultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  // 1. Initialize the device and stream.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 2. Construct the input and output.
  std::vector<int64_t> selfShape = {3, 2, 2};
  std::vector<int64_t> signOutShape = {3};
  std::vector<int64_t> logOutShape = {3};

  void* selfDeviceAddr = nullptr;
  void* signOutDeviceAddr = nullptr;
  void* logOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* signOut = nullptr;
  aclTensor* logOut = nullptr;

  ret = CreateInputs(
      selfShape, signOutShape, logOutShape, &selfDeviceAddr, &signOutDeviceAddr, &logOutDeviceAddr, &self, &signOut,
      &logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator API.
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      self, signOut, logOut, &workspaceAddr, workspaceSize, signOutDeviceAddr, logOutDeviceAddr, signOutShape,
      logOutShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 6. Release aclTensor.
  aclDestroyTensor(self);
  aclDestroyTensor(signOut);
  aclDestroyTensor(logOut);

  // 7. Release device resources.
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
