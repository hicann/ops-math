# aclnnLogdet

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/logdet)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Computes the natural logarithm of the determinant of input `self`.

- Formula:

  $$
  out = log(det(self))
  $$
  
  - If the result of `$det(self)$` is 0, `$out` equals `-inf$`.
  - If the result of `$det(self)$` is negative, `$out` equals `nan$`.

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLogdetGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLogdet` is called to perform computation.

- `aclnnStatus aclnnLogdetGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLogdet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLogdetGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): self in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.
    The shape must be in the `(*, n, n)` format, where `*` indicates the batch size with 0 or more dimensions. The shape cannot be greater than 8D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - `out` (aclTensor *, compute output): out in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128. The data type must meet the type deduction relationship with self. The shape must be the same as the batch of self. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.

  * `executor` (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown.
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                 2. The shape of self is less than two dimensions or greater than eight dimensions, or the sizes of the last two dimensions are different.
                                 3. The shape of out is different from the batch size of self.
```

## aclnnLogdet

- **Parameters:**

  - `workspace` (void *, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling first-phase API aclnnLogdetGetWorkspaceSize.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input parameter): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnLogdet` defaults to deterministic implementation.

The input data does not support the overflow value (Inf or NaN).

## Calling Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_logdet.h"

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
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, void** outDeviceAddr,
    aclTensor** self, aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<double> outHostData = {0, 0, 0};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_DOUBLE, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize, void* outDeviceAddr,
    std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnLogdetGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnLogdet(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogdet failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {3, 2, 2};
  std::vector<int64_t> outShape = {3};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(selfShape, outShape, &selfDeviceAddr, &outDeviceAddr, &self, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Destroy tensors.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // Release device memory.
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
