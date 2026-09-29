# aclnnSplitWithSize

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/split_v)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×      |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Splits the input self along the `dim` axis into the size of each element in `splitSize`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSplitWithSizeGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSplitWithSize` is called to perform computation.

- `aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor *self, const aclIntArray *splitSize, int64_t dim, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSplitWithSize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSplitWithSizeGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): input tensor to be split. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64. When the number of outputs is greater than 32, DOUBLE, COMPLEX128, or COMPLEX64 is not supported.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16. When the number of outputs is greater than 32, DOUBLE, COMPLEX128, or COMPLEX64 is not supported.
  - `splitSize` (aclIntArray*, computation input): size of each split block. The data type can be INT64 or INT32. The total size of all blocks must be equal to the shape size of `self` in the `dim` dimension.

  - `dim` (int64_t, computation input): dimension along which the input tensor is split. The data type can be INT64.

  - `out` (aclTensorList*, computation output): output tensor list after splitting. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64. When the number of outputs is greater than 32, DOUBLE, COMPLEX128, or COMPLEX64 is not supported.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16. When the number of outputs is greater than 32, DOUBLE, COMPLEX128, or COMPLEX64 is not supported.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

​   `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown.
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, splitSize, or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                  2. The length of self is not supported.
                                  3. The tensor length in out is not supported.
                                  4. The value of dim is out of range [-dimNum, dimNum - 1]. dimNum is the dimension size of self.
                                  5. The sum of elements in splitSize is not equal to the shape size of the split dimension.
```

## aclnnSplitWithSize

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnSplitWithSizeGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnSplitWithSize` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_split_with_size.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data from the host to the device.
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

void CheckResult(const std::vector<std::vector<int64_t>> &shapeList, const std::vector<void *> addrList) {
  for (size_t i = 0; i < shapeList.size(); i++) {
    auto size = GetShapeSize(shapeList[i]);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), addrList[i],
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return);
    for (int64_t j = 0; j < size; j++) {
      LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
    }
  }
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {5, 2};
  std::vector<int64_t> shape1 = {1, 2};
  std::vector<int64_t> shape2 = {4, 2};
  int64_t splitValue[] = {1, 4};
  int64_t dim = 0;

  void* selfDeviceAddr = nullptr;
  void* shape1DeviceAddr = nullptr;
  void* shape2DeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* shape1Addr = nullptr;
  aclTensor* shape2Addr = nullptr;
  aclIntArray *splitSize = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> shape1HostData = {0, 5};
  std::vector<float> shape2HostData = {1, 2, 3, 4, 6, 7, 8, 9};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  splitSize = aclCreateIntArray(splitValue, 2);
  CHECK_RET(splitSize != nullptr, return ret);

  ret = CreateAclTensor(shape1HostData, shape1, &shape1DeviceAddr, aclDataType::ACL_FLOAT, &shape1Addr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(shape2HostData, shape2, &shape2DeviceAddr, aclDataType::ACL_FLOAT, &shape2Addr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensorList.
  std::vector<aclTensor*> tmp = {shape1Addr, shape2Addr};
  aclTensorList* out = aclCreateTensorList(tmp.data(), tmp.size());
  CHECK_RET(out != nullptr, return ret);

  // Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  // Call the first-phase API of aclnnSplitWithSize.
  ret = aclnnSplitWithSizeGetWorkspaceSize(self, splitSize, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSplitWithSizeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSplitWithSize
  ret = aclnnSplitWithSize(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSplitWithSize failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CheckResult({shape1, shape2}, {shape1DeviceAddr, shape2DeviceAddr});

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(splitSize);
  aclDestroyTensorList(out);
  aclDestroyTensor(shape1Addr);
  aclDestroyTensor(shape2Addr);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(shape1DeviceAddr);
  aclrtFree(shape2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
