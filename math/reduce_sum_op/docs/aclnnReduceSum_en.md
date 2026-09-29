# aclnnReduceSum

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/reduce_sum_op)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    √    |

## Function

Description: Returns the sum of each row in a specified dimension of the input tensor.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnReduceSumGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnReduceSum` is called to perform computation.

* `aclnnStatus aclnnReduceSumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, bool keepDims, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnReduceSum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnReduceSumGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): aclTensor on the device. The shape supports zero to eight dimensions. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, or COMPLEX128. If the input is an empty tensor, the output type cannot be COMPLEX64 or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16. If the input is an empty tensor, the output type cannot be COMPLEX64 or COMPLEX128.

  - `dims` (aclIntArray*, computation input): aclIntArray on the host, which specifies the reduced dimensions. The data type can be INT64. The value range is [–self.dim(), self.dim() – 1].

  - `keepDims` (bool, computation input): BOOL value on the host, which specifies whether to retain the dimensions of the input tensor in the output tensor.

  - `dtype` (aclDataType, computation input): aclDataType on the device, which specifies the data type of the output tensor.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.

  - `out` (aclTensor*, computation output): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed `self`, `dims`, or `out` is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of `self` or `out` is not supported.
                                     2. The shape of `out` does not match the actual shape.
                                     3. The dimensions specified in the `dims` array exceed the range of dimensions of the input tensor.
                                     4. The axes in the `dims` array are duplicate.
                                     5. The data type of `dtype` is not supported.
                                     6. The value of `dtype` is inconsistent with the data type of `out`.
```

## aclnnReduceSum

- **Parameters:**

  * `workspace` (void*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnReduceSumGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnReduceSum` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_sum.h"

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
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of contiguous tensors.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dims = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0};
  std::vector<int64_t> dimsData = {0};
  bool keepDims = false;
  auto dtype = aclDataType::ACL_FLOAT;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a dims aclIntArray.
  dims = aclCreateIntArray(dimsData.data(), dimsData.size());
  CHECK_RET(dims != nullptr, return ret);
  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnReduceSum.
  ret = aclnnReduceSumGetWorkspaceSize(self, dims, keepDims, dtype, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnReduceSum.
  ret = aclnnReduceSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceSum failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclIntArray. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(dims);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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
