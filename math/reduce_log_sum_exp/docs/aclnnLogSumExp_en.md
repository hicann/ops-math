# aclnnLogSumExp

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/reduce_log_sum_exp)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Returns the log of summed exponents of the input tensor in a given dimension.
- Formula:
  In the formula, `i` is the dimension specified by `dim`, and `j` is the index of the input element in the specified dimension.
  
  $$
  logsumexp(x)_i = log\sum_{j} exp(x_{ij} )
  $$

- Examples

```text
Example 1:
  self: [2, 3, 4]       # self_shape=[2, 3, 4];
  dim: [2]              # dim_shape=[2], dim_data = {1, 2}, specifies the dimension.
  keepDim: false
  out: [2]              # out_shape=[2];

Example 2:
  self: [2, 3, 4]       # self_shape=[2, 3, 4];
  dim: [2]              # dim_shape=[2], dim_data = {1, 2}, specifies the dimension.
  keepDim: true
  out: [2, 1, 1]        # out_shape=[2, 1, 1];
```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLogSumExpGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLogSumExp` is called to perform computation.

- `aclnnStatus aclnnLogSumExpGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, bool keepDim, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnLogSumExp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnLogSumExpGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): `self` in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. The data type must be convertible to that of `out`. For details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, INT32, INT64, INT16, INT8, UINT8, or BOOL.

  - `dim` (aclIntArray*, compute input): dimension involved in the computation, `i` in the formula, aclIntArray on the host. The value range is [–self.dim(), self.dim() – 1]. The data type can be INT64.

  - `keepDim` (bool, compute input): whether to retain the dimension of the reduced axis, BOOL constant on the host.

  - `out` (aclTensor*, compute input): $logsumexp(x)$ in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. If `keepDim` is `true`, the shape must be the same as that of `self` except the dimension `dim` is of size 1. If `keepDim` is `false`, the shape must be the same as that of `self` except the reduced axis. The data type must be convertible to that of `self`. For details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): The passed self, out, or dim is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, out, or dim is not supported.
                                      2. The dimensions specified in the dim array exceed the range of dimensions of the input tensor.
                                      3. Duplicate elements exist in the dim array.
                                      4. The dimensions of self or out are greater than 8.
                                      5. The data type of self cannot be converted to that of out.
                                      6. The shape of out is not equal to the shape deduced from self, dim, and keepDim.
```

## aclnnLogSumExp

- **Parameters:**

  - `workspace` (void*, input): memory address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLogSumExpGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**
  - `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
## Constraints

- Deterministic computation
  - `aclnnLogSumExp` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_logsumexp.h"

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

  // Call aclrtMemcpy to copy host data to the device memory.
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

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {2, 2, 2};
  std::vector<int64_t> outShape = {2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dim = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {2, 3, 5, 8, 4, 12, 6, 7};
  std::vector<float> outHostData = {2, 3};
  std::vector<int64_t> dimData = {1, 2};
  bool keepdim = false;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a dim aclIntArray.
  ret = CreateAclIntArray(dimData, &dim);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLogSumExp.
  ret = aclnnLogSumExpGetWorkspaceSize(self, dim, keepdim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSumExpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnLogSumExp.
  ret = aclnnLogSumExp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSumExp failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor.
  aclDestroyTensor(self);
  aclDestroyIntArray(dim);
  aclDestroyTensor(out);

  // 7. Release device resources. Modification is required based on the specific API definition.
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
