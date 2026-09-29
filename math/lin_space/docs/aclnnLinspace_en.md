# aclnnLinspace

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/lin_space)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Generates an evenly spaced value sequence. It creates a one-dimensional vector whose size is `steps`. The values are evenly distributed linearly from `start` to `end` (included).

- Formula:

$$
out = (start, start + \frac{end - start}{steps - 1},...,start + (steps - 2) * \frac{end - start}{steps -1}, end)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLinspaceGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLinspace` is called to perform computation.

- `aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLinspace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLinspaceGetWorkspaceSize

- **Parameters**

  * `start` (aclScalar*, computation input): start position of the value range to be obtained, input `start` in the formula, `aclScalar` on the host.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, BFLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, or BOOL.
     * <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, or INT32.

  * `end` (aclScalar*, computation input): end position of the value range to be obtained, input `end` in the formula, `aclScalar` on the host.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, BFLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, or BOOL.
     * <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, or INT32.

  * steps (int64_t, computation input): number of generated elements. The data type can be INT64. The value of steps must be greater than or equal to 0.

  * `out` (aclTensor *, computation output): specified output tensor, containing values that are linearly and evenly distributed from `start` to `end` (inclusive). In the formula, `out` is the `aclTensor` on the device side. [Data Format](../../../docs/en/context/data_format.md) supports ND, and the number of elements in `out` must be the same as that in `steps`. Empty tensors are not supported.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, BFLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, or INT32.
     * <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, or INT32.

  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. start, end, steps, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of out is not supported.
                                        2. steps is less than 0.
                                        3. The number of elements in out is inconsistent with that of steps.
  ```

## aclnnLinspace

- **Parameters**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.

  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLinspaceGetWorkspaceSize`.

  * `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnLinspace` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_linspace.h"

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

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> outShape = {8};
  void* outDeviceAddr = nullptr;
  aclScalar* start = nullptr;
  aclScalar* end = nullptr;
  aclScalar* steps = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float startValue = 0.0f;
  float endValue = 1.0f;
  int64_t stepsValue = 8;

  // Create a start aclScalar.
  start = aclCreateScalar(&startValue, aclDataType::ACL_FLOAT);
  CHECK_RET(start != nullptr, return ret);
  // Create an end aclScalar.
  end = aclCreateScalar(&endValue, aclDataType::ACL_FLOAT);
  CHECK_RET(end != nullptr, return ret);
  // Create a steps aclScalar.
  steps = aclCreateScalar(&stepsValue, aclDataType::ACL_INT64);
  CHECK_RET(steps != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLinspace.
  ret = aclnnLinspaceGetWorkspaceSize(start, end, stepsValue, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinspaceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnLinspace.
  ret = aclnnLinspace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinspace failed. ERROR: %d\n", ret); return ret);

  // 4. Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar.
  aclDestroyScalar(start);
  aclDestroyScalar(end);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
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
