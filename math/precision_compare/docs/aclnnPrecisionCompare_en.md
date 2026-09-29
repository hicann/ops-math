# aclnnPrecisionCompare

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/precision_compare)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Operator function: Compares the precisions of two tensors by checking whether they have the same elements. This will return the operator execution status code and will not trigger fault reporting of the management plane.

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnPrecisionCompareGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnPrecisionCompare is called to perform computation.

* `aclnnStatus aclnnPrecisionCompareGetWorkspaceSize(const aclTensor *golden, const aclTensor *realdata, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnPrecisionCompare(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnPrecisionCompareGetWorkspaceSize

- **Parameters:**

  * `golden` (aclTensor*, computation input): aclTensor on the device. The data type can be FLOAT16, FLOAT, or BFLOAT16, and must be the same as that of realdata. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape supports 0 to 8 dimensions and must be the same as that of `realdata`.
  * `realdata` (aclTensor*, compute input): aclTensor on the device. The data type can be FLOAT16, FLOAT, or BFLOAT16, and must be the same as that of golden. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports 0 to 8 dimensions and must be the same as that of `golden`.
  * `out` (aclTensor*, compute output): aclTensor on the device. The data type can be UINT32, the [data format](../../../docs/en/context/data_format.md) can be ND, and the shape can be 0-dimensional.

  ```text
  0: success.
  1: input parameter verification failure of the precision comparison operator.
  2: execution failure of the precision comparison operator.
  503: precision comparison exception, with no fault reported from the management plane.
  ```

  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown.
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input golden, realdata, or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of golden, realdata, or out is not supported.
                                    2. The data types of golden and realdata are different.
                                    3. The shape of golden or realdata is greater than 8D.
                                    4. The shapes of golden and realdata are different.
                                    5. The shape of out is not 0D.
```

## aclnnPrecisionCompare

- **Parameters:**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  * `workspaceSize`(uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnPrecisionCompareGetWorkspaceSize.
  * `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPrecisionCompare defaults to deterministic implementation.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_precision_compare.h"

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
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> goldenShape = {4, 2};
  std::vector<int64_t> realdataShape = {4, 2};
  std::vector<int64_t> outShape = {};
  void* goldenDeviceAddr = nullptr;
  void* realdataDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* golden = nullptr;
  aclTensor* realdata = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> goldenHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> realdataHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint32_t> outHostData = {0};
  // Create a self aclTensor.
  ret = CreateAclTensor(goldenHostData, goldenShape, &goldenDeviceAddr, aclDataType::ACL_FLOAT, &golden);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(realdataHostData, realdataShape, &realdataDeviceAddr, aclDataType::ACL_FLOAT, &realdata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPrecisionCompare.
  ret = aclnnPrecisionCompareGetWorkspaceSize(golden, realdata, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPrecisionCompareGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnPrecisionCompare.
  ret = aclnnPrecisionCompare(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPrecisionCompare failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(golden);
  aclDestroyTensor(realdata);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(goldenDeviceAddr);
  aclrtFree(realdataDeviceAddr);
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
