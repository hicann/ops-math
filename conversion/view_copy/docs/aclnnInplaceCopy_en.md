# aclnnInplaceCopy

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/view_copy)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Copies elements in `src` to the `selfRef` tensor and returns `selfRef`.

- Formula:

  $$
  {selfRef}_{i} = {src}_{i}
  $$

- Example:

  ```text
  Input selfRef:
  tensor([[1, 2],
          [3, 4]])
  Input src:
  tensor([[5, 6],
          [7, 8]])
  
  Output selfRef:
  tensor([[5, 6],
          [7, 8]])
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnInplaceCopyGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnInplaceCopy` is called to perform computation.

- `aclnnStatus aclnnInplaceCopyGetWorkspaceSize(aclTensor *selfRef, const aclTensor *src, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnInplaceCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnInplaceCopyGetWorkspaceSize

- **Parameters:**

  - `selfRef` (aclTensor*, computation input | computation output): `selfRef` in the formula. The copy between complex numbers is supported only when `selfRef` is contiguous. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `src`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT32, BOOL, DOUBLE, COMPLEX64, COMPLEX128, UINT16, UINT32, UINT64 or BFLOAT16.
    - <term>Atlas training products</term>, <term>Atlas inference products</term>, and <term>Atlas 200I/500 A2 inference products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT32, BOOL, DOUBLE, COMPLEX64, COMPLEX128, UINT16, UINT32 or UINT64.
  - `src` (aclTensor*, computation input): `src` in the formula. The copy between complex numbers is supported only when `selfRef` is contiguous. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT32, BOOL, DOUBLE, COMPLEX64, COMPLEX128, UINT16, UINT32, UINT64 or BFLOAT16.
    - <term>Atlas training products</term>, <term>Atlas inference products</term>, and <term>Atlas 200I/500 A2 inference products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT32, BOOL, DOUBLE, COMPLEX64, COMPLEX128, UINT16, UINT32 or UINT64.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. selfRef or src is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or src is not supported.
                                    2. The shape of selfRef exceeds 8 dimensions.
                                    3. The shape of src cannot be broadcast to selfRef.
                                    4. The data type of src is not supported or cannot be converted to selfRef.
  ```

## aclnnInplaceCopy

- **Parameters:**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceCopyGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnInplaceCopy` defaults to a deterministic implementation.

When the data types of `src` and `selfRef` are inconsistent, the conversion rules for floating-point types (HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN) are as follows:

 - HIFLOAT8 -> FLOAT32, BFLOAT16, FLOAT16
 - FLOAT8_E5M2 -> FLOAT32, BFLOAT16, FLOAT16
 - FLOAT8_E4M3FN -> FLOAT32, BFLOAT16, FLOAT16
 - BFLOAT16 -> HIFLOAT8
 - FLOAT16 -> HIFLOAT8
 - FLOAT32 -> HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_copy.h"

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> srcShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* src = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> srcHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceCopy.
  ret = aclnnInplaceCopyGetWorkspaceSize(selfRef, src, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnInplaceCopy.
  ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(src);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(srcDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
