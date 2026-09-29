# aclnnHardtanh&aclnnInplaceHardtanh

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/clip_by_value_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Limits all input elements within the range of [clipValueMin,clipValueMax]. If an element is greater than clipValueMax, the element is limited to clipValueMax. If an element is less than clipValueMin, the element is limited to clipValueMin. Otherwise, the element itself is used.
- Formula:

  $$
  HardTanh(x) = \left\{\begin{matrix}\begin{array}{l} 
  clipValueMax, \ if\ x>clipValueMax \\
  clipValueMin, \ if\ x<clipValueMin \\ 
  x, \ otherwise \\\end{array}\end{matrix}\right.\begin{array}{l}\end{array}
  $$
  
## Prototype

- `aclnnHardtanh` and `aclnnInplaceHardtanh` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnHardtanh`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceHardtanh`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnHardtanhGetWorkspaceSize` or `aclnnInplaceHardtanhGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnHardtanh` or `aclnnInplaceHardtanh` is called to perform computation.

  - `aclnnStatus aclnnHardtanhGetWorkspaceSize(const aclTensor *self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnHardtanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceHardtanhGetWorkspaceSize(aclTensor *selfRef, const aclScalar* clipValueMin, const aclScalar* clipValueMax, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceHardtanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnHardtanhGetWorkspaceSize

- **Parameters:**
  - `self` (aclTensor*, computation input): `x` in the formula, aclTensor on the device. The data type and shape must be the same as those of `out`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `clipValueMin` (aclScalar*, computation input): aclScalar on the host, lower bound. The data type must be convertible to that of `self`. If `clipValueMin` and `clipValueMax` coexist, they must be the same.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `clipValueMax` (aclScalar*, computation input): aclScalar on the host, upper bound. The data type must be convertible to that of `self`. If `clipValueMin` and `clipValueMax` coexist, they must be the same.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `out` (aclTensor\*, computation output): aclTensor on the device. The data type and shape must be the same as those of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, out, clipValueMax, or clipValueMin is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                      2. The data types of self and out are inconsistent.
                                      3. The shapes of self and out are inconsistent.
                                      4. The value of clipValueMin is greater than that of clipValueMax.
  ```

## aclnnHardtanh

- **Parameters:**
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnHardtanhGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceHardtanhGetWorkspaceSize

- **Parameters:**
  - `selfRef` (aclTensor\*, computation input): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `clipValueMin` (aclScalar*, computation input): aclScalar on the host, lower bound. The data type must be convertible to the data type of `selfRef`. If `clipValueMin` and `clipValueMax` coexist, they must be the same.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `clipValueMax` (aclScalar*, computation input): aclScalar on the host, upper bound. The data type must be convertible to that of `selfRef`. If `clipValueMin` and `clipValueMax` coexist, they must be the same.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT16, BFLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64 is supported.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter validation. The following error codes may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef, clipValueMax, or clipValueMin is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef is not supported.
                                      2. The value of clipValueMin is greater than that of clipValueMax.
```

## aclnnInplaceHardtanh

- **Parameters:**
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceHardtanhGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

None

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_hardtanh.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* clipValueMin = nullptr;
  aclScalar* clipValueMax = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float clipValueMinValue = 1.2f;
  float clipValueMaxValue = 2.4f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a clipValueMin aclScalar.
  clipValueMin = aclCreateScalar(&clipValueMinValue, aclDataType::ACL_FLOAT);
  CHECK_RET(clipValueMin != nullptr, return ret);
  // Create a clipValueMax aclScalar.
  clipValueMax = aclCreateScalar(&clipValueMaxValue, aclDataType::ACL_FLOAT);
  CHECK_RET(clipValueMax != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnHardtanh API call example
  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnHardtanh.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnHardtanhGetWorkspaceSize(self, clipValueMin, clipValueMax, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardtanhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnHardtanh.
  ret = aclnnHardtanh(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardtanh failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
 
  // Example of calling aclnnInplaceHardtanh
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceHardtanh\n");
  // Call the first-phase API of aclnnInplaceHardtanh.
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  ret = aclnnInplaceHardtanhGetWorkspaceSize(self, clipValueMin, clipValueMax, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceHardtanhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnInplaceHardtanh.
  ret = aclnnInplaceHardtanh(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceHardtanh failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  size = GetShapeSize(outShape);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(clipValueMin);
  aclDestroyScalar(clipValueMax);
  aclDestroyTensor(out);
 
  // 7. Release device resources. Modify the configuration based on the API definition.
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
