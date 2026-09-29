# aclnnIsInScalarTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/equal)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Checks whether elements in `element` are equal to `testElement`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnIsInScalarTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnIsInScalarTensor` is called to perform computation.

+ `aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(const aclScalar *element, const aclTensor *testElements, bool assumeUnique, bool invert, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
+ `aclnnStatus aclnnIsInScalarTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnIsInScalarTensorGetWorkspaceSize

- **Parameters:**

  + `element` (aclScalar*, compute input): The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with testElements.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BFLOAT16.
     * <term>Atlas training products</term>: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, or UINT8.
  + `testElements` (aclTensor*, compute input): The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with element. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BFLOAT16.
     * <term>Atlas training products</term>: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, or UINT8.
  + `assumeUnique` (bool, compute input): indicates whether values in testElements are unique. If its value does not correctly indicate the uniqueness of elements in `testElements`, the result won't be affected.
  + `invert` (bool, compute input): indicates whether the output result is inverted.
  + `out` (aclTensor*, compute output): The data type can be BOOL. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  + `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  + `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. The passed element, testElements, or out is a null pointer.
  161002 ACLNN_ERR_PARAM_INVALID: 1. The data type of element or testElements is not supported.
                                  2. Data type deduction cannot be performed for element and `testElements`.
                                  3. element and testElements cannot be converted to the deduced data type.
                                  4. The data type of out is not BOOL.
                                  5. The dimensions of testElements are greater than eight.
                                  6. The dimension of out is not 0.
  ```

## aclnnIsInScalarTensor

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnIsInScalarTensorGetWorkspaceSize.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnQr` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isin.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& testElementsShape, std::vector<int64_t>& outShape, void** testElementsDeviceAddr,
    void** outDeviceAddr, aclTensor** testElements, aclTensor** out, aclScalar** element, bool& assumeUnique,
    bool& invert)
{
  std::vector<double> testElementsHostData = {1.0, 2.0, 3.0};
  std::vector<char> outHostData = {0};
  double elementValue = 4.0;

  // Create a testElements tensor.
  auto ret = CreateAclTensor(
      testElementsHostData, testElementsShape, testElementsDeviceAddr, aclDataType::ACL_DOUBLE, testElements);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an element Scalar.
  *element = aclCreateScalar(&elementValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*element != nullptr, return ACL_ERROR_INVALID_PARAM);

  // Create an output tensor.
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclScalar* element, aclTensor* testElements, bool assumeUnique, bool invert, aclTensor* out,
    void** workspaceAddrOut, uint64_t& workspaceSize, void* outDeviceAddr, aclrtStream stream)
{
  aclOpExecutor* executor;

  // First-phase API
  auto ret = aclnnIsInScalarTensorGetWorkspaceSize(
      element, testElements, assumeUnique, invert, out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate workspace.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Second-phase API
  ret = aclnnIsInScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInScalarTensor failed. ERROR: %d\n", ret); return ret);

  // Synchronization
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Output copy.
  char resultData = 0;
  ret = aclrtMemcpy(&resultData, sizeof(char), outDeviceAddr, sizeof(char), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("result is: %d\n", static_cast<bool>(resultData));

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> testElementsShape = {3};
  std::vector<int64_t> outShape = {};
  void* testElementsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* testElements = nullptr;
  aclTensor* out = nullptr;
  aclScalar* element = nullptr;

  bool assumeUnique = false;
  bool invert = true;

  ret = CreateInputs(
      testElementsShape, outShape, &testElementsDeviceAddr, &outDeviceAddr, &testElements, &out, &element, assumeUnique,
      invert);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret =
      ExecOpApi(element, testElements, assumeUnique, invert, out, &workspaceAddr, workspaceSize, outDeviceAddr, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release.
  aclDestroyScalar(element);
  aclDestroyTensor(testElements);
  aclDestroyTensor(out);

  aclrtFree(testElementsDeviceAddr);
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
