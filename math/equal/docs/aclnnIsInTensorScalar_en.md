# aclnnIsInTensorScalar

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

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnIsInTensorScalarGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnIsInTensorScalar` is called to perform computation.

+ `aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor *element, const aclScalar *testElement,bool assumeUnique, bool invert, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
+ `aclnnStatus aclnnIsInTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnIsInTensorScalarGetWorkspaceSize

- **Parameters:**

  * `element` (aclTensor*, compute input): The shape cannot exceed 8D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, BFLOAT16, FLOAT16, INT32, INT64, INT16, INT8, UINT8, or DOUBLE, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `testElement`.
    * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, or DOUBLE, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `testElement`.
  * `testElement` (aclScalar*, compute input): The data type must meet the [TensorScalar deduction relationship](../../../docs/en/context/TensorScalar_deduction_relationship.md) with `element`.
    * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, BFLOAT16, FLOAT16, INT32, INT64, INT16, INT8, UINT8, or DOUBLE.
    * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, or DOUBLE.
  * `assumeUnique` (bool, compute input): assumes that elements in element and testElement are unique when the value is True, to speed up computation.
  * `invert` (bool, compute input): indicates whether the output result needs to be inverted.
  * `out` (aclTensor*, compute output): The data type can be BOOL. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The shapes of out and element should be the same. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. The passed element, testElement, or out is a null pointer.
  161002 ACLNN_ERR_PARAM_INVALID: 1. The data type of element or testElement is not supported.
                                  2. Data type deduction cannot be performed for element and `testElement`.
                                  3. The deduced data types of element and testElement are not supported.
                                  4. The data type of out is not BOOL.
                                  5. The dimensions of element and out are greater than 8.
                                  6. The shape of out is different from that of element.
  ```

## aclnnIsInTensorScalar

- **Parameters:**

  + `workspace` (void*, input): address of the workspace to be allocated on the device.
  + `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnIsInTensorScalarGetWorkspaceSize.
  + `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  + `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnProdDim` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isin_tensor_scalar.h"

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
    std::vector<int64_t>& elementShape, std::vector<int64_t>& outShape, void** elementDeviceAddr, void** outDeviceAddr,
    aclTensor** element, aclTensor** out, aclScalar** testElement, bool& assumeUnique, bool& invert)
{
  std::vector<double> elementHostData = {0, 1, 2, 3, 2};
  std::vector<char> outHostData = {5, 0};
  double testElementValue = 2;

  // Create a testElement scalar.
  *testElement = aclCreateScalar(&testElementValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*testElement != nullptr, return ACL_ERROR_INVALID_PARAM);

  // Create an element tensor.
  auto ret = CreateAclTensor(elementHostData, elementShape, elementDeviceAddr, aclDataType::ACL_DOUBLE, element);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out tensor.
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* element, aclScalar* testElement, bool assumeUnique, bool invert, aclTensor* out, void** workspaceAddrOut,
    uint64_t& workspaceSize, void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // First-phase API
  auto ret =
      aclnnIsInTensorScalarGetWorkspaceSize(element, testElement, assumeUnique, invert, out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate workspace.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Second-phase API
  ret = aclnnIsInTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInTensorScalar failed. ERROR: %d\n", ret); return ret);

  // Synchronization
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output.
  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> elementShape = {5};
  std::vector<int64_t> outShape = {5};
  void* elementDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* element = nullptr;
  aclScalar* testElement = nullptr;
  aclTensor* out = nullptr;

  bool assumeUnique = false;
  bool invert = false;

  ret = CreateInputs(
      elementShape, outShape, &elementDeviceAddr, &outDeviceAddr, &element, &out, &testElement, assumeUnique, invert);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      element, testElement, assumeUnique, invert, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release.
  aclDestroyScalar(testElement);
  aclDestroyTensor(element);
  aclDestroyTensor(out);

  aclrtFree(elementDeviceAddr);
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
