# aclnnIsInScalarTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/equal)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |   ×    |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √    |

## Function

Checks whether the element in the element is equal to testElement.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnIsInScalarTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnIsInScalarTensor` is called to perform computation.

```Cpp
aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(
  const aclScalar*  element,
  const aclTensor*  testElements,
  bool              assumeUnique,
  bool              invert,
  aclTensor*        out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnIsInScalarTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnIsInScalarTensorGetWorkspaceSize

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>element (aclScalar*) </td>
      <td>Input</td>
      <td>Input scalar, element in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, BOOL, COMPLEX64, COMPLEX128</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>testElements (aclTensor*) </td>
      <td>Input</td>
      <td>Input tensor, testElements in the formula.</td>
      <td>The data type must comply with the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Data Type Derivation Rules</a>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, BOOL, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>assumeUnique (bool) </td>
      <td>Input</td>
      <td>Whether to assume that the element is unique, assumeUnique in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>invert (bool) </td>
      <td>Input</td>
      <td>Whether to negate, invert in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>Output tensor, out in the formula.</td>
      <td>The data type is BOOL. The shape is the same as that of testElements.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - For <term>Atlas training products</term> and <term>Atlas inference products</term>, BFLOAT16, COMPLEX64, and COMPLEX128 are not supported.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input element, testElements, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of element, testElements, or out is not supported.</td>
    </tr>
    <tr>
      <td>The number of dimensions of testElements and out exceeds 8.</td>
    </tr>
    <tr>
      <td>The shapes of testElements and out are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnIsInScalarTensor

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnIsInScalarTensorGetWorkspaceSize API.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnIsInScalarTensor is used.

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

  // Create the testElements tensor.
  auto ret = CreateAclTensor(
      testElementsHostData, testElementsShape, testElementsDeviceAddr, aclDataType::ACL_DOUBLE, testElements);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the element Scalar.
  *element = aclCreateScalar(&elementValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*element != nullptr, return ACL_ERROR_INVALID_PARAM);

  // Create the out tensor.
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

  // Synchronization.
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
