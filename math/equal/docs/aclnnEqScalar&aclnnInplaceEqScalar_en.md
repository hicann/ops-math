# aclnnEqScalar&aclnnInplaceEqScalar

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

- Description: Compares the values of elements in `self` with those in `other`, and writes the comparison result to `out`.
- Formula:

  $$
  out_i = (self_i == \mathit{other} )  ?  [True] : [False]
  $$

## Prototype

- `aclnnEqScalar` and `aclnnInplaceEqScalar` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnEqScalar`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceEqScalar`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnEqScalarGetWorkspaceSize` or `aclnnInplaceEqScalarGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnEqScalar` or `aclnnInplaceEqScalar` is called to perform computation.

```Cpp
aclnnStatus aclnnEqScalarGetWorkspaceSize(
  const aclTensor *self, 
  const aclScalar *other, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnEqScalar(
  void*             workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor*    executor, 
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceEqScalarGetWorkspaceSize(
  const aclTensor *selfRef, 
  const aclScalar *other, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceEqScalar(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnEqScalarGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 168px">
  <col style="width: 136px">
  <col style="width: 213px">
  <col style="width: 200px">
  <col style="width: 373px">
  <col style="width: 115px">
  <col style="width: 142px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td><code>self</code> in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>Must not exceed 8D.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td><code>other</code> in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td><code>out</code> in the formula.</td>
      <td>The data type must be convertible from BOOL (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64, UINT32, UINT16</td>
      <td>ND</td>
      <td>The shape must be identical to that of <code>self</code>.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - The data types of `self` and `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - The data types of `self` and `other` cannot be UINT32 or UINT64.
    - The data type of `out` cannot be UINT64, UINT32, or UINT16.
  - <term>Atlas training products</term>:
    - The data types of `self` and `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - The data type cannot be BFLOAT16.
    - The data type of `out` cannot be UINT64, UINT32, UINT16, or BFLOAT16.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.
 
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed <code>self</code>, <code>other</code>, or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of <code>self</code>, <code>other</code>, or <code>out</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data types of <code>self</code> and <code>other</code> do not meet the type deduction rules.</td>
    </tr>
    <tr>
      <td>The shape of <code>self</code> is different from that of <code>out</code>.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>self</code> or <code>out</code> exceeds 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnEqScalar

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnEqScalarGetWorkspaceSize</code>.</td>
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
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceEqScalarGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1502px"><colgroup>
  <col style="width: 167px">
  <col style="width: 135px">
  <col style="width: 267px">
  <col style="width: 159px">
  <col style="width: 372px">
  <col style="width: 115px">
  <col style="width: 141px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>Input/Output</td>
      <td><code>selfRef</code> in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>Must not exceed 8D.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td><code>other</code> in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - The data types of `selfRef` and `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - The data types of `selfRef` and `other` cannot be UINT32 or UINT64.
  - <term>Atlas training products</term>:
    - The data types of `selfRef` and `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - The data type cannot be BFLOAT16.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed <code>selfRef</code> or <code>other</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of <code>selfRef</code> or <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data types of <code>selfRef</code> and <code>other</code> do not meet the type deduction rules.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>selfRef</code> exceeds 8D.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceEqScalar

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnInplaceEqScalarGetWorkspaceSize</code>.</td>
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
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnEqScalar` and `aclnnInplaceEqScalar` default to a deterministic implementation.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnEqScalar Sample Code**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_scalar.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides of the contiguous tensor.
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
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, void** outDeviceAddr,
    aclTensor** self, aclScalar** other, aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 1.2, 0.3, 4.1, 5, 1.6, 7};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  double otherValue = 1.2;

  // Create a self tensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an other scalar.
  *other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*other != nullptr, return ret);

  // Create an out tensor.
  ret = CreateAclTensor(outHostData, outShape, &(*outDeviceAddr), aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclScalar* other, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Obtain the workspace size.
  auto ret = aclnnEqScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate the workspace (released in main).
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the operator.
  ret = aclnnEqScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqScalar failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output.
  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(char),
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(selfShape, outShape, &selfDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release tensors and scalars.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // Release device memory.
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

**aclnnInplaceEqScalar Sample Code**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_scalar.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides of the contiguous tensor.
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(std::vector<int64_t>& selfShape, void** selfDeviceAddr, aclTensor** self, aclScalar** other)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  double otherValue = 2.0;

  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an other aclScalar.
  *other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*other != nullptr, return ACL_ERROR_INVALID_PARAM);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclScalar* other, void* selfDeviceAddr, std::vector<int64_t>& selfShape, aclrtStream stream,
    void** workspaceAddrOut)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  auto ret = aclnnInplaceEqScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate workspace.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  *workspaceAddrOut = workspaceAddr;

  // Execute.
  ret = aclnnInplaceEqScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqScalar failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output.
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclScalar* other = nullptr;

  ret = CreateInputs(selfShape, &selfDeviceAddr, &self, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  ret = ExecOpApi(self, other, selfDeviceAddr, selfShape, stream, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Destroy.
  aclDestroyTensor(self);
  aclDestroyScalar(other);

  aclrtFree(selfDeviceAddr);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  // Release.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
