# aclnnEqTensor&aclnnInplaceEqTensor

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

- Description: Compares elements of two tensors element-wise; returns `True(1.)` if an element in `self` equals the corresponding element in `other`, and `False(0.)` otherwise.
- Formula:

  $$
  out = (self_i == other_i)  ?  [True] : [False]
  $$

## Prototype

- `aclnnEqTensor` and `aclnnInplaceEqTensor` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnEqTensor`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceEqTensor`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnEqTensorGetWorkspaceSize` or `aclnnInplaceEqTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnEqTensor` or `aclnnInplaceEqTensor` is called to perform computation.

```Cpp
aclnnStatus aclnnEqTensorGetWorkspaceSize(
  const aclTensor *self, 
  const aclTensor *other, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnEqTensor(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceEqTensorGetWorkspaceSize(
  const aclTensor *selfRef, 
  const aclTensor *other, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceEqTensor(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnEqTensorGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 168px">
  <col style="width: 136px">
  <col style="width: 215px">
  <col style="width: 286px">
  <col style="width: 340px">
  <col style="width: 116px">
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
      <td>Input tensor.</td>
      <td><ul><li>The data type must meet the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>) with <code>other</code>. </li><li>The shape must be broadcastable with that of <code>other</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>Must not exceed 8D.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td><ul><li>The data type must meet the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>) with <code>self</code>. </li><li>The shape must be broadcastable with that of <code>self</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>Must not exceed 8D.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td><ul><li>The data type must be convertible from BOOL (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>). </li><li>The shape must match the broadcast result of <code>self</code> and <code>other</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64, UINT32, UINT16</td>
      <td>ND</td>
      <td>Must not exceed 8D.</td>
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
    - The data types of `selfRef` and `other` cannot be UINT32 or UINT64.
    - The data type of `out` cannot be UINT64, UINT32, or UINT16.
  - <term>Atlas training products</term>:
    - The data type cannot be BFLOAT16.
    - The data type of `out` cannot be BFLOAT16, UINT64, UINT32, or UINT16.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 750px">
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
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The passed <code>self</code>, <code>other</code>, or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>self</code>, <code>other</code>, or <code>out</code> exceeds 8.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed for <code>self</code> and <code>other</code>.</td>
    </tr>
    <tr>
      <td>The shapes of <code>self</code> and <code>other</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with that of the broadcast result.</td>
    </tr>
  </tbody>
  </table>

## aclnnEqTensor

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnEqTensorGetWorkspaceSize</code>.</td>
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

## aclnnInplaceEqTensorGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1542px"><colgroup>
  <col style="width: 166px">
  <col style="width: 134px">
  <col style="width: 212px">
  <col style="width: 283px">
  <col style="width: 348px">
  <col style="width: 114px">
  <col style="width: 140px">
  <col style="width: 145px">
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
      <td>Input/output tensor, <code>self</code> and <code>out</code> in the formula.</td>
      <td><ul><li>The data type must meet the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>) with <code>other</code>. </li><li>The shape must be broadcastable with that of <code>other</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td><code>other</code> in the formula.</td>
      <td><ul><li>The data type must meet the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>) with <code>selfRef</code>. </li><li>The shape must be broadcastable with that of <code>selfRef</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
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
    - The data types of `selfRef` and `other` cannot be UINT32 or UINT64.
  - <term>Atlas training products</term>:
    - The data type cannot be BFLOAT16.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 750px">
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
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of <code>selfRef</code> or <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed for <code>selfRef</code> and <code>other</code>.</td>
    </tr>
    <tr>
      <td>The shapes of <code>selfRef</code> and <code>other</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td>The shape of <code>selfRef</code> is inconsistent with that of the broadcast result of <code>selfRef</code> and <code>other</code>.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>selfRef</code> and <code>other</code> exceeds 8.</td>
    </tr>
    <tr>
      <td>Exactly one of <code>selfRef</code> or <code>other</code> is an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceEqTensor

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnInplaceEqTensorGetWorkspaceSize</code>.</td>
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
  - `aclnnEqTensor` and `aclnnInplaceEqTensor` default to a deterministic implementation.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnEqTensor Sample Code**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_tensor.h"

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
    std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,
    void** selfDeviceAddr, void** otherDeviceAddr, void** outDeviceAddr, aclTensor** self, aclTensor** other,
    aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {0, 1, 9, 9, 9, 9, 9, 9};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(otherHostData, otherShape, otherDeviceAddr, aclDataType::ACL_DOUBLE, other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* other, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnEqTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnEqTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqTensor failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

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
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(
      selfShape, otherShape, outShape, &selfDeviceAddr, &otherDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release resources.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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

**aclnnInplaceEqTensor Sample Code**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_tensor.h"

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
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, void** selfDeviceAddr, void** otherDeviceAddr,
    aclTensor** self, aclTensor** other)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {1, 1, 3, 3, 5, 5, 7, 7};

  //Create a self tensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //Create an other tensor.
  ret = CreateAclTensor(otherHostData, otherShape, otherDeviceAddr, aclDataType::ACL_DOUBLE, other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* other, void** workspaceAddrOut, uint64_t& workspaceSize, void* selfDeviceAddr,
    std::vector<int64_t>& selfShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Obtain the workspace size.
  auto ret = aclnnInplaceEqTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate workspace.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Execute the operator.
  ret = aclnnInplaceEqTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqTensor failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output.
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(double),
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
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;

  ret = CreateInputs(selfShape, otherShape, &selfDeviceAddr, &otherDeviceAddr, &self, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, &workspaceAddr, workspaceSize, selfDeviceAddr, selfShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Destroy.
  aclDestroyTensor(self);
  aclDestroyTensor(other);

  // Release.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
