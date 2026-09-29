# aclnnDiv&aclnnInplaceDiv

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/div)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Performs division.

- Formula:

  $$
  out_i = \frac{input_i}{other_i}
  $$

- Note: In complex-number operations, if the modulus of the denominator is 0, the general complex-number formula is used without special processing. The behavior is different from that of the CPU/GPU.

## Prototype

- `aclnnDiv` and `aclnnInplaceDiv` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnDiv`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceDiv`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnDivGetWorkspaceSize` or `aclnnInplaceDivGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnDiv` or `aclnnInplaceDiv` is called to perform computation.
  
```Cpp
aclnnStatus aclnnDivGetWorkspaceSize(
  const aclTensor *self, 
  const aclTensor *other, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDiv(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceDivGetWorkspaceSize(
  aclTensor       *selfRef, 
  const aclTensor *other, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceDiv(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDivGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1531px"><colgroup>
  <col style="width: 155px">
  <col style="width: 125px">
  <col style="width: 214px">
  <col style="width: 337px">
  <col style="width: 302px">
  <col style="width: 115px">
  <col style="width: 136px">
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
      <td><code>input</code> in the formula.</td>
      <td><ul><li>Its data type and the data type of <code>other</code> must meet the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>). </li><li>The shape must be broadcastable with that of <code>other</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input <code>other</code> in the formula.</td>
      <td><ul><li>Its data type and the data type of <code>self</code> must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>). </li><li>The shape must be broadcastable with that of <code>self</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td><code>out</code> in the formula.</td>
      <td>The data type must be convertible from the deduced data type of <code>self</code> and <code>other</code>. The shape must be the broadcast result of <code>self</code> and <code>other</code>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
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

  - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type cannot be BFLOAT16.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 270px">
  <col style="width: 124px">
  <col style="width: 755px">
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
      <td>The data type or format of <code>self</code> or <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td><code>self</code> and <code>other</code> do not meet the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to that of <code>out</code>.</td>
    </tr>
    <tr>
      <td>The shapes of <code>self</code> and <code>other</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>self</code> or <code>other</code> exceeds 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnDiv

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnDivGetWorkspaceSize</code>.</td>
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

## aclnnInplaceDivGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1530px"><colgroup>
  <col style="width: 155px">
  <col style="width: 125px">
  <col style="width: 214px">
  <col style="width: 336px">
  <col style="width: 302px">
  <col style="width: 115px">
  <col style="width: 136px">
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
      <td>selfRef</td>
      <td>Input/Output</td>
      <td><code>input</code> and <code>out</code> in the formula.</td>
      <td><ul><li>Its data type and the data type of <code>other</code> must meet the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>). </li><li>The data type must be convertible from the deduced data type of <code>selfRef</code> and <code>other</code> (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>. </li><li>The shape must be broadcastable with that of <code>other</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input <code>other</code> in the formula.</td>
      <td><ul><li>Its data type and the data type of <code>selfRef</code> must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>). </li><li>The shape must be broadcastable with that of <code>selfRef</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
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

  - <term>Atlas training products</term>: The data type cannot be BFLOAT16.
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 270px">
  <col style="width: 124px">
  <col style="width: 755px">
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
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type or format of <code>selfRef</code> or <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td><code>selfRef</code> and <code>other</code> do not meet the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The shapes of <code>selfRef</code> and <code>other</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>selfRef</code> or <code>other</code> exceeds 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceDiv

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnInplaceDivGetWorkspaceSize</code>.</td>
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
  - `aclnnDiv` and `aclnnInplaceDiv` default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_div.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnDiv API call example
  LOG_PRINT("test aclnnDiv\n");

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // Call the first-phase API of aclnnDiv.
  ret = aclnnDivGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDiv.
  ret = aclnnDiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDiv failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceDiv API call example
  LOG_PRINT("\ntest aclnnInplaceDiv\n");

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // Call the first-phase API of aclnnInplaceDiv.
  ret = aclnnInplaceDivGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceDiv.
  ret = aclnnInplaceDiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDiv failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources.
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
