# aclnnAdd&aclnnInplaceAdd

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/add)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Function: Performs the addition calculation.
- Formula:

  $$
  out_i = self_i+alpha \times other_i
  $$

## Prototype

- `aclnnAdd` and `aclnnInplaceAdd` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnAdd`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceAdd`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAddGetWorkspaceSize` or `aclnnInplaceAddGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnAdd` or `aclnnInplaceAdd` is called to perform computation.

  ```Cpp
  aclnnStatus aclnnAddGetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* other, 
    const aclScalar* alpha, 
    aclTensor*       out, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnAdd(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddGetWorkspaceSize(
    const aclTensor* selfRef, 
    const aclTensor* other, 
    const aclScalar* alpha, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAdd(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
  ```

## aclnnAddGetWorkspaceSize

 **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1496px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 205px">
  <col style="width: 305px">
  <col style="width: 317px">
  <col style="width: 121px">
  <col style="width: 134px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input <code>self</code> in the formula.</td>
      <td>
        <ul>
          <li>Its data type and the data type of <code>other</code> must follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shapes of <code>self</code> and <code>other</code> must meet the broadcast relationship (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>).</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
      <td>ND</td>
      <td>The dimensions cannot exceed 8.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input <code>other</code> in the formula.</td>
      <td>
        <ul>
          <li>Its data type and the data type of <code>self</code> must follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shapes of <code>other</code> and <code>self</code> must meet the broadcast relationship (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>).</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
      <td>ND</td>
      <td>The dimensions cannot exceed 8.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td><code>alpha</code> in the formula.</td>
      <td>Its data type can be cast to the type promoted from <code>self</code> and <code>other</code>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td><code>out</code> in the formula.</td>
      <td>
        <ul>
          <li>The type promoted from <code>self</code> and <code>other</code> can be cast to the data type of <code>out</code> (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
          <li>Its shape must be the shape obtained by broadcasting <code>self</code> and <code>other</code>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
      <td>ND</td>
      <td>The dimensions cannot exceed 8.</td>
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
      <td>Operator executor, covering the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas training products</term>: The data type cannot be BFLOAT16.

 **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 257px">
  <col style="width: 125px">
  <col style="width: 772px">
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
      <td><code>self</code>, <code>other</code>, <code>alpha</code>, or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of <code>self</code> or <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td>Type promotion between <code>self</code> and <code>other</code> cannot be performed.</td>
    </tr>
    <tr>
      <td>The promoted data type cannot be cast to the specified type of <code>out</code>.</td>
    </tr>
    <tr>
      <td>The shapes of <code>self</code> and <code>other</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td>The data shape of <code>alpha</code> cannot be cast to the data type promoted from <code>self</code> and <code>other</code>.</td>
    </tr>
    <tr>
      <td>The shape of <code>out</code> is not the shape obtained by broadcasting <code>self</code> and <code>other</code></td>
    </tr>
    <tr>
      <td><code>self</code>, <code>other</code>, and <code>out</code> each have more than 8 dimensions.</td>
    </tr>
  </tbody>
  </table>

## aclnnAdd

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1045px"><colgroup>
  <col style="width: 148px">
  <col style="width: 125px">
  <col style="width: 772px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnAddGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, covering the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1496px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 205px">
  <col style="width: 305px">
  <col style="width: 317px">
  <col style="width: 121px">
  <col style="width: 134px">
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
      <td>Input/output tensor, that is, <code>self</code> and <code>out</code> in the formula.</td>
      <td>
        <ul>
          <li>Its shape must be broadcast-compatible with <code>other</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>). The resultant shape after broadcasting must match the shape of <code>selfRef</code>.</li>
          <li>The data types of <code>selfRef</code> and <code>other</code> must follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>). The promoted type can be cast to the data type of <code>selfRef</code> (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
      <td>ND</td>
      <td>The dimensions cannot exceed 8.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input <code>other</code> in the formula.</td>
      <td>
        <ul>
          <li>Its data type and the data type of <code>selfRef</code> must follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>Its shape must be broadcast-compatible with <code>selfRef</code> (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>).</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
      <td>ND</td>
      <td>The dimensions cannot exceed 8.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td><code>alpha</code> in the formula.</td>
      <td>Its data type can be cast to the type promoted from <code>selfRef</code> and <code>other</code>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16</td>
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
      <td>Operator executor, covering the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: The data type cannot be BFLOAT16.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1157px"><colgroup>
  <col style="width: 258px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>The input <code>selfRef</code>, <code>other</code>, or <code>alpha</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of <code>selfRef</code> or <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td>Type promotion between <code>selfRef</code> and <code>other</code> cannot be performed.</td>
    </tr>
    <tr>
      <td>The promoted type cannot be cast to the type of <code>selfRef</code>.</td>
    </tr>
    <tr>
      <td>The shapes of <code>selfRef</code> and <code>other</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td>The resultant shape after broadcasting is not the same as that of <code>selfRef</code>.</td>
    </tr>
    <tr>
      <td>The data shape of <code>alpha</code> cannot be cast to the data type promoted from <code>selfRef</code> and <code>other</code>.</td>
    </tr>
    <tr>
      <td><code>selfRef</code> and <code>other</code> each have more than 8 dimensions.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAdd

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1045px"><colgroup>
  <col style="width: 148px">
  <col style="width: 125px">
  <col style="width: 772px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnInplaceAddGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, covering the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnAdd` and `aclnnInplaceAdd` each default to a deterministic implementation.

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

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
  // Boilerplate code for resource initialization.
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
  // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
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
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 1.2f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // aclnnAdd API call example 
  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnAdd.
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAdd.
  ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

    
  // aclnnInplaceAdd API call example 
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceAdd\n");
  // Call the first-phase API of aclnnInplaceAdd.
  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAdd.
  ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdd failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }  
     
    
  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. Free device resources. Modify the code based on the API definition.
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
