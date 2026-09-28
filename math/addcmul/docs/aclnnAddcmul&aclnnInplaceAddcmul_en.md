# aclnnAddcmul&aclnnInplaceAddcmul

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/addcmul)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Performs element-wise multiplication of `tensor1` and `tensor2`, multiplies the result by the scalar `value`, and then performs element-wise addition of this scaled result with the input `self` or `selfRef`.
- Formula:

  $$
  out = self + value \times tensor1 \times tensor2
  $$

  When `aclnnAddcmul` is used, `self` and `out` in the formula correspond to those in the first-phase API. When `aclnnInplaceAddcmul` is used, both `self` and `out` in the formula correspond to `selfRef` in the first-phase API.

## Prototype

- `aclnnAddcmul` and `aclnnInplaceAddcmul` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnAddcmul`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceAddcmul`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAddcmulGetWorkspaceSize` or `aclnnInplaceAddcmulGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnAddcmul` or `aclnnInplaceAddcmul` is called to perform computation.

  ```Cpp
  aclnnStatus aclnnAddcmulGetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* tensor1, 
    const aclTensor* tensor2,  
    const aclScalar* value, 
    aclTensor*       out, 
    uint64_t*        workspaceSize, 
  aclOpExecutor**    executor)
  ```

  ```Cpp
  aclnnStatus aclnnAddcmul(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(
    const aclTensor* selfRef, 
    const aclTensor* tensor1, 
    const aclTensor* tensor2,  
    const aclScalar* value, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddcmul(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
  ```

## aclnnAddcmulGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 150px">
  <col style="width: 121px">
  <col style="width: 206px">
  <col style="width: 456px">
  <col style="width: 211px">
  <col style="width: 122px">
  <col style="width: 135px">
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
      <td>self</td>
      <td>Input</td>
      <td><code>self</code> in the formula.</td>
      <td>
        <ul>
          <li>The data types of <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the promoted type must be within the supported input types.</li>
          <li>The shapes of <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> follow the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor1</td>
      <td>Input</td>
      <td>Input <code>tensor1</code> in the formula.</td>
      <td>
        <ul>
          <li>The data types of <code>tensor1</code>, <code>self</code>, and <code>tensor2</code> follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and and the promoted type must be within the supported input types.</li>
          <li>The shapes of <code>tensor1</code>, <code>self</code>, and <code>tensor2</code> follow the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor2</td>
      <td>Input</td>
      <td>Input <code>tensor2</code> in the formula.</td>
      <td>
        <ul>
          <li>The data types of <code>tensor2</code>, <code>self</code>, and <code>tensor1</code> follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and and the promoted type must be within the supported input types..</li>
          <li>The shapes of <code>tensor2</code>, <code>self</code>, and <code>tensor1</code> follow the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Input <code>value</code> in the formula.</td>
      <td>Its data type can be cast to the data type promoted from <code>self</code>, <code>tensor1</code>, and <code>tensor2</code>. For details, see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
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
          <li>The data type promoted from <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> can be cast to the data type of <code>out</code>. For details, see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>.</li>
          <li>The shape of <code>out</code> is the same as the shape obtained by broadcasting <code>self</code>, <code>tensor1</code>, and <code>tensor2</code>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
 
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
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
      <td>The input <code>self</code>, <code>tensor1</code>, <code>tensor2</code>, <code>value</code>, or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types or formats of <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> are not supported.</td>
    </tr>
    <tr>
      <td><code>self</code>, <code>tensor1</code>, and <code>tensor2</code> do not meet the deduction relationship.</td>
    </tr>
    <tr>
      <td>The data type promoted from <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data type promoted from <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> cannot be cast to the specified type of <code>out</code>.</td>
    </tr>
    <tr>
      <td>The shape of <code>self</code>, <code>tensor1</code>, or <code>tensor2</code> has more than 8 dimensions.</td>
    </tr>
    <tr>
      <td>The shapes of <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> do not follow the broadcast relationship.</td>
    </tr>
    <tr>
      <td>The shape of <code>out</code> is different from the shape obtained by broadcasting <code>self</code>, <code>tensor1</code>, and <code>tensor2</code> </td>
    </tr>
  </tbody>
  </table>

## aclnnAddcmul

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnAddcmulGetWorkspaceSize</code>.</td>
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

## aclnnInplaceAddcmulGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1546px"><colgroup>
  <col style="width: 150px">
  <col style="width: 121px">
  <col style="width: 206px">
  <col style="width: 455px">
  <col style="width: 211px">
  <col style="width: 122px">
  <col style="width: 135px">
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
      <td><code>self</code> or <code>out</code> in the formula.</td>
      <td>
        <ul>
          <li>The data types of <code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code> follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the promoted data type can be cast to the data type of <code>selfRef</code> (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>). In addition, the promoted type must be within the supported input types.</li>
          <li>The shapes of selfRef, tensor1, and tensor2 meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast broadcast relationship</a> requirements. The shape must be the same as that obtained after broadcasting selfRef, tensor1, and tensor2.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor1</td>
      <td>Input</td>
      <td>Input <code>tensor1</code> in the formula.</td>
      <td>
        <ul>
          <li>The data types of <code>tensor1</code>, <code>selfRef</code>, and <code>tensor2</code> follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the promoted type must be within the supported input types.</li>
          <li>The shapes of <code>tensor1</code>, <code>selfRef</code>, and <code>tensor2</code> follow the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor2</td>
      <td>Input</td>
      <td>Input <code>tensor2</code> in the formula.</td>
      <td>
        <ul>
          <li>The data types of <code>tensor2</code>, <code>selfRef</code>, and <code>tensor1</code> follow the deduction relationship (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and and the promoted type must be within the supported input types.</li>
          <li>The shapes of <code>tensor2</code>, <code>selfRef</code>, and <code>tensor1</code> follow the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
      <td>ND</td>
      <td>0 to 8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Input <code>value</code> in the formula.</td>
      <td>Its data type can be cast to the data type promoted from <code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code>. For details, see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, and UINT8</td>
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

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
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
      <td>The input <code>selfRef</code>, <code>tensor1</code>, <code>tensor2</code>, or <code>value</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types or formats of <code>selfRef</code>, <code>tensor1</code> and <code>tensor2</code> are not supported.</td>
    </tr>
    <tr>
      <td><code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code> do not meet the deduction relationship.</td>
    </tr>
    <tr>
      <td>The data type promoted from <code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data type promoted from the inputs <code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code> cannot be cast to the specified type of output <code>selfRef</code>.</td>
    </tr>
    <tr>
      <td>The shape of <code>selfRef</code>, <code>tensor1</code>, or <code>tensor2</code> has more than 8 dimensions.</td>
    </tr>
    <tr>
      <td>The shapes of <code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code> do not follow the broadcast relationship.</td>
    </tr>
    <tr>
      <td>The shape of <code>selfRef</code> does not match the shape obtained by broadcasting the inputs <code>selfRef</code>, <code>tensor1</code>, and <code>tensor2</code>.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAddcmul

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnInplaceAddcmulGetWorkspaceSize</code>.</td>
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
  - `aclnnAddcmul` and `aclnnInplaceAddcmul` each default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

aclnnAddcmul

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addcmul.h"

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
  std::vector<int64_t> tensor1Shape = {4, 2};
  std::vector<int64_t> tensor2Shape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float scalarValue = 1.2f;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a tensor1 aclTensor.
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a tensor2 aclTensor.
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a value aclScalar.
  value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAddcmul.
  ret = aclnnAddcmulGetWorkspaceSize(self, tensor1, tensor2, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAddcmul.
  ret = aclnnAddcmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyTensor(out);
  aclDestroyScalar(value);

  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
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

aclnnInplaceAddcmul

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addcmul.h"

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
  std::vector<int64_t> tensor1Shape = {4, 2};
  std::vector<int64_t> tensor2Shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclScalar* value = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  float scalarValue = 1.2f;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a tensor1 aclTensor.
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a tensor2 aclTensor.
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a value aclScalar.
  value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceAddcmul.
  ret = aclnnInplaceAddcmulGetWorkspaceSize(self, tensor1, tensor2, value, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddcmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAddcmul.
  ret = aclnnInplaceAddcmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddcmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyScalar(value);

  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
