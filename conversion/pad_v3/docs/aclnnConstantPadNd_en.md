# aclnnConstantPadNd

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/pad_v3)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- This API is used to pad the input tensor self based on the pad parameter. The padding value is value.

- Formulas:

  - Shape deduction of the `out` tensor:
    $$
    \begin{aligned}
    Assume that the shape of the input self is as follows:
    \\ &[dim0_{in},dim1_{in}, dim2_{in}, dim3_{in}]
    Assume that
    \\ &pad = 
    &[dim3_{begin},dim3_{end},
    \\&&dim2_{begin},dim2_{end},
    \\&&dim1_{begin},dim1_{end},
    \\&&dim0_{begin},dim0_{end}]
    \end{aligned}
    $$

    $$
    \begin{aligned}
    & Then, the shape of out is as follows:
    \\ &[dim0_{out}, dim1_{out}, dim2_{out}, dim3_{out}] =
    &[dim0_{begin}+dim0_{in}+dim0_{end},
    \\&&dim1_{begin}+dim1_{in}+dim1_{end},
    \\&&dim2_{begin}+dim2_{in}+dim2_{end},
    \\&&dim3_{begin}+dim3_{in}+dim3_{end}]
    \end{aligned}
    $$

  - Example 1: 
    (The length of the `pad` array is twice the number of dimensions of `self`.)

    $$
    \begin{aligned}
    selfShape &= [1, 1, 1, 1, 1]\\
    pad &= \lbrace 0, 1, 2, 3, 4, 5, 6, 7, 8, 9\rbrace \\
    outputShape &= [8+1+9, 6+1+7, 4+1+5, 2+1+3, 0+1+1]\\
    &= [18,14,10,6,2]
    \end{aligned}
    $$

  - Example 2: 
    (The length of the `pad` array is less than twice the number of dimensions of `self`.)

    $$
    \begin{aligned}
    selfShape &= [1, 1, 1, 1, 1]\\
    pad &= \lbrace 0, 1, 2, 3, 4, 5\rbrace \\
    outputShape &= [0+1+0, 0+1+0, 4+1+5, 2+1+3, 0+1+1]\\
    &= [1,1,10,6,2]
    \end{aligned}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnConstantPadNdGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnConstantPadNd` is called to perform computation.

```cpp
  aclnnStatus aclnnConstantPadNdGetWorkspaceSize(
    const aclTensor*   self,
    const aclIntArray* pad, 
    const aclScalar*   value, 
    aclTensor*         out, 
    uint64_t*          workspaceSize, 
    aclOpExecutor**    executor)
```
    
```cpp
    aclnnStatus aclnnConstantPadNd(
      void*          workspace, 
      uint64_t       workspaceSize, 
      aclOpExecutor* executor, 
      aclrtStream    stream)
```

## aclnnConstantPadNdGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
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
      <td>self (aclTensor*) </td>
      <td>Input</td>
      <td>Original input data to be filled</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128 , BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0.</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pad (aclIntArray*) </td>
      <td>Input</td>
      <td>Dimensions to be padded for each axis in the input</td>
      <td>The array length must be an even number and cannot exceed twice the number of dimensions of self.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value (aclScalar*) </td>
      <td>Input</td>
      <td>Padding value of the padded part</td>
      <td>-</td>
      <td>Data type that can be converted to self</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>Output tensor</td>
      <td>Output result after padding</td>
      <td>Same as that of self</td>
      <td>ND</td>
      <td>The shape is the same as that of self.</td>
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

    - For <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type cannot be BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT8_E8M0.
    - For <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type cannot be HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT8_E8M0.
    - The data types of `value` and `self` must meet the data type deduction rules (see [Deduction Relationship](../../../docs/en/context/deduction_relationship.md)).
    - If the data type of self is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT8_E8M0, only the bit values of value can be all 0s.

- **Return Value**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned:

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 291px">
    <col style="width: 135px">
    <col style="width: 724px">
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
        <td>The input self, pad, value, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data type of self, value, or out is not supported.</td>
      </tr>
      <tr>
        <td>The data types of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The data types of self and value do not meet the data type inference rules.</td>
      </tr>
      <tr>
        <td>The shape inferred from the shape of self and the input of pad is inconsistent with the shape of out.</td>
      </tr>
      <tr>
        <td>The number of elements in pad is not an even number or exceeds twice the number of dimensions of self.</td>
      </tr>
      <tr>
        <td>The number of dimensions of self or out is greater than 8.</td>
      </tr>
      <tr>
        <td>The shape of out cannot be less than 0 for each value in pad. If there are positive numbers in pad, the shape of out cannot contain 0.</td>
      </tr>
      <tr>
        <td>When the data format of self is not ND, the data format of out is inconsistent with that of self.</td>
      </tr>
      <tr>
      <td>When the data type of self is fp8, the elements in pad cannot be negative.</td>
      </tr>
      <tr>
      <td>When the data type of self is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT8_E8M0, value cannot be 0.</td>
      </tr>
    </tbody>
    </table>

## aclnnConstantPadNd

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnConstantPadNdGetWorkspaceSize API.</td>
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
  - `aclnnConstantPadNd` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_constant_pad_nd.h"

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
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> outShape = {5, 5};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* pad = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData(25, 0);
  float valueValue = 0.0f;
  std::vector<int64_t> padData = {1,1,1,1};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a pad array.
  pad = aclCreateIntArray(padData.data(), 4);
  CHECK_RET(pad != nullptr, return ret);
  // Create a value aclScalar.
  value = aclCreateScalar(&valueValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConstantPadNd.
  ret = aclnnConstantPadNdGetWorkspaceSize(self, pad, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConstantPadNdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnConstantPadNd.
  ret = aclnnConstantPadNd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConstantPadNd failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(pad);
  aclDestroyScalar(value);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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
