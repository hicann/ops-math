# aclnnPowScalarTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/pow)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Interface function: Uses each element of `exponent` as the power of the corresponding element of `input` to complete the computation.

- Formula:

$$
out_i = x_i^{exponent_i}
$$

- Operator restrictions: If INT32 computation is performed beyond the following ranges, timeout occurs.

  | shape  | exponent_value|
  |----|----|
  |≤ 100000 (100 thousand)|–200000000 to +200000000 (200 million)|
  |≤ 1000000 (1 million)|–20000000 to +20000000 (20 million)|
  |≤ 10000000 (10 million)|–2000000 to +2000000 (2 million)|
  |≤ 100000000 (100 million)|–200000 to +200000 (200 thousand)|
  |≤ 1000000000 (1 billion)|–20000 to +20000 (20 thousand)|

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnPowScalarTensorGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnPowScalarTensor is called to perform computation.

```Cpp
aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(
 const aclScalar* self,
 const aclTensor* exponent,
 const aclTensor* out,
 uint64_t*        workspaceSize,
 aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnPowScalarTensor(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnPowScalarTensorGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
    <col style="width: 155px">
    <col style="width: 120px">
    <col style="width: 248px">
    <col style="width: 223px">
    <col style="width: 229px">
    <col style="width: 120px">
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
        <td>self</td>
        <td>Input</td>
        <td>Input aclScalar*</td>
      <td>The data types of this parameter and `exponent` must meet the <a href ="../../../docs/en/context/TensorScalar_deduction_relationship.md">TensorScalar deduction relationship</a>.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>exponent</td>
        <td>Input</td>
        <td>Input aclTensor*</td>
      <td>The data types of this parameter and `self` must meet the <a href ="../../../docs/en/context/TensorScalar_deduction_relationship.md">TensorScalar deduction relationship</a>.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>Must not exceed 8.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output aclTensor*</td>
      <td>The data type must be convertible from the deduced data type of `self` and `exponent` (see <a href = "../../../docs/en/context/conversion_relationship.md"> conversion relationships</a>).</td>
        <td></td>
        <td>ND</td>
        <td>Same as `exponent`</td>
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
    </tbody></table>

    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type cannot be BFLOAT16.
    
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>The passed `self`, `exponent`, or `out` is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of `exponent` is not supported.</td>
      </tr>
      <tr>
      <td>`self` and `exponent` do not meet the data type deduction rules.</td>
      </tr>
      <tr>
      <td>The deduced data type cannot be converted to the data type of `out`.</td>
      </tr>
      <tr>
      <td>The shapes of `exponent` and `out` are inconsistent.</td>
      </tr>
    </tbody>
    </table>

## aclnnPowScalarTensor

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPowScalarTensorGetWorkspaceSize.</td>
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
  - `aclnnPowScalarTensor` defaults to a deterministic implementation.

<term>Atlas training products</term> and <term>Atlas inference products</term>: If the computation result exceeds the value range of the specified data type, the boundary value of the data type is returned as the result.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"

#define CHECK_RET(cond, return_exponentr) \
  do {                               \
    if (!(cond)) {                   \
      return_exponentr;                   \
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> exponentShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* exponentDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* exponent = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> exponentHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float selfValue = 1.2f;
  // Create a self aclScalar.
  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  CHECK_RET(self != nullptr, return ret);
  //Create an exponent aclTensor.
  ret = CreateAclTensor(exponentHostData, exponentShape, &exponentDeviceAddr, aclDataType::ACL_FLOAT, &exponent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPowScalarTensor.
  ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPowScalarTensor.
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowScalarTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
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

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyScalar(self);
  aclDestroyTensor(exponent);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(exponentDeviceAddr);
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
