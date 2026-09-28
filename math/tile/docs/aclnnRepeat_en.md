# aclnnRepeat

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/tile)

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

- Function: Replicates the input tensor along the number of times specified for each dimension in repeats.
- For example, if the input tensor is [[a,b],[c,d],[e,f]], that is, the shape is [3,2] and repeats is (2,4), the shape of the generated tensor is [6,8], and the values are as follows:

  ```text
  >>> x = torch.tensor([[a,b],[c,d],[e,f]])
  >>> x.repeat(2,4)
  tensor([[a,b,a,b,a,b,a,b],
          [c,d,c,d,c,d,c,d],
          [e,f,e,f,e,f,e,f],
          [a,b,a,b,a,b,a,b],
          [c,d,c,d,c,d,c,d],
          [e,f,e,f,e,f,e,f],
          ])
  ```

- When `repeats` is (2,4,2), that is, the number of elements in `repeats` exceeds the dimensions in the tensor, the output tensor is equivalent to the following operation: Expand the shape of the input tensor to the dimension [1,3,2] that is the same as the number of `repeats`; continue to expand the tensor based on the corresponding dimension and the value of `repeats`, and the tensor is output as [2,12,4]. The result is as follows:

  ```text
  >>> x.repeat(2,4,2)
  tensor([[[a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f]],

          [[a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f]]])
  ```

- The following conditions must be met during computation:

    - The number of parameters in **repeats** cannot be less than the dimensions of the input tensor.
    - The value of **repeats** must be greater than or equal to 0.

## Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). You must call aclnnRepeatGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnRepeat to perform computation.

```cpp
aclnnStatus aclnnRepeatGetWorkspaceSize(
    const aclTensor   *self, 
    const aclIntArray *repeats, 
    aclTensor         *out, 
    uint64_t          *workspaceSize, 
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnRepeat(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnRepeatGetWorkspaceSize

- **Parameter Description**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 290px">
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
      <td>-</td>
      <td>-</td>
      <td>FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats (aclIntArray*) </td>
      <td>Input</td>
      <td>-</td>
      <td>Number of times that the input tensor is repeated along each dimension. The number of parameters cannot be greater than 8.<br>Currently, the repeat operation cannot be performed on more than four dimensions at the same time. For details about the restrictions, see the restrictions.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>Same as that of self</td>
      <td>ND</td>
      <td>≤8</td>
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

  - For <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>, the data type does not support HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN.

  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type cannot be BFLOAT16, HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
  The first-phase API implements input parameter validation. The following error codes may be returned.
  
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
      <td>The input <code>self</code> or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type and format of self and out are not supported.</td>
    </tr>
    <tr>
      <td>The types of self and out do not match.</td>
    </tr>
    <tr>
      <td>The number of repeats is less than the number of dimensions of the input tensor.</td>
    </tr>
    <tr>
      <td>The repeats parameter contains a value less than 0.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self exceeds 8.</td>
    </tr>
    <tr>
      <td>The number of repeats parameters exceeds 8.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>The kernel fails to be executed, and the intermediate result is null.</td>
    </tr>
    <tr>
      <td>The repeat operation is performed on more than four dimensions at the same time.</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeat

- **Parameter Description**

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnRepeatGetWorkspaceSize.</td>
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
  - aclnnRepeat defaults to a deterministic implementation.

The kernel of the internal broadcast of the repeat function supports a maximum of eight dimensions. Currently, the dimensions cannot exceed eight after expansion. The details are as follows: 
  Constraint 1: When the first axis needs to be repeated, a maximum of four dimensions can be repeated at the same time. That is, the number of repeat parameters whose value is not 1 cannot exceed 4.

  ```text
   x.repeat(2, 3, 4, 5, 6)  # Not supported. An error is reported during verification. The repeat value of the first axis is 2, and there are five non-1 repeat parameters.
   x.repeat(2, 3, 1, 5, 6)  # Supported. The repeat value of the first axis is 2, and there are four non-1 repeat parameters.
  ```

  Constraint 2: When the first axis does not need to be repeated, a maximum of three dimensions can be repeated at the same time. That is, the number of repeat parameters whose value is not 1 cannot exceed 3.

  ```text
   x.repeat(1, 3, 4, 5, 6)  # Not supported. An error is reported during verification. The repeat value of the first axis is 1, and there are four non-1 repeat parameters.
   x.repeat(1, 3, 1, 5, 6)  # Supported. The repeat value of the first axis is 1, and there are three non-1 repeat parameters.
  ```

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
 } while (0)

#define LOG_PRINT(message, ...)      \
 do {                                \
    printf(message, ##__VA_ARGS__);  \
 } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
        shape_size *= i;
    }
    return shape_size;
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

template <typename T>
void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray) {
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {3, 2};
  std::vector<int64_t> outShape = {2, 12, 4};
  std::vector<int64_t> repeatsArray = {2, 4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclIntArray* repeat = nullptr;
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 1);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a normalizedShape aclIntArray.
  aclCreateIntArrayP(repeatsArray, &repeat);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeat.
  ret = aclnnRepeatGetWorkspaceSize(self, repeat, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeat.
  ret = aclnnRepeat(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeat failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyIntArray(repeat);

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
