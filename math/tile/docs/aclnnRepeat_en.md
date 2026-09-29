# aclnnRepeat

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/tile)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Repeats the input tensor for the number of times specified by `repeats` along each dimension. Example:
Assume that the input tensor is [[a,b],[c,d],[e,f]]. That is, `shape` is [3,2] and `repeats` is (2,4). In this case, the value of `shape` of the generated tensor is [6,8]. The values are as follows:

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

When `repeats` is (2,4,2), that is, the number of elements in `repeats` exceeds the dimensions in the tensor, the output tensor is equivalent to the following operation: Expand the shape of the input tensor to the dimension [1,3,2] that is the same as the number of `repeats`; continue to expand the tensor based on the corresponding dimension and the value of `repeats`, and the tensor is output as [2,12,4]. The result is as follows:

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

The following conditions must be met during computation: 
The number of parameters in `repeats` cannot be less than the number of dimensions of the input tensor. 
The value in `repeats` must be greater than or equal to 0. 

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRepeatGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRepeat` is called to perform computation.

* `aclnnStatus aclnnRepeatGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnRepeat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRepeatGetWorkspaceSize

- **Parameters:**

  * `self` (aclTensor*, compute input): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape cannot be greater than 8D.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, or BFLOAT16.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, or BOOL.
  * `repeats` (aclIntArray*, compute input): aclIntArray on the host. The data type is INT64, indicating the repeats of the input tensor along each dimension. The number of parameters cannot exceed 8. Currently, repeats cannot be performed on more than four dimensions at the same time. For details about the constraints, see [Constraints] (#Constraints).

  * `out` (aclTensor \*, compute output): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The dimensions cannot be greater than 8. The type must be the same as that of `self`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, or BFLOAT16.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, or BOOL.
  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.

  * `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns**:

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed; width: 1207px"><colgroup>
  <col style="width: 268px">
  <col style="width: 138px">
  <col style="width: 801px">
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
      <td>The input self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or format of self or out is not supported.</td>
    </tr>
    <tr>
      <td>The types of `self` and `out` do not match.</td>
    </tr>
    <tr>
      <td>The number of parameters of `repeats` is less than the dimensions of the input tensor.</td>
    </tr>
    <tr>
      <td>The value of `repeats` is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The number of dimensions of `self` exceeds 8.</td>
    </tr>
    <tr>
      <td>The number of parameters of `repeats` exceeds 8.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>The kernel execution fails, and the intermediate result is null.</td>
    </tr>
    <tr>
      <td>Repeat is performed on more than four dimensions at the same time.</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeat

- **Parameters:**

  * `workspace` (void \*, input): address of the workspace memory to be allocated on the device.

  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRepeatGetWorkspaceSize.

  * `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.

  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - aclnnRepeat defaults to a deterministic implementation.

The kernel of the internal broadcast of the repeat function supports a maximum of eight dimensions. Currently, the dimensions cannot exceed eight after expansion. The details are as follows: 
  Restriction 1: When repeat needs to be performed on the first axis, it can be performed on a maximum of four dimensions at the same time (that is, the number of non-1 repeat parameters cannot exceed 4).

  ```text
   x.repeat(2, 3, 4, 5, 6)  # Not supported. An error is reported during verification. The repeat value of the first axis is 2, and there are five non-1 repeat parameters.
   x.repeat(2, 3, 1, 5, 6)  # Supported. The repeat value of the first axis is 2, and there are four non-1 repeat parameters.
  ```

  Restriction 2: When repeat needs to be performed on the first axis, it can be performed on a maximum of three dimensions at the same time (that is, the number of `repeats` in non-1 format cannot exceed 3).

  ```text
   x.repeat(1, 3, 4, 5, 6)  # Not supported. An error is reported during verification. The repeat value of the first axis is 1, and there are four non-1 repeat parameters.
   x.repeat(1, 3, 1, 5, 6)  # Supported. The repeat value of the first axis is 1, and there are three non-1 repeat parameters.
  ```

## Example

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

template <typename T>
void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray) {
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {3, 2};
  std::vector<int64_t> outShape = {2, 12, 4};
  std::vector<int64_t> repeatsArray = {2, 4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclIntArray* repeat = nullptr;
  std::vector<float> selfHostData(GetShapeSize(selfShape) * 2, 1);
  std::vector<float> outHostData(GetShapeSize(outShape) * 2, 1);
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
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
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

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
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

  // 7. Release device resources. Modify the configuration based on the API definition.
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
