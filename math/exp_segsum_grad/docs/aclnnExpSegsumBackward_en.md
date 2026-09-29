# aclnnExpSegsumBackward

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/exp_segsum_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs backward pass of [aclnnExpSegsum](../../segsum/docs/aclnnExpSegsum_en.md).
- Formula (using 5D input as an example):

  1. Multiply the input `gradOutput` of shape (N1, N2, N3, N4, N4) with the input `gradSelf` (forward output).

     $$
     out\_mul = gradOutput * gradSelf
     $$

  2. Generate a BOOL triangular matrix A of shape (N4, N4), where the upper triangle is `True`, the lower triangle is `False`, and the diagonal is `True`.
  3. Fill the elements in $out\_mul$ with `0` at the positions where the corresponding elements in matrix A are `True`.

     $$
     out\_mul_i=
     \begin{cases}out\_mul_i,\quad A_i==False
     \\0, \quad A_i==True
     \end{cases}
     $$

  4. Reverse the second-to-last dimension of `out_mul` to generate `out_flip`.
  5. Perform a cumulative sum along the second-to-last dimension of `$out\_flip$`. For an element from a dimensional perspective, other dimension indices remain unchanged while the current index increases. $out\_cumsum\_{i}$ is the element at the corresponding position in the output tensor.

     $$
     out\_cumsum_{i} = out\_flip_{1} + out\_flip_{2} + out\_flip_{3} + ...... + out\_flip_{i}
     $$

  6. Reverse the second-to-last dimension of $out\_cumsum$ to generate `out_flip2`.
  7. Generate a BOOL triangular matrix B of shape (N4, N4), where the upper triangle is `True`, the lower triangle is `False`, and the diagonal is `True`.
  8. Fill the elements in $out\_flip2$ with `0` at the positions where the corresponding elements in matrix B are `True`.

     $$
     out\_flip2_i=
     \begin{cases}out\_flip2_i,\quad B_i==False
     \\0, \quad B_i==True
     \end{cases}
     $$

  9. Return `gradInput` as the sum of each row along the last dimension of `out\_flip2`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnExpSegsumBackwardGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnExpSegsumBackward` is called to perform computation.

```Cpp
aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     gradSelf,
  aclTensor           *gradInput,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnExpSegsumBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnExpSegsumBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient for backward pass. It corresponds to gradOutput in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The last two dimensions of the shape must have the same size.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4–5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradSelf</td>
      <td>Input</td>
      <td>Forward pass output. It corresponds to gradSelf in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type are the same as those of gradOutput.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4–5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Input</td>
      <td>Backward pass output. It corresponds to gradInput in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of gradOutput. </li><li>The output dimension must be one less than the input dimension. 3D or 4D is supported. When the input gradOutput is 4D, the output dimension size is the same as the first three dimensions of gradOutput. When the input gradOutput is 5D, the output dimension size is the same as the first four dimensions of gradOutput.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API performs input parameter validation. The following errors may be returned:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOutput, gradSelf, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of gradOutput, gradSelf, or gradInput is not supported.</td>
    </tr>
    <tr>
      <td>The shape and data type of gradOutput, gradSelf, or gradInput do not meet the parameter requirements.</td>
    </tr>
  </tbody></table>

## aclnnExpSegsumBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnExpSegsumBackwardGetWorkspaceSize.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnExpSegsumBackward` defaults to a deterministic implementation.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_exp_segsum_backward.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Perform the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  void *gradOutputDeviceAddr = nullptr;
  void *gradSelfDeviceAddr = nullptr;
  void *gradInputDeviceAddr = nullptr;
  aclTensor *gradOutput = nullptr;
  aclTensor *gradSelf = nullptr;
  aclTensor *gradInput = nullptr;
  std::vector<int64_t> gradOutputShape = {1, 1, 1, 4, 4};
  std::vector<int64_t> gradSelfShape = {1, 1, 1, 4, 4};
  std::vector<int64_t> gradInputShape = {1, 1, 1, 4};
  std::vector<float> gradOutputHostData(16,1);
  std::vector<float> gradSelfHostData(16,1);
  std::vector<float> gradInputHostData(4, 0);

  // Create an input aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradSelfHostData, gradSelfShape, &gradSelfDeviceAddr, aclDataType::ACL_FLOAT, &gradSelf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnExpSegsumBackward.
  ret = aclnnExpSegsumBackwardGetWorkspaceSize(gradOutput, gradSelf, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnExpSegsumBackward.
  ret = aclnnExpSegsumBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumBackward failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(gradSelf);
  aclDestroyTensor(gradInput);
  
  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(gradSelfDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
