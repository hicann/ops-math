# aclnnUnfoldGrad

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/unfold_grad)

## Supported Products

| Product                                             | Supported|
|:------------------------------------------------| :------: |
| Ascend 950PR/Ascend 950DT         |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>   |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                      |    ×     |
| <term>Atlas training products</term>                      |    ×     |

## Function

- Function: implements the reverse function of the Unfold operator and calculates the corresponding gradient.

- The Unfold operator computes all slices whose size is `$size$` in dimension `$dim$` based on the input parameter `self`. The step between two slices is given by `$step$`. If `$sizedim$` is the size of dimension `$dim$` of the input parameter self, the size of dimension `$dim$` in the returned tensor is $(sizedim – size)/step + 1$. An additional dimension whose size is `$size$` is added to the returned tensor.

- The shape of the input `gradOut` of the UnfoldGrad operator is the shape of the forward output of the Unfold operator. The shape of the input `inputSizes` is the shape of the forward input `self` of the Unfold operator. The shape of the output `gradIn` of the UnfoldGrad operator is the shape of the forward input `self` of the Unfold operator.

- Example:

  ```text
  >>> x = torch.arange(1., 8)
  >>> x
  tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
  >>> x.unfold(0, 2, 1)
  tensor([[ 1.,  2.],
          [ 2.,  3.],
          [ 3.,  4.],
          [ 4.,  5.],
          [ 5.,  6.],
          [ 6.,  7.]])
  >>> x.unfold(0, 2, 2)
  tensor([[ 1.,  2.],
          [ 3.,  4.],
          [ 5.,  6.]])
  >>> res = torch.ops.aten.unfold_backward(grad, [7], 0, 2, 2)
  tensor([1, 2, 3, 4, 5, 6, 0])
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnUnfoldGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnUnfoldGrad` is called to perform computation.

```cpp
aclnnStatus aclnnUnfoldGradGetWorkspaceSize(
    const aclTensor   *gradOut, 
    const aclIntArray *inputSizes, 
    int64_t            dim, 
    int64_t            size, 
    int64_t            step, 
    const aclTensor   *gradIn, 
    uint64_t          *workspaceSize, 
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnUnfoldGrad(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnUnfoldGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 200px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-consecutive Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOut (aclTensor *) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Gradient update coefficient.</td>
      <td class="tg-0pky">The shape is (..., (sizedim – size)/step + 1, size). The `dim` dimension of `gradOut` must be equal to $(inputSizes[dim] – size)/step + 1$, and the size of `gradOut` must be equal to size of inputSizes plus 1.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">inputSizes (aclIntArray*) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the shape of the output tensor.</td>
      <td class="tg-0lax">The value is (..., sizedim), and the size of inputSizes is less than or equal to 8.</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">dim (int64_t) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the dimension in which the expansion occurs, that is, dim in the formula.</td>
      <td class="tg-0pky">The value of dim can only be len(inputSizes)-1 or len(inputSizes)-2.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">size (int64_t) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the size of each slice to be expanded, that is, size in the formula.</td>
      <td class="tg-0lax"><ul><li>The size must be greater than 0 and less than or equal to the size of the dimth dimension of inputSizes. </li><li>When dim is equal to len(inputSizes) – 1, the size is less than or equal to 49088 for the fp32 data type. The size is less than or equal to 32720 for the fp16 data type. </li><li>When dim is equal to len(inputSizes) – 2, the size is less than or equal to 88 for the fp32 data type. The step and size are less than or equal to 72 for the fp16 data type.</li></ul></td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">step (int64_t) </td>
      <td class="tg-0lax">Inputs</td>
      <td class="tg-0lax">Indicates the step between slices, that is, step in the formula.</td>
      <td class="tg-0lax"><ul><li>The step must be greater than 0. </li><li>When dim is equal to len(inputSizes) – 1, the size is less than or equal to 49088 for the fp32 data type. The size is less than or equal to 32720 for the fp16 data type. </li><li>When dim is equal to len(inputSizes) – 2, the size is less than or equal to 88 for the fp32 data type. The step and size are less than or equal to 72 for the fp16 data type.</li></ul></td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">gradIn (aclTensor *) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Gradient of Unfold.</td>
      <td class="tg-0lax">The shape is inputSizes.</td>
      <td class="tg-0lax">Same as gradOut</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Size of the workspace to be allocated on the device.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the operator executor, which contains the operator computation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 291px">
    <col style="width: 135px">
    <col style="width: 724px">
    </colgroup>
    <thead>
      <tr>
        <th class="tg-0pky">Return Value</th>
        <th class="tg-0pky">Error Code</th>
        <th class="tg-0pky">Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
        <td class="tg-0pky">161001</td>
        <td class="tg-0pky">The input and output tensors are null pointers.</td>
      </tr>
      <tr>
        <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky">161002</td>
        <td class="tg-0pky">The input and output data types and formats are not supported.</td>
      </tr>
      <tr>
        <td class="tg-0lax" rowspan="7">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td class="tg-0lax" rowspan="7">561002</td>
        <td class="tg-0lax">The dim dimension of gradOut is not equal to (inputSizes[dim] – size)/step + 1.</td>
      </tr>
      <tr>
        <td class="tg-0lax">The size of gradOut is not equal to the size of inputSizes plus 1.</td>
      </tr>
      <tr>
        <td class="tg-0lax">The size is less than or equal to 0 or greater than the dim dimension of inputSizes.</td>
      </tr>
      <tr>
        <td class="tg-0lax">The step is less than or equal to 0.</td>
      </tr>
      <tr>
        <td class="tg-0lax">The dim dimension is not equal to len(inputSizes) – 1 or len(inputSizes) – 2.</td>
      </tr>
      <tr>
        <td class="tg-0lax">When dim is equal to len(inputSizes) – 1, the step and size of the fp32 data type are greater than 49088. For the fp16 data type, the step and size are greater than 32720.</td>
      </tr>
      <tr>
        <td class="tg-0lax">When dim is equal to len(inputSizes) – 2, the step and size of the fp32 data type are greater than 88. For the fp16 data type, the step and size are greater than 72.</td>
      </tr>
    </tbody>
    </table>

## aclnnUnfoldGrad

- **Parameter description**:
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
      <td>Size of the workspace allocated on the device, which is obtained by the aclnnUnfoldGradGetWorkspaceSize API.</td>
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

- Deterministic computing:
  - `aclnnUnfoldGrad` defaults to a deterministic implementation.

1. The shape of `gradOut` must meet the following constraints:
    - The dimth dimension of gradOut is equal to (inputSizes[dim] – size)/step + 1.
    - The size of gradOut is equal to the size of inputSizes plus 1.
2. Requirements for `dim`, `size`, and `step`:
    - The size is greater than 0 and less than or equal to the dimth dimension of inputSizes.
    - The step is greater than 0.
    - The dim is equal to len(inputSizes) – 1 or len(inputSizes) – 2.
    - When dim is equal to len(inputSizes) – 1, the step and size are greater than 49088 for the fp32 data type. For the fp16 data type, the step and size are greater than 32720.
    - When dim is equal to len(inputSizes) – 2, the step and size are greater than 88 for the fp32 data type. For the fp16 data type, the step and size are greater than 72.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unfold_grad.h"

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
  // Call `aclrtMalloc` to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call `aclrtMemcpy` to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call `aclCreateTensor` to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> gradOutShape = {3, 2, 3};
  std::vector<int64_t> gradInShape = {8, 2};

  void* gradOutDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* gradIn = nullptr;

  std::vector<float> gradOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<int64_t> inputSizesData = {8, 2};
  std::vector<float> gradInHostData(16, 0);

  // Create a gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an aclIntArray.
  auto inputSizes = aclCreateIntArray(inputSizesData.data(), 2);
  CHECK_RET(inputSizes != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of `aclnnUnfoldGrad`.
  ret = aclnnUnfoldGradGetWorkspaceSize(gradOut, inputSizes, 0, 3, 2, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnfoldGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed `workspaceSize`.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of `aclnnUnfoldGrad`.
  ret = aclnnUnfoldGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnfoldGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release `aclTensor` and `aclIntArray`. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOut);
  aclDestroyIntArray(inputSizes);
  aclDestroyTensor(gradIn);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
