# aclnnSliceV2

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/strided_slice_v3)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Extracts the tensor $out$ from the input tensor $self$ based on the specified dimension $axes$, range $[starts, ends]$, and step $steps$.
$starts[i]$ and $ends[i]$ can be set to values other than $[0, self.shape[axes[i]]]$. After the values are set, they are converted to valid values using the following formula (assume $self.shape[axes[i]] = N$):

$$
starts[i] = \begin{cases}
&0, & if\;starts[i] < -N \\
&N, & if\;starts[i] >= N\\
&(starts[i]+N) \% N, & otherwise
\end{cases}
$$

$$
ends[i] = \begin{cases}
&N, & if\; ends[i] >= N\\
&starts, & else\;if\;  (ends[i]+N)\%N < starts[i] \\
&(ends[i]+N)\%N, & otherwise\\
\end{cases}
$$

$out.shape$ is the same as $self.shape$ except on the `axes` axis.

$$
out.shape[axes[i]] = ⌊\frac{ends[i] - starts[i] + steps[i] - 1}{steps[i]}⌋
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSliceV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSliceV2` is called to perform computation.

- `aclnnStatus aclnnSliceV2GetWorkspaceSize(const aclTensor* self, const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes, const aclIntArray* steps, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSliceV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnSliceV2GetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): aclTensor on the device, $self$ in the formula. The shape can have 1 to 8 dimensions. The data types of `self` and `out` are the same. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT8, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT8, UINT8, BOOL, or BFLOAT16.

  - `starts` (aclIntArray*, computation input): aclIntArray on the host, $starts$ in the formula, indicating the start index of the slice position. `starts`, `ends`, `axes`, and `steps` have the same number of elements.

  - `ends` (aclIntArray*, computation input): aclIntArray on the host, $ends$ in the formula, indicating the end index of the slice position. `starts`, `ends`, `axes`, and `steps` have the same number of elements.

  - `axes` (aclIntArray*, computation input): aclIntArray on the host, $axes$ in the formula, indicating the axis for slicing. `starts`, `ends`, `axes`, and `steps` have the same number of elements. The elements of `axes` must be within the range [–self.dim(), self.dim() – 1], and the axes must be unique.

  - `steps` (aclIntArray*, computation input): aclIntArray on the host, $steps$ in the formula, indicating the slice step. `starts`, `ends`, `axes`, and `steps` have the same number of elements. The elements of `steps` are all positive integers.

  - `out` (aclTensor*, computation output): aclTensor on the device, $out$ in the formula. The shape can have 0 to 8 dimensions and must meet the deduction rules in the function description. The data types of `self` and `out` are the same. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT8, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT8, UINT8, BOOL, or BFLOAT16.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, starts, ends, axes, steps, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                        2. The shape of self is greater than 8D.
                                        3. Some values of axes are out of the range [–self.dim(), self.dim() – 1], or some axes are the same.
                                        4. Some values of steps are less than or equal to 0.
                                        5. The data types of self and out are inconsistent.
                                        6. The shape of out does not meet the deduction rules in the function description.
                                        7. starts, ends, axes, and steps have different numbers of elements.
  ```

## aclnnSliceV2

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnSliceV2GetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnSliceV2` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_slice_v2.h"

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {2, 1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclIntArray *dim = nullptr;
  aclIntArray *start = nullptr;
  aclIntArray *end = nullptr;
  aclIntArray *step = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0};
  int64_t dimValue[] = {0,1};
  int64_t startValue[] = {1,1};
  int64_t endValue[] = {3,2};
  int64_t stepValue[] = {1,1};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  dim = aclCreateIntArray(dimValue, 2);
  CHECK_RET(dim != nullptr, return ret);
  start = aclCreateIntArray(startValue, 2);
  CHECK_RET(start != nullptr, return ret);
  end = aclCreateIntArray(endValue, 2);
  CHECK_RET(end != nullptr, return ret);
  step = aclCreateIntArray(stepValue, 2);
  CHECK_RET(step != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSliceV2.
  ret = aclnnSliceV2GetWorkspaceSize(self, start, end, dim, step, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSliceV2.
  ret = aclnnSliceV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(dim);
  aclDestroyIntArray(start);
  aclDestroyIntArray(end);
  aclDestroyIntArray(step);
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
