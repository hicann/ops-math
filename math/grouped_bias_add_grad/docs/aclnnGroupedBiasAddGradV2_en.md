# aclnnGroupedBiasAddGradV2

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/grouped_bias_add_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Implements backpropagation of `groupBiasAdd`. This API is an extension of [aclnnGroupedBiasAddGrad](./aclnnGroupedBiasAddGrad_en.md), with the `groupIdxType` attribute added and the `groupIdx` type specified.
- Formula:<br>
(1) If `groupIdxOptional` is available and `groupIdxType` is 0:

$$
out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
$$

&emsp;&emsp; (2) If `groupIdxOptional` is available and `groupIdxType` is 1:
$$
groupIdx(i) = \sum_{i=0}^{j} groupIdxOptional(j), j=0...G
$$

$$
out(G,H) = \begin {cases} \sum_{i=groupIdx(j-1)}^{groupIdx(j)} gradY(i,H), & 1 \leq j \leq G-1 \\ \sum_{i=0}^{groupIdx(j)} gradY(i, H), & j=0 \end {cases}
$$

&emsp;&emsp;`gradY` has two dimensions, `H` indicates the size of the last dimension of `gradY`, while `G` indicates the size of dimension 0 of `groupIdxOptional`. That is, `groupIdxOptional` has `G` numbers, and `groupIdxOptional(j)` indicates the size of the *j*th dimension. After computation, `out` is 2-dimensional, with the shape (G, H).<br>
&emsp;&emsp;(3) If `groupIdxOptional` is unavailable:

$$
out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
$$

&emsp;&emsp;`gradY` has three dimensions. `G`, `C`, and `H` indicate the sizes of dimensions 0 to 2 of `gradY`. After computation, `out` is 2-dimensional, with the shape (G, H).

- Example:<br>
(1) If `groupIdxOptional` is available and `groupIdxType` is 0:<br>
  The shape of `gradY` is (1000, 30), and the shape of `groupIdxOptional` is (400, 600, 1000). `gradY` is divided into three groups, and the accumulated number of rows in each group is 400, 200, and 400 respectively. After computation, the shape of `out` is (3, 30).<br>
(2) If `groupIdxOptional` is available and `groupIdxType` is 1:<br>
  The shape of `gradY` is (1000, 30), and the shape of `groupIdxOptional` is (400, 210, 390). `gradY` is divided into three groups, and the accumulated number of rows in each group is 400, 210, and 390 respectively. After computation, the shape of `out` is (3, 30).<br>
(3) If `groupIdxOptional` is unavailable:<br>
  The shape of `gradY` is (10, 100, 30). `gradY` is divided into 10 groups. The accumulated number of rows in each group is 100. After computation, the shape of `out` is (10, 30).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedBiasAddGradV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGroupedBiasAddGradV2` is called to perform computation.

- `aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(const aclTensor *gradY, const aclTensor *groupIdxOptional, int64_t groupIdxType, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGroupedBiasAddGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGroupedBiasAddGradV2GetWorkspaceSize

- **Parameters:**

  * `gradY` (aclTensor\*, computation input): required parameter, backpropagation gradient, `gradY` in the formula, aclTensor on the device. The data type can be FLOAT, FLOAT16, or BFLOAT16. If `groupIdxOptional` is available, the shape can only be 2-dimensional. If `groupIdxOptional` is not available, the shape can only be 3-dimensional. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `groupIdxOptional` (aclTensor\*, computation input): optional parameter, end position of each group, `groupIdxOptional` in the formula, aclTensor on the device. The data type can be INT32 or INT64. The shape supports only one dimension. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `groupIdxType` (int64_t, computation input): `groupIdx` type. The options are as follows:
    * `0` indicates that the value of `groupIdxOptional` is the end index of each group.
    * `1` indicates that the value of `groupIdxOptional` is the size of each group.
  * `out` (aclTensor\*, output): gradient of bias, out in the formula, aclTensor on the device. The data type can be FLOAT, FLOAT16, or BFLOAT16. The data type must be the same as that of `gradY`. The shape supports only two dimensions. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  * `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API performs input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradY or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or shape of gradY, groupIdxOptional, or out is not supported.
                                   2. The dimensions of gradY, groupIdxOptional, and out do not match.
                                   3. The number of groups exceeds 2048.
                                   4. The value of groupIdxType is not supported.
  ```
  
## aclnnGroupedBiasAddGradV2

- **Parameters:**

  * `workspace` (void\*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnGroupedBiasAddGradV2GetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnGroupedBiasAddGradV2` defaults to a deterministic implementation.
- * `groupIdxOptional` supports a maximum of 2048 numbers.
- * If `groupIdxOptional` is available, ensure that the tensor values are not greater than the maximum INT32 value and are not negative numbers.
- If `groupIdxOptional` is available and `groupIdxType` is `0`, ensure that the tensor values are sorted in ascending order and the last value is equal to the size of the 0th dimension of `gradY`.
- If `groupIdxOptional` is available and `groupIdxType` is 1, ensure that the sum of tensor values is equal to the size of the 0th dimension of `gradY`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_bias_add_grad.h"

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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> gradYShape = {40, 10};
  std::vector<int64_t> groupIdxShape = {4};
  std::vector<int64_t> outShape = {4, 10};
  void* gradYDeviceAddr = nullptr;
  void* groupIdxDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradY = nullptr;
  aclTensor* groupIdx = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradYHostData(400, 1.0);
  std::vector<int32_t> groupIdxHostData = {5, 15, 10, 10};
  std::vector<float> outHostData(40, 0.0);
  int64_t groupIdxType = 1;

  //Create a gradY aclTensor.
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a groupIdxOptional aclTensor.
  ret = CreateAclTensor(groupIdxHostData, groupIdxShape, &groupIdxDeviceAddr, aclDataType::ACL_INT32, &groupIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupedBiasAddGradV2.
  ret = aclnnGroupedBiasAddGradV2GetWorkspaceSize(gradY, groupIdx, groupIdxType, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupedBiasAddGradV2.
  ret = aclnnGroupedBiasAddGradV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(gradY);
  aclDestroyTensor(groupIdx);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(groupIdxDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
