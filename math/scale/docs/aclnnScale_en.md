# aclnnScale

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/scale)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Formula:

If `bias` is not input:

$$
  y=x*scale
$$

  If `bias` is input:

$$
  y=x*scale + bias
$$

  Note: `scale` and `bias` support broadcasting with X. The shape rules of `scale` and `bias` are as follows:

  - When scaleFromBlob is True (axis is converted to a positive number, and numAxes is -1, indicating the last axis):

    scaleShape is xShape[axis:axis + numAxes].

    biasShape is xShape[axis:axis + numAxes].

  - When scaleFromBlob is False (axis is converted to a positive number, and numAxes is -1, indicating the first axis):

    scaleShape is xShape[axis:axis + rank(scaleShape)].

    biasShape is xShape[axis:axis + rank(scaleShape)].

  Example:

  - scaleFromBlob = True:

    xShape = [a, b, c, d, e, f] axis = 3 numAxes = 2  --> scaleShape = [d, e]

    xShape = [a, b, c, d, e, f] axis = 3 numAxes = 3  --> scaleShape = [d, e, f]

    xShape = [a, b, c, d, e, f] axis = 3 numAxes = -1 --> scaleShape = [d, e, f]

  - scaleFromBlob = False:

    xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 2 --> scaleShape = [d, e]

    xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 3 --> scaleShape = [d, e, f]
  
    xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 1 --> scaleShape = [d]

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnScaleGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnScale` is called to perform computation.

* `aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *bias, int64_t axis, int64_t numAxes, bool scaleFromBlob, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScaleGetWorkspaceSize

- **Parameters:**

  - `x` (aclTensor*, compute input): tensor of the operator input, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.   
  - `scale` (aclTensor*, compute input): tensor of the operator input, aclTensor on the device. The data type must be the same as that of `x`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape meets the broadcast requirements. For details, see [Function](##Function Description).
    - <term>Atlas inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.
  - `bias` (aclTensor*, compute input): tensor of the operator input, aclTensor on the device. If `bias` is not empty, the data type must be the same as that of `scale`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape must be the same as that of `scale`.
    - <term>Atlas inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.
  - `axis` (int64_t, compute input): INT64 type on the host, specifying the start axis for scaling. The value range is [–x_rank, x_rank), where x_rank indicates the shape of `x`.
  - `numAxes` (int64_t, compute input): INT64 type on the host, specifying the number of axes for scaling. The value range is >= -1, and numAxes = -1 indicates that scaling starts from the axis to the last axis.
  - `scaleFromBlob` (bool, compute input): BOOL type on the host, specifying whether to scale from blob. True: scale from blob, and numAxes + axis is used for scaling. False: scale from input scale, and scale the input scale length from `axis` and ignore the value of `numAxes`.
  - `y` (aclTensor*, compute output): output tensor, aclTensor on the device. The shape and data type must be the same those of `x`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input x, scale, and y are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of x is not supported.
                                        2. When bias is not empty, the data type of bias is different from that of scale.
                                        3. The data type of scale is different from that of x.
                                        4. The data type of y is different from that of x.
                                        5. The shapes of x and y are different.
                                        6. When bias is not empty, the shape of bias is different from that of scale.
                                        7. The shape of x and scale is greater than 8 dimensions.
                                        8. The value of axis is not in the range of [–x_rank, x_rank).
                                        9. The value of numAxes is less than –1.
                                        10. scaleFromBlob is True, numAxes is 0, and the shape of scale is not [1].
                                        11. After axis is converted to a positive number, the sum of axis and numAxes is greater than x_rank.
                                        12. The shape of scale is not as expected (for details about the expected shape derivation, see the function description).
  ```
  
## aclnnScale

- **Parameters:**

  * `workspace` (void \*, input): size of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnScaleGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnScale` defaults to a deterministic implementation.
<term>Atlas inference products</term>: The input of `scale` and `offset` cannot be bf16.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scale.h"

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> tensor1Shape = {4};
  std::vector<int64_t> tensor2Shape = {4};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t axis = 0;
  int64_t numAxes = 1;
  bool fromBlob = true;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a tensor1 aclTensor.
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a tensor2 aclTensor.
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScale.
  ret = aclnnScaleGetWorkspaceSize(self, tensor1, tensor2, axis, numAxes, fromBlob,  out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScale.
  ret = aclnnScale(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScale failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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
