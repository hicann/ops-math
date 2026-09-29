# aclnnConstantPadNd

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/pad_v3)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Pads the input tensor `self` with data based on the `pad` parameter. The padded data is specified by `value`.

- Formulas:

  - Shape deduction of the `out` tensor:

    $$
    \begin{align}
    \text{Assume that the shape of the input} \textit{self} \text{ is as follows:} \\
    [dim0_{in}, dim1_{in}, dim2_{in}, dim3_{in}]\\
    \text{Assume }\textit{pad} = \lbrace dim3_{begin},dim3_{end},\\
    dim2_{begin},dim2_{end},\\
    dim1_{begin},dim1_{end},\\
    dim0_{begin},dim0_{end} \rbrace
    \end{align}
    $$

    $$
    \begin{aligned}
    \text{Then, the shape of } \textit{out} \text{ is as follows:} \\
    [dim0_{out}, dim1_{out}, dim2_{out}, dim3_{out}] = \\
    [dim0_{begin} + dim0_{in} + dim0_{end}, \\
    dim1_{begin} + dim1_{in} + dim1_{end}, \\
    dim2_{begin} + dim2_{in} + dim2_{end}, \\
    dim3_{begin} + dim3_{in} + dim3_{end}]
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

  - `aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor* self, const aclIntArray* pad, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnConstantPadNd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnConstantPadNdGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, computation input): original input data to be padded, `aclTensor` on the device. The shape must be 0D to 8D. The data types of `value` and `self` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, or COMPLEX128.
  - `pad` (aclIntArray*, computation input): padding for each axis of the input, `aclIntArray` on the host. The array length must be an even number and cannot exceed twice the number of dimensions of `self`.

  - `value` (aclScalar*, computation input): padding value, `aclScalar` on the host. The data types of `value` and `self` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).

  - `out` (aclTensor*, computation output): output result after padding, `aclTensor` on the device. The shape must meet the deduction rules in the example. The data type must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, or COMPLEX128.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter validation. The following error codes may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR):
                                      1. The passed self, pad, value, or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID):
                                      1. The data type of self, value, or out is not supported.
                                      2. The data types of self and out are inconsistent.
                                      3. The data types of self and value do not meet the type deduction rules.
                                      4. The shape deduced from self and pad is inconsistent with that of out.
                                      5. The number of elements in pad is not an even number or exceeds twice the number of dimensions of self.
                                      6. The number of dimensions of self or out exceeds 8.
                                      7. No value in pad can result in a dimension size of out being less than 0. If pad contains any positive value, no dimension size of out can be 0.
                                      8. If the data format of self is not ND, the data format of out is inconsistent with that of self.
```

## aclnnConstantPadNd

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnConstantPadNdGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

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
