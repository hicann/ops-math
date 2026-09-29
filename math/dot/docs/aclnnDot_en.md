# aclnnDot

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/dot)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Computes the dot product of two 1D tensors.

- Formula:

$$
self = [x_{1}, x_{2}, ..., x_{n}]
$$

$$
tensor = [y_{1}, y_{2}, ..., y_{n}]
$$

$$
out = x_{1}*y_{1} + x_{2}*y_{2} + ... + x_{n}*y_{n}
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnDotGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnDot` is called to perform computation.

```Cpp
aclnnStatus aclnnDotGetWorkspaceSize(
  const aclTensor* self, 
  const aclTensor* tensor, 
  aclTensor*       out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnDot(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnDotGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 291px">
  <col style="width: 286px">
  <col style="width: 284px">
  <col style="width: 120px">
  <col style="width: 141px">
  <col style="width: 146px">
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
      <td>Input tensor for the dot product operation, <code>self</code> in the formula.</td>
      <td>The data type must be the same as those of <code>tensor</code> and <code>out</code>. The shape must be identical to that of <code>tensor</code>.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor</td>
      <td>Input</td>
      <td>Input tensor for dot product computation, <code>tensor</code> in the formula.</td>
      <td>The data type must be the same as those of <code>tensor</code> and <code>out</code>. The shape must be identical to that of <code>tensor</code>.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Dot product computation result, <code>out</code> in the formula.</td>
      <td>The data type must be the same as that of <code>tensor</code> and <code>out</code>.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
      <td>0</td>
      <td>-</td>
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

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type cannot be BFLOAT16.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
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
      <td><code>self</code>, <code>tensor</code>, or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of <code>self</code>, <code>tensor</code>, or <code>out</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data types of <code>self</code>, <code>tensor</code>, and <code>out</code> are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>self</code> or <code>tensor</code> is not 1.</td>
    </tr>
    <tr>
      <td>The shapes of <code>self</code> and <code>tensor</code> are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>out</code> is not 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnDot

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnDotGetWorkspaceSize</code>.</td>
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
  - `aclnnDot` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dot.h"

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
  std::vector<int64_t> selfShape = {4};
  std::vector<int64_t> tensorShape = {4};
  std::vector<int64_t> outShape = {};
  void* selfDeviceAddr = nullptr;
  void* tensorDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> tensorHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an aclTensor.
  ret = CreateAclTensor(tensorHostData, tensorShape, &tensorDeviceAddr, aclDataType::ACL_FLOAT, &tensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnDot.
  ret = aclnnDotGetWorkspaceSize(self, tensor, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDot.
  ret = aclnnDot(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDot failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
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

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(tensor);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensorDeviceAddr);
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
