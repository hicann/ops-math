# aclnnTransMatmulWeight

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/trans_data)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |     ×      |

## Function

This API must be used together with [aclnnCalculateMatmulWeightSize](./aclnnCalculateMatmulWeightSize_en.md) and [aclnnCalculateMatmulWeightSizeV2](./aclnnCalculateMatmulWeightSizeV2_en.md) to a weight tensor for computation performance affinity of the Matmul operator. The data type can be FLOAT16, BFLOAT16, or Int8.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTransMatmulWeightGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnTransMatmulWeight` is called to perform computation.

```Cpp
aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(
  aclTensor      *mmWeightRef, 
  uint64_t       *workspaceSize, 
  aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnTransMatmulWeight(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnTransMatmulWeightGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 325px">
  <col style="width: 230px">
  <col style="width: 128px">
  <col style="width: 118px">
  <col style="width: 239px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>mmWeightRef</td>
      <td>Input</td>
      <td>Indicates the weight tensor of a Matmul operator to be processed.</td>
      <td>After this API is called, the weight tensor of the Matmul operator is updated to a preprocessed one. The format is converted into the ND or private format based on the affinity.</td>
      <td>FLOAT16, INT8</td>
      <td>ND</td>
      <td>2-6</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. The following errors may be returned:

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 303px">
  <col style="width: 118px">
  <col style="width: 726px">
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
      <td>The input is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The shape (dimension) of the input tensor does not meet the requirements.</td>
    </tr>
    <tr>
      <td>The format of the input tensor does not meet the requirements.</td>
    </tr>
    <tr>
      <td>The data type of the input tensor does not meet the requirements.</td>
    </tr>
    <tr>
      <td>Empty tensor input is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransMatmulWeight

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1142px"><colgroup>
  <col style="width: 167px">
  <col style="width: 145px">
  <col style="width: 830px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnTransMatmulWeightGetWorkspaceSize</code>.</td>
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
  - `aclnnTransMatmulWeight` defaults to a deterministic implementation.

- Cache is not supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_cast.h"

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

// Convert the uint16_t representation of FP16 to its corresponding float value.
float Fp16ToFloat(uint16_t h) {
  int s = (h >> 15) & 0x1;              // sign
  int e = (h >> 10) & 0x1F;             // exponent
  int f =  h        & 0x3FF;            // fraction
  if (e == 0) {
    // Zero or Denormal
    if (f == 0) {
      return s ? -0.0f : 0.0f;
    } 
    // Denormals
    float sig = f / 1024.0f;
    float result = sig * pow(2, -24);
    return s ? -result : result;
  } else if (e == 31) {
      // Infinity or NaN
      return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
  }
    // Normalized FP32
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
  
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

template <typename T>
int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                          aclDataType dataType, aclTensor** tensor) {
  auto size = static_cast<uint64_t>(GetShapeSize(shape));

  const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
  auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
  size *= sizeof(T);

  // Call aclrtMalloc to allocate device memory.
  ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  std::vector<int64_t> storageShape;
  storageShape.push_back(GetShapeSize(shape));

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            storageShape.data(), storageShape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {16, 32};
  std::vector<int64_t> mat2Shape = {32, 16};
  std::vector<int64_t> outShape = {16, 16};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<uint16_t> selfHostData(512, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
  std::vector<uint16_t> mat2HostData(512, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
  std::vector<uint16_t> outHostData(256, 0);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call TransWeight.
  ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnTransMatmulWeight.
  ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

  // Call the first-phase API of aclnnMm.
  uint64_t workspaceSizeMm = 0;
  ret = aclnnMmGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddrMm = nullptr;
  if (workspaceSizeMm > 0) {
    ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMm.
  ret = aclnnMm(workspaceAddrMm, workspaceSizeMm, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMm failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // In C language, FP16 values cannot be printed directly. They must be read as uint16 and then converted from their binary representation into a float value.
  for (int64_t i = 0; i < size; i++) {
    float fp16Float = Fp16ToFloat(resultData[i]);
    LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(mat2DeviceAddr);
  aclrtFree(outDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (workspaceSizeMm > 0) {
    aclrtFree(workspaceAddrMm);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
