# aclnnCalculateMatmulWeightSize

[📄 View Source Code](https://gitcode.com/cann/ops-math/tree/master/conversion/trans_data)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function Description

- Description:
  Calculates the size of the weight to be allocated when the Matmul operator is in ND format. This API is used only to determine the size required for preprocessing the weight tensor to achieve optimal Matmul operator execution performance.
  For example, for performance optimization, when the input shape is [510, 510], the function reshapes it to [512, 512]. As a result, the input is interpreted as having 262,144 elements.

- Formula:

  $$
  Float16/Bfloat16:
  result=\prod_{i \in(0, 3]}Align(tensorShape[i], 16)
  $$

  $$
  INT8:
  result = Align(Shapesize[0], 16) * Align(Shapesize[1], 32)
  $$

## Prototype

`aclnnStatus aclnnCalculateMatmulWeightSize(const aclIntArray *tensorShape, uint64_t *weightTensorSize)`

## aclnnCalculateMatmulWeightSize

- **Parameters:**

  * `tensorShape` (aclIntArray *, computation input): specifies the shape of the weight matrix for the Matmul operation. This parameter corresponds to `Shapesize` in the formula. It is represented as an aclIntArray on the host side. Only 2D shapes (n, k) are supported, where `n` is the size of the first dimension and `k` is the size of the second dimension. Empty arrays are not supported.
    - <term>Atlas inference products</term>: The data type can be FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
  * `weightTensorSize` (uint64_t \*, computation output): specifies the number of elements required for weight based on the internal processing logic of MatMul. This parameter corresponds to `result` in the formula.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The computation failed.
  ```

## Constraints

- Deterministic computation:
  - `aclnnCalculateMatmulWeightSize` defaults to a deterministic implementation.

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
  // Boilerplate code for resource initialization.
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
int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                          aclDataType dataType, aclTensor** tensor) {
  auto size = static_cast<uint64_t>(GetShapeSize(shape));

  const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
  auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
  size *= sizeof(T);

  // Call aclrtMalloc to allocate memory on the device.
  ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data from the host to the device.
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
  // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
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
  std::vector<uint16_t> selfHostData(512, 0x3C00); // In float16 format, 1 is represented as 0x3C00 (uint16_t).
  std::vector<uint16_t> mat2HostData(512, 0x3C00); // In float16 format, 1 is represented as 0x3C00 (uint16_t).
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call TransWeight.
  ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize calculated by the first-phase API.
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

  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddrMm = nullptr;
  if (workspaceSizeMm > 0) {
    ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMm.
  ret = aclnnMm(workspaceAddrMm, workspaceSizeMm, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMm failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // In C language, FP16 values cannot be printed directly. They must be read as uint16_t and then cast from their binary representation into a float value.
  for (int64_t i = 0; i < size; i++) {
    float fp16Float = Fp16ToFloat(resultData[i]);
    LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);

  // 7. Free device resources. Modify the code based on the API definition.
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
