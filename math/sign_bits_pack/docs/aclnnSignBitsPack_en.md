# aclnnSignBitsPack

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/sign_bits_pack)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Packs one bit of Adam of the float16 or float32 type into uint8.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSignBitsPackGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSignBitsPack` is called to perform computation.

- `aclnnStatus aclnnSignBitsPackGetWorkspaceSize(const aclTensor* self, int64_t size, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnSignBitsPack(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnSignBitsPackGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor\*, compute input): 1D tensor used for computation, tensor on the device. Empty tensors are supported. The data type can be FLOAT16 or FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `size` (int64_t, compute input): processing dimension, which is the first dimension of the output tensor during reshaping.

  - `out` (aclTensor\*, compute output): output tensor. Only two-dimension tensors are supported. The data type can be UINT8. If the number of `self` elements is not exactly divided by 8, the total length of the output `out` is (Number of `self` elements/8) + 1. If the number of `self` elements is exactly divided by 8, the total length of the output `out` is (Number of `self` elements/8). The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `workSpaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown.
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of the passed self or out is not supported.
                                  2. The shape of self is not one-dimensional.
                                  3. size is less than 0.
                                  4. The total length of the out output cannot be exactly divided by size.
                                 
```

## aclnnSignBitsPack

- **Parameters:**

  - `workspace` (void \*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnSignBitsPackGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnSignBitsPack` defaults to a deterministic implementation.
  - The value of `size` cannot be greater than the total length of the `out` output.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sign_bits_pack.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnSignBitsPackTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  int64_t outsize = 2;
  
  std::vector<int64_t> selfShape = {14};
  std::vector<int64_t> outShape = {2,1};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData{5, 4, 3, 2, 0, -1, -2, 4, 3, 2, 1, 0, -1, -2};
  std::vector<uint8_t> outHostData{0, 0};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSignBitsPack.
  ret = aclnnSignBitsPackGetWorkspaceSize(self,
                                       outsize,
                                       out,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsPackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnSignBitsPack.
  ret = aclnnSignBitsPack(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsPack failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint8_t> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %u\n", i, outData[i]);
  }
  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnSignBitsPackTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsPackTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
