# aclnnLinalgCross

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/cross)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √    |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √    |

## Function

- Description: Performs linear_cross operation on the input tensor.

- Formula:

$$
out = self\times other = \begin{vmatrix}i&j&k\\x_1&y_1&z_1\\x_2&y_2&z_2\end{vmatrix} = (y_1z_2-y_2z_1)i-(x_1z_2-x_2z_1)j+(x_1y_2-x_2y_1)k
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLinalgCrossGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLinalgCross` is called to perform computation.

- `aclnnStatus aclnnLinalgCrossGetWorkspaceSize(const aclTensor* self, const aclTensor* other, int64_t dim, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnLinalgCross(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnLinalgCrossGetWorkspaceSize

- **Parameters**

  - self (aclTensor*, computation input): self in the formula, aclTensor on the device. The data type is the same as that of other and out. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports 0 to 8 dimensions, and must meet [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`, and the value of the axis specified by `dim` is 3.
    - <term>Atlas training products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, FLOAT64, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>;: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, BFLOAT16, FLOAT, FLOAT64, COMPLEX64, or COMPLEX128.

  - `other` (aclTensor*, computation input): `other` in the formula, `aclTensor` on the device. The data type is the same as that of `self` and `out`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports 0 to 8 dimensions, and must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`.
    - <term>Atlas training products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, FLOAT64, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>;: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, BFLOAT16, FLOAT, FLOAT64, COMPLEX64, or COMPLEX128.

  - `dim` (int64_t, computation input): axis for linear_cross of `self`. The data type is INT64. If `dim` is not specified, the default value is `–1`. The value range is [–Number of dimensions of `self`, Number of dimensions of `self` – 1].

  - `out` (aclTensor*, computation output): output `out` in the formula, `aclTensor` on the device. The data type is the same as that of `self` and `other`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND, and the shape must be the same as that after `self` and `other` are broadcast.
    - <term>Atlas training products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, FLOAT64, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>;: The data type can be INT8, INT16, INT32, INT64, UINT8, FLOAT16, BFLOAT16, FLOAT, FLOAT64, COMPLEX64, or COMPLEX128.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): The input self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of self, other, and out are inconsistent, or the data format is not supported.
                                        2. The number of dimensions of self and other is greater than 8.
                                        3. self and other do not comply with the broadcast relationship.
                                        4. The shape after self and other are broadcast is inconsistent with that of out.
                                        5. The shape of self in the corresponding dim dimension is not 3.
                                        6. The value of dim is not within the range of [–Number of dimensions of self, Number of dimensions of self – 1].

  ```

## aclnnLinalgCross

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLinalgCrossGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnLinalgCross` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_linalg_cross.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,
    void** selfDeviceAddr, void** otherDeviceAddr, void** outDeviceAddr, aclTensor** self, aclTensor** other,
    aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, otherDeviceAddr, aclDataType::ACL_DOUBLE, other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_DOUBLE, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* other, aclTensor* out, int64_t dim, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnLinalgCross.
  auto ret = aclnnLinalgCrossGetWorkspaceSize(self, other, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgCrossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the second-phase API of aclnnLinalgCross.
  ret = aclnnLinalgCross(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgCross failed. ERROR: %d\n", ret); return ret);

  // Implement synchronization.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the result from the device to the host.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  // 1. Initialize the device/stream.
  int32_t deviceId = 0;
  aclrtStream stream;

  CHECK_RET(InitAcl(deviceId, &stream) == ACL_SUCCESS, return -1);

  // 2. Construct the input and output.
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> outShape = {3, 3};

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  aclError ret = CreateInputs(
      selfShape, otherShape, outShape, &selfDeviceAddr, &otherDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator API.
  int64_t dim = 1;
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, out, dim, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 6. Destroy aclTensor.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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
