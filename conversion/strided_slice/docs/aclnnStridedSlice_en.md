# aclnnStridedSlice

[📄 View Source Code](https://gitcode.com/cann/ops-math/tree/master/conversion/strided_slice)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Function: Extracts a sub-tensor from the input tensor based on the specified start position, end position, and stride.
- The formula is as follows: Extracts the sub-tensor $out$ from the input tensor $self$ in the specified dimension $dim$ based on the specified start position $begin$, end position $end$, and stride $strides$.
  $begin$ and $end$ can be set to values other than $[0, self.shape[dim]]$. After the values are set, they are converted into valid values according to the following formula. Assume that self.shape[dim] = N:

  $$
  begin = \begin{cases}
  &0, & if\;begin < -N \\
  &N, & if\;begin >= N\\
  &(begin+N) \% N, & otherwise
  \end{cases}
  $$

  $$
  end = \begin{cases}
  &N, & if\; end >= N\\
  &begin, & else\;if\;  (end+N)\%N < begin \\
  &(end+N)\%N, & otherwise\\
  \end{cases}
  $$

  $out.shape$ is the same as $self.shape$ except on the `dim` axis.

  $$
  out.shape[dim] = ⌊\frac{end - begin + strides - 1}{strides}⌋
  $$

  If the mask parameter exists, outshape is further calculated according to the following rules:
  $beginMask$ specifies that the $begin$ of the index dimension corresponding to the $bit$ whose value is 1 is ignored,
  $endMask$ specifies that the $end$ of the index dimension corresponding to the $bit$ whose value is 1 is ignored,
  $ellipsisMask$ selects all subsequent dimensions starting from the index dimension corresponding to the $bit$ whose value is 1 until the specified $begin$ is encountered,
  $newAxisMask$ specifies that the $shape$ whose dimension is 1 is added to the index dimension corresponding to the $bit$ whose value is 1,
  $shrinkAxisMask$ specifies that the index dimension corresponding to the $bit$ whose value is 1 is forcibly reduced to 1.

## Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). You must call aclnnStridedSliceGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnStridedSlice to perform the computation.

```Cpp
aclnnStatus aclnnStridedSliceGetWorkspaceSize(
    const aclTensor   *self, 
    const aclIntArray *begin, 
    const aclIntArray *end, 
    const aclIntArray *strides,
    int64_t           beginMask, 
    int64_t           endMask, 
    int64_t           ellipsisMask, 
    int64_t           newAxisMask, 
    int64_t           shrinkAxisMask,
    aclTensor         *out, 
    uint64_t          *workspaceSize, 
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnStridedSlice(
    void          *workspace,
    uint64_t       workspaceSize, 
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnStridedSliceGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>The data type of self is the same as that of out.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>begin</td>
      <td>Input</td>
      <td>Start value of each dimension.</td>
      <td>The lengths of the begin, end, and strides arrays must be the same.</td>
      <td>INT32 or INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end</td>
      <td>Input</td>
      <td>End value of each dimension.</td>
      <td>The lengths of the begin, end, and strides arrays must be the same.</td>
      <td>INT32 or INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input</td>
      <td>Value span of each point in each dimension.</td>
      <td>The lengths of the begin, end, and strides arrays must be the same. The strides value cannot be 0.</td>
      <td>INT32 or INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beginMask</td> 
      <td>Input</td>
      <td>The index dimension whose bit is 1 is ignored. The begin value is ignored.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>endMask</td>
      <td>Input</td>
      <td>The end index of the dimension corresponding to the bit set to 1 is ignored.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ellipsisMask</td>
      <td>Input</td>
      <td>Selects all dimensions starting from the dimension corresponding to the bit set to 1 until the specified begin is encountered.</td>
      <td>Only one bit in ellipsisMask can be set to 1.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>newAxisMask</td>
      <td>Input</td>
      <td>Adds a shape with the dimension size of 1 to the dimension corresponding to the bit set to 1.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shrinkAxisMask</td>
      <td>Input</td>
      <td>Forcibly reduces the dimension corresponding to the bit set to 1 to 1.</td> 
      <td>The stride corresponding to the index whose bit is set to 1 in shrinkAxisMask must be greater than 0, that is, a positive number.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The data type of self is the same as that of out.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL, COMPLEX32, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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

- **Return value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
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
      <td>The input self, begin, end, strides, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of self or out is not supported.</td>
    </tr>
    <tr>
      <td>The data types of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is greater than 8.</td>
    </tr>
    <tr>
      <td>The lengths of begin, end, and strides are inconsistent.</td>
    </tr>
    <tr>
      <td>strides contains elements that are equal to 0.</td>
    </tr>
    <tr>
      <td>The data dimension of out is different from that of infershape.</td>
    </tr>
    <tr>
      <td>The product model is not supported.</td>
    </tr>
    <tr>
      <td>ellipsisMask has more than one bit set to 1.</td>
    </tr>
    <tr>
      <td>The stride corresponding to the index of the bit set to 1 in shrinkAxisMask is less than 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnStridedSlice

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
      <td>Size of the workspace allocated on the device, which is obtained by the aclnnStridedSliceGetWorkspaceSize API.</td>
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

- **Return value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnStridedSlice is implemented in deterministic mode by default.

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
#include "aclnnop/aclnn_strided_slice.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
    // (Boilerplate) Perform initialization.
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
    // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
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

int main()
{
      // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> selfShape = {4, 3};
    std::vector<int64_t> outShape = {2, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* begin = nullptr;
    aclIntArray* end = nullptr;
    aclIntArray* strides = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int64_t> beginData = {1, 1};
    std::vector<int64_t> endData = {3, 3};
    std::vector<int64_t> stridesData = {1, 1};
    std::vector<float> outHostData(4, 0);
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an aclIntArray.
    begin = aclCreateIntArray(beginData.data(), 2);
    CHECK_RET(begin != nullptr, return ret);
    end = aclCreateIntArray(endData.data(), 2);
    CHECK_RET(end != nullptr, return ret);
    strides = aclCreateIntArray(stridesData.data(), 2);
    CHECK_RET(strides != nullptr, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnCast.
    ret = aclnnStridedSliceGetWorkspaceSize(
        self, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, out, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnCast.
    ret = aclnnStridedSlice(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSlice failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(self);
    aclDestroyIntArray(begin);
    aclDestroyIntArray(end);
    aclDestroyIntArray(strides);
    aclDestroyTensor(out);

    // 7. Release device resources.
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
