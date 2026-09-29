# aclnnExpSegsum

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/segsum)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs segmentation and computation to generate a semi-separable matrix with a zero diagonal and an upper triangle being `-inf`.
- Formula (using 4D input as an example):

  1. Reshape `self` from (N1, N2, N3, N4) to (N1, N2, N3, N4, 1).
  2. Perform broadcasting obtain (N1, N2, N3, N4, N4).
  3. Generate a BOOL triangular matrix A of shape (N4, N4), where the upper triangle is `True`, the lower triangle is `False`, and the diagonal is `True`.
  4. Fill the elements in `self` with `0` at the positions where the corresponding elements in matrix A are `True`.

      $$
      self_i=
      \begin{cases}self_i,\quad A_i==False
      \\0, \quad A_i==True
      \end{cases}
      $$

  5. Perform a cumulative sum along the second-to-last dimension of `self`. For an element from a dimensional perspective, other dimension indices remain unchanged while the current index increases. $selfTemp\_{i}$ is the element at the corresponding position in the output tensor.

     $$
     selfTemp_{i} = self_{1} + self_{2} + self_{3} + ...... + self_{i}
     $$

  6. Generate a BOOL triangular matrix B of shape (N4, N4), where the upper triangle is `True`, the lower triangle is `False`, and the diagonal is `False`.
  7. Fill the elements in `selfTemp` with `-inf` at the positions where the corresponding elements in matrix B are `True`.

     $$
     out_i=
     \begin{cases}selfTemp_i,\quad B_i==False
     \\-inf, \quad B_i==True
     \end{cases}
     $$

  8. Compute the exponent of each element in `selfTemp`.

     $$
     out_i=e^{selfTemp_i}
     $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnExpSegsumGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnExpSegsum` is called to perform computation.

```Cpp
aclnnStatus aclnnExpSegsumGetWorkspaceSize(
  const aclTensor   *self,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnExpSegsum(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnExpSegsumGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>Input for segmentation and computation, corresponding to self in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>If the last axis is too large, the output will occupy excessive space. For example, if the last axis of the input is N, the output memory will be N times that of the input.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output after segmentation and computation, corresponding to out in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of self. </li><li>The output dimension must be one greater than the input dimension. <ul><li>For a 3D input self, the sizes of the first three dimensions of the output remain the same as those of self, while the size of the last dimension matches the third dimension. </li><li>For a 4D input self, the sizes of the first four dimensions of the output remain the same as those of self, while the size of the last dimension matches the fourth dimension.</li></ul></li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>4–5</td>
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
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API performs input parameter validation. The following errors may be returned:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of self or out is not supported.</td>
    </tr>
    <tr>
      <td>The shape of self or out does not meet the parameter requirements.</td>
    </tr>
  </tbody></table>

## aclnnExpSegsum

- **Parameters:**
  
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnExpSegsumGetWorkspaceSize.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnExpSegsum` defaults to a deterministic implementation.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_segsum.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Perform the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> selfShape = {1, 1, 1, 4};
    std::vector<int64_t> outShape = {1, 1, 1, 4, 4};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3};
    std::vector<int64_t> outputSizeHostData = {16};
    std::vector<float> outHostData(16, 0);

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnExpSegsum.
    ret = aclnnExpSegsumGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnExpSegsum.
    ret = aclnnExpSegsum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsum failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);

    // 7. Free device resources. Modify the code based on the API definition.
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
