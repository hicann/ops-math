# aclnnAddLora

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/add_lora)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     √    |
|  <term>Atlas training products</term>   |     ×    |

## Function Description

- Description:

  Multiplies the input `x` by the corresponding `weightA` and `weightB` based on the input index values in `indices`, and accumulates the results into the input `y`to produce the output.

- Formulas:

  Given an input tensor `x` with the last dimension length of 2d, the AddLora function performs the following operations:

  1. Rearrange `x` based on the index values in `indices`. Elements of `x` corresponding to the same weight group are placed together.
  
  2. Loop through each LoRA group and perform matrix multiplication using its corresponding `x` element and `weightA`.

     $$
     Z1 = x_{i} \cdot weightA[i, layerIdx, :, :]
     $$
  3. Multiply the resultant `Z1` with `weightB`.

     $$
     Z2 = Z1 \cdot weightB[i, layerIdx, :, :] \times scale
     $$
  4. Finally, accumulate `Z2` into `y` to obtain the output.

     $$
     \text{out} = y[:, yOffset: yOffset+ySliceSize] + Z2
     $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAddLoraGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the computation process. Then, `aclnnAddLora` is called to perform computation.

```Cpp
aclnnStatus aclnnAddLoraGetWorkspaceSize(
    const aclTensor *y,
    const aclTensor *x,
    const aclTensor *weightB,
    const aclTensor *indices,
    const aclTensor *weightAOptional,
    int64_t          layerIdx,
    double           scale,
    int64_t          yOffset,
    int64_t          ySliceSize,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLora(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAddLoraGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>y (aclTensor*) </td>
      <td>Input</td>
      <td><code>y</code> in the formula, which is a tensor to be updated by accumulation.</td>
      <td><ul><li>Its shape has two dimensions: [B, H3], where <code>H3</code> is a multiple of 16 and must range from 1 to 131072. </li><li>The first dimension must be the same as the that of <code>x</code>. This shared dimension is denoted by <code>B</code>. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*) </td>
      <td>Input</td>
      <td><code>x</code> in the formula, which indicates the input tensor before grouping.</td>
      <td><ul><li>Its shape has two dimensions: [B, H1], where <code>H1</code> is a multiple of 16. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightB (aclTensor*) </td>
      <td>Input</td>
      <td><code>weightB</code> in the formula, which indicates the second weight matrix used for matrix multiplication.</td>
      <td><ul><li>Its shape has four dimensions: [W, L, H2, R], where the third dimension must be less than or equal to the second dimension of y (H2 ≤ H3). <code>H2</code> must be a multiple of 16 and must range from 1 to 131072. The value of <code>R</code> must range from 1 to 128 and be a multiple of 16. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND and NZ</td>
      <td>4</td>
      <td>√</td>
    </tr>
     <tr>
      <td>indices (aclTensor*) </td>
      <td>Input</td>
      <td>input <code>indices</code> in the formula, which indicates the group index of the input <code>x</code>.</td>
      <td><ul><li>Its shape has one dimension: [B]. </li><li>The first dimension must be the same as that of both <code>x</code> and <code>y</code>. This shared dimension is denoted by <code>B</code>. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightAOptional (aclTensor*) </td>
      <td>Input</td>
      <td><code>weightA</code> in the formula, which indicates the first weight matrix used for matrix multiplication. If this parameter is empty, the first matrix multiplication is skipped.</td>
      <td><ul><li>Its shape has four dimensions: [W, L, R, H1]. The first two dimensions (<code>W</code> and <code>L</code>) must be the same as those of <code>weightB</code>, where <code>W</code> ranges from 1 to 32, and <code>L</code> ranges from 1 to 32. The third dimension must be the same as the fourth dimension of <code>weightB</code>, and both are denoted by <code>R</code>. The fourth dimension must be the same as the second dimension of <code>x</code>, and both are denoted by <code>H1</code>, which must be a multiple of 16. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND and NZ</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>layerIdx (int64_t) </td>
      <td>Input</td>
      <td><code>layerIdx</code> in the formula, which indicates the layer index.</td>
      <td>The value must be less than the second dimension <code>L</code> of <code>weightB</code>.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (double) </td>
      <td>Input</td>
      <td><code>scale</code> in the formula, which indicates the scaling coefficient.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset (int64_t) </td>
      <td>Input</td>
      <td><code>yOffset</code> in the formula, which indicates the offset for <code>y</code> update.</td>
      <td>The value must be less than the second dimension <code>H3</code> of <code>y</code>.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ySliceSize (int64_t) </td>
      <td>Input</td>
      <td><code>ySliceSize</code> in the formula, which indicates the slice size of <code>y</code> to be updated.</td>
      <td>The value must be less than or equal to the second dimension <code>H3</code> of <code>y</code>.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td><code>out</code> in the formula, which indicates the output tensor.</td>
      <td><ul><li>The output data type is the same as the input data type. </li><li>The shape of <code>out</code> has the same dimensions as the shape of the input <code>y</code>.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, covering the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data formats of <code>weightB</code> and <code>weightAOptional</code> can be ND.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameters (<code>x</code>, <code>y</code>, <code>weightB</code>, and <code>indices</code>) or the output parameter <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of the input parameters (<code>x</code>, <code>y</code>, <code>weightB</code>, and <code>indices</code>) or the data type of the output parameter is not supported.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>The shapes of multiple input tensors do not match. For details, see <a href="##parameters">Parameters</a>.</td>
    </tr>
    <tr>
      <td>The shape of the input tensor is not supported. For details, see <a href="##parameters">Parameters</a>.</td>
    </tr>
  </tbody>
  </table>

## aclnnAddLora

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnAddLoraGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, covering the operator computation process.</td>
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
  - `aclnnAddLora` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_lora.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
    // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int32_t batchSize = 1;
  int32_t H1 = 16;
  int32_t H2 = 16;
  int32_t R = 16;
  int32_t loraNum = 1;
  int32_t layerNum = 1;

  std::vector<int64_t> xShape = {batchSize, H1};
  std::vector<int64_t> yShape = {batchSize, H2};
  std::vector<int64_t> weightBShape = {loraNum, layerNum, H2, R};
  std::vector<int64_t> indicesShape = {batchSize};
  std::vector<int64_t> weightAShape = {loraNum, layerNum, R, H1};
  std::vector<int64_t> outShape = {batchSize, H2};

  std::vector<float> xHostData(batchSize * H1, 1);
  std::vector<float> yHostData(batchSize * H2, 1);
  std::vector<float> weightBHostData(loraNum * layerNum * H2 * R, 1);
  std::vector<float> indicesHostData(batchSize, 0);
  std::vector<float> weightAHostData(loraNum * layerNum * R * H1, 1);
  std::vector<float> outHostData(batchSize * H2, 1);

  void* xInputDeviceAddr = nullptr;
  void* yInputDeviceAddr = nullptr;
  void* weightBInputDeviceAddr = nullptr;
  void* indicesInputDeviceAddr = nullptr;
  void* weightAInputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* xInput = nullptr;
  aclTensor* yInput = nullptr;
  aclTensor* weightBInput = nullptr;
  aclTensor* indicesInput = nullptr;
  aclTensor* weightAInput = nullptr;
  aclTensor* out = nullptr;

  // Create input x.
  ret = CreateAclTensor(xHostData, xShape, &xInputDeviceAddr, aclDataType::ACL_FLOAT16, &xInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input y.
  ret = CreateAclTensor(yHostData, yShape, &yInputDeviceAddr, aclDataType::ACL_FLOAT16, &yInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input weightB.
  ret = CreateAclTensor(weightBHostData, weightBShape, &weightBInputDeviceAddr, aclDataType::ACL_FLOAT16, &weightBInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input indices.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesInputDeviceAddr, aclDataType::ACL_INT32, &indicesInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input weightA.
  ret = CreateAclTensor(weightAHostData, weightAShape, &weightAInputDeviceAddr, aclDataType::ACL_FLOAT16, &weightAInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t layer_idx = 0;
  double scale = 1.0;
  int64_t y_offset = 0;
  int64_t y_slice_size = H2;

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnAddLora.
  ret = aclnnAddLoraGetWorkspaceSize(yInput, xInput, weightBInput, indicesInput, weightAInput, layer_idx, scale, y_offset, y_slice_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLoraGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnAddLora.
  ret = aclnnAddLora(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLora failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(xInput);
  aclDestroyTensor(yInput);
  aclDestroyTensor(weightBInput);
  aclDestroyTensor(indicesInput);
  aclDestroyTensor(weightAInput);
  aclDestroyTensor(out);

  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(xInputDeviceAddr);
  aclrtFree(yInputDeviceAddr);
  aclrtFree(weightBInputDeviceAddr);
  aclrtFree(indicesInputDeviceAddr);
  aclrtFree(weightAInputDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
