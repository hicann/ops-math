# aclnnTransConvolutionWeight

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/trans_data)

## Supported Products

| Product                                             | Supported|
|:------------------------------------------------| :------: |
| Ascend 950PR/Ascend 950DT         |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>   |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                      |    √     |
| <term>Atlas training products</term>                      |    ×     |

## Function

Description: Creates a weight tensor for computation performance affinity of the Convolution operator. This API must be used in conjunction with [aclnnCalculateConvolutionWeightSize](./aclnnCalculateConvolutionWeightSize.md).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnTransConvolutionWeightGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnTransConvolutionWeight** is called to perform computation.

```cpp
aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(
    const aclTensor* weightIn,
    bool             transposed,
    const int64_t    groups,
    aclTensor*       weightOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnTransConvolutionWeight(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnTransConvolutionWeightGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:211px">Parameter Name</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:266px">Description</th>
  <th style="width:308px">Instructions</th>
  <th style="width:240px">Data Type</th>
  <th style="width:110px">Data Format</th>
  <th style="width:150px">Dimension (shape) </th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>weightIn (uint64_t*) </td>
  <td>Input</td>
  <td>Indicates the weight tensor of a convolution operator to be processed.</td>
  <td>Empty tensor input is supported. If <code>weightIn</code> is an empty tensor, <code>weightOut</code> must also be an empty tensor.</td>
  <td>FLOAT16, FLOAT32</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>transposed (bool) </td>
  <td>Input</td>
  <td>Indicates whether the convolution is a transposed convolution.</td>
  <td>Currently, the value must be <code>false</code>.</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups (int64_t) </td>
  <td>Input</td>
  <td>Number of groups that connect input channels to output channels.</td>
  <td>The value range is [1, 65535].</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>weightOut (aclTensor*) </td>
  <td>Output</td>
  <td>Indicates the tensor obtained after the input weight is converted into private format.</td>
  <td>Empty tensor output is supported. If <code>weightOut</code> is an empty tensor, <code>weightIn</code> must also be an empty tensor.</td>
  <td>FLOAT16</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*) </td>
  <td>Output</td>
  <td>Size of the workspace required to be allocated on the device.</td>
  <td>It cannot be a null pointer. In the empty tensor scenario, the return value is <code>0</code>.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**) </td>
  <td>Output</td>
  <td>Operator executor, containing the operator computation process.</td>
  <td>It cannot be a null pointer.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **Returns**

  aclnnStatus: status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

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
      <td>The input is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type, format, and other parameters of the input and output tensors do not meet the expectation. For example, the input <code>weightIn</code> is not of the FLOAT16 or FLOAT32 type, or not in NCHW format; or the status of the empty tensor of <code>weightIn</code> is inconsistent with that of the empty tensor of <code>weightOut</code>.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransConvolutionWeight

- **Parameters**

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnTransConvolutionWeightGetWorkspaceSize</code>.</td>
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

  aclnnStatus: status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

## Constraints

- Deterministic computation:
  - `aclnnTransConvolutionWeight` defaults to a deterministic implementation.

- Only forward Conv2D is supported.
- Transposed convolution is not supported.
- Cache is not supported.
- Empty tensor is supported: When both `weightIn` and `weightOut` are empty tensors, the actual conversion is not performed and `workspaceSize` returns `0`.

## Example

The following is the sample code, which is for reference only. For details about the compilation and execution procedures, see <a href="../../../docs/en/context/compile_and_run_sample.md">Compile and Run Sample</a>.

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_trans_convolution_weight.h"
using namespace std;
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
  // (Boilerplate) Initialize the stream.
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
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateWeightAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, uint64_t &TransWeightSize)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call the transweight host API to compute the actual number of elements.
    aclIntArray* weightSize = aclCreateIntArray(shape.data(), shape.size());
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)>weightSizePtr(weightSize, aclDestroyIntArray);
    auto ret = aclnnCalculateConvolutionWeightSize(weightSize, false, 1, aclDataType::ACL_FLOAT16, &TransWeightSize);
    // Call aclrtMalloc to allocate memory on the device.
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnTransConvolutionWeightTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> inputShape = {1, 4, 16, 16};
  std::vector<int64_t> weightShape = {2, 4, 8, 8};
  std::vector<int64_t> biasShape = {2};
  std::vector<int64_t> outShape = {1, 2, 9, 9};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> inputHostData(1024, 1);
  std::vector<float> weightHostData(512, 1);
  std::vector<float> biasHostData(2, 1);
  std::vector<float> outHostData(162, 0);
  uint64_t transWeightSize = 0;

  // Create an input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>selfTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a weight aclTensor.
  ret = CreateWeightAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT,
    &weight, transWeightSize);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a bias aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a transweight aclTensor.
  void* transWeightDeviceAddr = nullptr;
  uint64_t size = transWeightSize * sizeof(float) / 2;
  ret = aclrtMalloc(&transWeightDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);return ret);

  std::vector<float> transData;
  transData.resize(transWeightSize * 2);

  // Call aclrtMemcpy to copy the data on the host to the memory transData.data() on the device.
  ret = aclrtMemcpy(transWeightDeviceAddr, size, transData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Compute the strides of the contiguous tensor.
  vector<int64_t> shape = weightShape;
  std::vector<int64_t> s(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      s[i] = shape[i + 1] * s[i + 1];
  }

  aclTensor* transWeight = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, s.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                shape.data(), shape.size(), transWeightDeviceAddr);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>transWeightTensorPtr(transWeight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> transWeightDeviceAddrAddrPtr(transWeightDeviceAddr, aclrtFree);

  // 3. Call aclnnTransConvolutionWeight.
  int8_t cubeMathType = 2; // USE_FP16
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  bool transposed = 0;
  uint64_t groups = 1;
  // Call TransWeight.
  ret = aclnnTransConvolutionWeightGetWorkspaceSize(weight, transposed, groups, transWeight,
    &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnTransConvolutionWeight.
  ret = aclnnTransConvolutionWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeight failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> convStrides = {1, 1};
  std::vector<int64_t> convPads = {0, 0};
  std::vector<int64_t> convOutPads = {1, 1};
  std::vector<int64_t> convDilations = {1, 1};

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), convStrides.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)>stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), convPads.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)>padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), convOutPads.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)>outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), convDilations.size());
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)>dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 4. Call aclnnConvolution.
  workspaceSize = 0;
  // Call the first-phase API of aclnnConvolution.
  ret = aclnnConvolutionGetWorkspaceSize(input, transWeight, bias, strides, pads, dilations, false, outPads, groups,
    out, cubeMathType, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnConvolution.
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnTransConvolutionWeightTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeightTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
