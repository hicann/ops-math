# aclStft

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Function: Computes the Fourier transform of the input in a sliding window.
- Formula:

  - When `normalized` is set to `false`:

    $$
    X[w,m]=\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft})
    $$

  - When `normalized` is set to `true`:
  
    $$
    X[w,m]=\frac{1}{\sqrt{nFft}}(\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft}))
    $$

  Where:
  - FFT works on frequency `$w$`.
  - `$m$` is the index of the sliding window.
  - `$self$` is a 1D or 2D tensor. When $self$ is 1D, there is only one time sequence. When `$self$` is 2D, there are multiple time sequences.
  - `$hopLength$` is the sliding window size.
  - `$window$` is a 1D tensor, which is the window function (for example, hann_window) of STFT. Its length is `$winLength$`.
  - `$exp(-j*\frac{2{\pi}wk}{nFft})$` is the rotation factor.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclStftGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclStft` is called to perform computation.

```Cpp
aclnnStatus aclStftGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *windowOptional,
  aclTensor       *out,
  int64_t          nFft,
  int64_t          hopLength,
  int64_t          winLength,
  bool             normalized,
  bool             onesided,
  bool             returnComplex,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnExpSegsum(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclStftGetWorkspaceSize

- **Parameters**

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
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input for the computation, corresponding to `self` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [L]/[B, L]. L indicates the length of the time sequence, and B indicates the number of time sequences.</li></ul></td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>windowOptional</td>
      <td>Input</td>
      <td>The value must be a 1D tensor, corresponding to `window` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type must be the same as that of `self`. </li><li>The shape is [winLength]. `winLength` indicates the length of the STFT window function.</li></ul></td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Fourier transform result of self within the window, corresponding to `X` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>If `returnComplex` is set to `True`, `out` is a complex tensor with the shape of [N, T] or [B, N, T]. </li><li>If `returnComplex` is set to `False`, `out` is a real tensor with the shape of [N, T, 2] or [B, N, T, 2]. </li></ul>N = nFft (onesided = False) or (nFft // 2 + 1) (onesided = True); T is the number of sliding windows, and T = (L – nFft) // hopLength + 1.</td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>nFft</td>
      <td>Input</td>
      <td>Number of FFT points (greater than 0), corresponding to `nFft` in the formula.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hopLength</td>
      <td>Input</td>
      <td>Interval of the sliding window (greater than 0), corresponding to `hopLength` in the formula.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLength</td>
      <td>Input</td>
      <td>Window size (greater than 0), corresponding to `winLength` in the formula.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>normalized</td>
      <td>Input</td>
      <td>Whether to normalize the Fourier transform result.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onesided</td>
      <td>Input</td>
      <td>Whether to return all results or half of the results.</td>
      <td>When the data type of input `self` is COMPLEX64 or COMPLEX128, this parameter can only be set to `False`.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnComplex</td>
      <td>Input</td>
      <td>Whether the return value is a complex tensor or a tensor with the real and imaginary components separated.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:

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
      <td>The passed `self` or `out` is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data format of `self` is not supported.</td>
    <tr>
      <td>The data types of `self` and `windowOptional` are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of `self`, `windowOptional`, and `out` are not supported.</td>
    </tr>
    </tr>
      <td>`nFft`, `hopLength`, and `winLength` have invalid values.</td>
    <tr>
      <td>The dimensions of `self`, `windowOptional`, and `out` are not supported.</td>
    </tr>
    <tr>
      <td>When the data type of input `self` is COMPLEX64 or COMPLEX128, the value of `onesided` is `True`.</td>
    </tr>
  </tbody></table>

## aclStft

- **Parameters**

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API `aclStftGetWorkspaceSize`.</td>
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

- The input `self` is different from that of the PyTorch interface. The input `self` of the PyTorch interface is the original input, while the input `self` of `aclStftGetWorkspaceSize` is the result after the original input is padded by PyTorch.
- When the shape of input `self` is [B, L], if the calculation result of the following formula is large, the calculation of the current API may time out.
  
  $$
  B * ((L - nFft) / hopLength + 1) * nFft
  $$

- nFft ≤ L
- winLength ≤ nFft
- normalized = True:
  
  $$
  STFT(w,m)=\frac{1}{\sqrt{N}}X[w,m]
  $$

- The mappings between the input and output data types of `self`, `windowOptional`, `returnComplex`, and `out` are as follows:
  
  |self|windowOptional|returnComplex|out|
  |-------|-------|-------|-------|
  |FLOAT32| FLOAT32        | True          | COMPLEX64  |
  | DOUBLE     | DOUBLE         | True          | COMPLEX128 |
  |COMPLEX64|COMPLEX64|True|COMPLEX64|
  |COMPLEX128|COMPLEX128|True|COMPLEX128|
  |FLOAT32| FLOAT32        | False     | FLOAT32 |
  | DOUBLE     | DOUBLE         | False     | DOUBLE |
  |COMPLEX64|COMPLEX64|False|FLOAT32|
  |COMPLEX128|COMPLEX128|False|DOUBLE|

- Deterministic computing:
  - `aclStft` defaults to a deterministic implementation.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_stft.h"

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
  // Call `aclrtMalloc` to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call `aclrtMemcpy` to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  // Call `aclCreateTensor` to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {5};
  std::vector<int64_t> windowShape = {4};
  std::vector<int64_t> outShape = {3, 1, 2};
  void* selfDeviceAddr = nullptr;
  void* windowDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* window = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 6, 8, 5, 7};
  std::vector<float> windowHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  // Create a window aclTensor.
  ret = CreateAclTensor(windowHostData, windowShape, &windowDeviceAddr, aclDataType::ACL_FLOAT, &window);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int n_fft = 4;
  int hop_length = 2;
  int win_length = 4;
  bool normalized = false;
  bool onesided = true;
  bool returnComplex = false;
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of `aclStft`.
  ret = aclStftGetWorkspaceSize(self, window, out, n_fft, hop_length, win_length, normalized, onesided, returnComplex, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclStftGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of `aclStft`.
  ret = aclStft(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclStft failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. Release `aclTensor`. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(window);
  aclDestroyTensor(out);
  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(windowDeviceAddr);
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
