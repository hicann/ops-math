# aclnnSvd

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- API function: Computes the singular value decomposition of one or more matrices.

  When the dimension of the input tensor is greater than 2, the high-dimensional tensor is processed as a batch of matrices. For an input tensor with shape (..., M, N), the dimensions (...) before the second-to-last dimension are considered as batch dimensions, and singular value decomposition is performed independently on each (M, N) matrix.

- Formulas:

$$
\mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSvdGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSvd` is called to perform computation.

```c++
aclnnStatus aclnnSvdGetWorkspaceSize(
    const aclTensor *input,
    const bool       fullMatrices,
    const bool       computeUV,
    aclTensor       *sigma,
    aclTensor       *u,
    aclTensor       *v,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnSvd(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnSvdGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1544px"><colgroup>
  <col style="width: 141px">
  <col style="width: 120px">
  <col style="width: 354px">
  <col style="width: 461px">
  <col style="width: 141px">
  <col style="width: 101px">
  <col style="width: 81px">
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
      <th>Dimension</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Tensor on which singular value decomposition needs to be performed, corresponding to input in the formula.</td>
      <td>Empty tensors are not supported.<br>The shape dimension must be greater than 2.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>fullMatrices</td>
      <td>Input</td>
      <td>Input parameter, indicating whether to completely calculate the output tensors u and v.</td>
      <td>Controls whether to compute the complete SVD decomposition.<br>If this parameter is set to true, the complete u and v are output,<br>If this parameter is set to false, only the economical version is calculated to save memory and computing resources.<br>When the shape of the input is [..., M, N], K = min(M, N),<br>When this parameter is set to true, the output shape is as follows:<br>u: [..., M, M], sigma: [..., K], v: [..., N, N].<br>When this parameter is set to false, the output shape is as follows:<br>u: [..., M, K], sigma: [..., K], v: [..., N, K].</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>computeUV</td>
      <td>Input</td>
      <td>Input parameter, indicating whether to calculate the output tensors u and v.</td>
      <td>When this parameter is set to true, the output tensor sigma is calculated, and both u and v are calculated.<br>When this parameter is set to false, only sigma is calculated, and the shapes of u and v are not verified.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>Output</td>
      <td>Output tensor, corresponding to sigma in the formula.</td>
      <td>The shape needs to be deduced based on the shape of the input and the fullMatrices parameter.<br>The data type must be identical to that of `input`.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>u</td>
      <td>Output</td>
      <td>Output tensor, corresponding to U in the formula.</td>
      <td>The shape needs to be deduced based on the shape of the input and the fullMatrices parameter.<br>The data type must be identical to that of `input`.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Output</td>
      <td>Output tensor, corresponding to V in the formula.</td>
      <td>The shape needs to be deduced based on the shape of the input and the fullMatrices parameter.<br>The data type must be identical to that of `input`.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
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
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1145px"><colgroup>
  <col style="width: 296px">
  <col style="width: 135px">
  <col style="width: 714px">
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
      <td>input, sigma, u, or v is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of input, sigma, u, or v is not supported.</td>
    </tr>
    <tr>
      <td>The data types of input and sigma, u, and v are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input is less than 2 or greater than 8.</td>
    </tr>
    <tr>
      <td>The shape of sigma, u, and v is inconsistent with the shape derived from the input.</td>
    </tr>
  </tbody>
  </table>

## aclnnSvd

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnSvdGetWorkspaceSize.</td>
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
  - `aclnnSvd` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_svd.h"

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

struct SVDTensors {
  void* inputDeviceAddr = nullptr;
  void* uDeviceAddr = nullptr;
  void* sigmaDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* u = nullptr;
  aclTensor* sigma = nullptr;
  aclTensor* v = nullptr;
  std::vector<int64_t> uShape = {2, 2};
  std::vector<int64_t> sigmaShape = {2};
  std::vector<int64_t> vShape = {3, 3};
};

struct SVDWorkspace {
  void* addr = nullptr;
  uint64_t size = 0;
};


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
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
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



int SetupAndExecuteSVD(aclrtStream stream, SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;

  // Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> inputShape = {2, 3};
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> uHostData = {0, 0, 0, 0};
  std::vector<float> sigmaHostData = {0, 0};
  std::vector<float> vHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  bool fullMatrices = true;  
  bool computeUV = true;


  // Create an input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &tensors.inputDeviceAddr, aclDataType::ACL_FLOAT, &tensors.input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a u aclTensor.
  ret = CreateAclTensor(uHostData, tensors.uShape, &tensors.uDeviceAddr, aclDataType::ACL_FLOAT, &tensors.u);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a sigma aclTensor.
  ret = CreateAclTensor(sigmaHostData, tensors.sigmaShape, &tensors.sigmaDeviceAddr, aclDataType::ACL_FLOAT, &tensors.sigma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a v aclTensor.
  ret = CreateAclTensor(vHostData, tensors.vShape, &tensors.vDeviceAddr, aclDataType::ACL_FLOAT, &tensors.v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Call the CANN operator library API, which needs to be replaced with the actual API.
  aclOpExecutor* executor;
  // Call the first part of the aclnnSvdGetWorkspaceSize API.
  ret = aclnnSvdGetWorkspaceSize(tensors.input, fullMatrices, computeUV, tensors.sigma, tensors.u, tensors.v, 
                                 &workspace.size, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  workspace.addr = nullptr;
  if (workspace.size > 0) {
    ret = aclrtMalloc(&workspace.addr, workspace.size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second API of aclnnSvd.
  ret = aclnnSvd(workspace.addr, workspace.size, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvd failed. ERROR: %d\n", ret); return ret);

  // Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  return 0;
}

int ProcessAndCleanupSVD(SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;
  
  // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto uSize = GetShapeSize(tensors.uShape);
  std::vector<float> uData(uSize, 0);
  ret = aclrtMemcpy(uData.data(), uData.size() * sizeof(uData[0]), tensors.uDeviceAddr,
                    uSize * sizeof(uData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor U from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < uSize; i++) {
    LOG_PRINT("u[%ld] is: %f\n", i, uData[i]);
  }

  auto sigmaSize = GetShapeSize(tensors.sigmaShape);
  std::vector<float> sigmaData(sigmaSize, 0);
  ret = aclrtMemcpy(sigmaData.data(), sigmaData.size() * sizeof(sigmaData[0]), tensors.sigmaDeviceAddr,
                    sigmaSize * sizeof(sigmaData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor sigma from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < sigmaSize; i++) {
    LOG_PRINT("sigma[%ld] is: %f\n", i, sigmaData[i]);
  }
  
  auto vSize = GetShapeSize(tensors.vShape);
  std::vector<float> vData(vSize, 0);
  ret = aclrtMemcpy(vData.data(), vData.size() * sizeof(vData[0]), tensors.vDeviceAddr,
                    vSize * sizeof(vData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor V from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < vSize; i++) {
    LOG_PRINT("v[%ld] is: %f\n", i, vData[i]);
  }  
  
  // Destroy aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(tensors.input);
  aclDestroyTensor(tensors.u);
  aclDestroyTensor(tensors.sigma);
  aclDestroyTensor(tensors.v);

  // Release device resources.
  aclrtFree(tensors.inputDeviceAddr);
  aclrtFree(tensors.uDeviceAddr);
  aclrtFree(tensors.sigmaDeviceAddr);
  aclrtFree(tensors.vDeviceAddr);
  if (workspace.size > 0) {
    aclrtFree(workspace.addr);
  }
  
  return 0;
}

int ExecuteSVDOperator(aclrtStream stream) {
  SVDTensors tensors;
  SVDWorkspace workspace;
  
  auto ret = SetupAndExecuteSVD(stream, tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  ret = ProcessAndCleanupSVD(tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  return 0;
}


int main() {
  // Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  //Perform the GtScalar operation.
  ret = ExecuteSVDOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteGtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and terminate the ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
