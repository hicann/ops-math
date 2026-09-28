# aclnnCumprod&aclnnInplaceCumprod

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/cumprod)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT         |      ×   |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×    |
| <term>Atlas inference products</term>                      |     ×    |
| <term>Atlas training products</term>                      |     ×    |

## Function

- `cumprod` is used to calculate the cumulative product of the input tensor along the specified dimension. For example, if a tensor represents a series of values, `cumprod` can calculate the product sequence of these values from the start position to the current position.

- Formulas:

  - **One-dimensional tensor (vector)**
      For a one-dimensional tensor, the cumulative product $y=[y_1,y_2,y_3...,y_n]$ is calculated as follows:

      $y_1=x_1$
      $y_2=x_1 \times x_2$
      $y_3=x_1 \times x_2\times x_3$
      ...
      $y_n=x_1\times x_2\times x_3\times x_n$

      $y_i=\prod_{j=1}^ix_j$, where $i=1,2...,n$.

  - **High-dimensional tensor (using a two-dimensional tensor as an example, dim=0 along the row direction)**
    For a two-dimensional tensor: 

    $$
    X=\begin{bmatrix}x_{11}&x_{12}&...&x_{1m}\\x_{21}&x_{22}&...&x_{2m}\\...&...&...&...&\\x_{n1}&x_{n2}&...&x_{nm}&\end{bmatrix}
    $$

    The result tensor after calculation is as follows:

    $$
      Y=\begin{bmatrix}y_{11}&y_{12}&...&y_{1m}\\y_{21}&y_{22}&...&y_{2m}\\...&...&...&...&\\y_{n1}&y_{n2}&...&y_{nm}&\end{bmatrix}
    $$

    For the first column (j = 1):

    $$
    y_{i1}=x_{11}\times x_{21}\times ...\times x_{i1} (for i = 1, 2,..., n)
    $$

    Therefore, for any column j, the following rule also applies:

    $$
    y_{ij}=\prod_{k=1}^{i} x_{kj}
    $$

  - **High-dimensional tensor (using a two-dimensional tensor as an example, dim=1 along the column direction)**
    Therefore, for any column j, the following rule also applies:

    $$
    y_{ij}=\prod_{k=1}^{j} x_{ik}
    $$
  
  - **Other parameters can be deduced based on the preceding rules.**

## Prototype

aclnnCumprod and aclnnInplaceCumprod provide the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.

- aclnnCumprod: You need to create an output tensor object to store the computation result.
- aclnnInplaceCumprod: You do not need to create an output tensor object. Instead, the computation result is directly stored in the memory of the input tensor.

Each operator is divided into two phases (../../../docs/en/context/two_phase_api.md). You must call aclnnCumprodGetWorkspaceSize or aclnnInplaceCumprodGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnCumprod or aclnnInplaceCumprod to perform the computation.

```Cpp
aclnnStatus aclnnCumprodGetWorkspaceSize(
  const aclTensor*   input,
  const aclScalar*   dim,
  const aclDataType  dtype,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnCumprod(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

```Cpp
aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(
  aclTensor*       input,
  const aclScalar* dim,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceCumprod(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnCumprodGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
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
      <td>input (aclTensor*) </td>
      <td>Input</td>
      <td>Data for which the accumulated product needs to be calculated.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>The dimension cannot exceed 8.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (aclScalar*) </td>
      <td>Input</td>
      <td> specifies the dimension for calculating the cumulative product. For a two-dimensional tensor, dim=0 indicates that the product is calculated along the row direction, and dim=1 indicates that the product is calculated along the column direction.</td>
      <td>Value range: [-rank (input), rank (input))</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype (aclDataType) </td>
      <td>Input</td>
      <td>Specifies the data type of the input in the calculation process.</td>
      <td><ul><li>If the value is ACL_DT_UNDEFINED, the original type of the input is used for calculation. </li><li> If a specific type is specified (the data type must be supported by the input), the input is converted to this type before calculation.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>Accumulated product result.</td>
      <td><ul><li>When dtype is ACL_DT_UNDEFINED, the data type must be the same as that of the input. </li><li>When dtype is specified, the data type must be the same as that of dtype. </li><li>The shape of out must be the same as that of the input.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
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

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>The input and dim pointers are null.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of the input and dim are not supported.</td>
    </tr>
    <tr>
      <td>The input and dim shape constraints are not met.</td>
    </tr>
    <tr>
      <td>The shapes of the output and input are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnCumprod

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnCumprodGetWorkspaceSize API.</td>
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

## aclnnInplaceCumprodGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
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
      <td>input (aclTensor*) </td>
      <td>Input/Output</td>
      <td>Data and result for which the cumulative product needs to be calculated.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>The dimension cannot exceed 8.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (aclScalar*) </td>
      <td>Input</td>
      <td>Specifies the dimension along which the cumulative product is computed. For a 2D tensor, dim=0 indicates that the computation is performed along the row direction, and dim=1 indicates that the computation is performed along the column direction.</td>
      <td>Value range: [–rank(input), rank(input)].</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
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

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>The input and dim pointers are null.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and formats of the input and dim are not supported.</td>
    </tr>
    <tr>
      <td>The input and dim shape constraints are not met.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceCumprod

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnInplaceCumprodGetWorkspaceSize.</td>
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
  - `aclnnCumprod` and `aclnnInplaceCumprod` default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumprod.h"

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++)
    {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
}

template<typename T>
void PrintOutFloatResult(std::vector<T> &shape, void **deviceAddr, const char *name)
{
    std::vector<float> resultData(shape.size(), 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, shape.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < shape.size(); i++)
    {
        LOG_PRINT("result var %s[%ld] is: %f\n", name, i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // (Boilerplate) Initialize AscendCL.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclScalar(aclDataType dataType, T &hostData, aclScalar **scalar)
{
    *scalar = aclCreateScalar(&hostData, dataType);
    if (*scalar == nullptr)
    {
        return -1;
    }
    return 0;
}

int main()
{
    //1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs. Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    void *xDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    std::vector<int64_t> xShape = {3};
    std::vector<int64_t> xHostData = {1,2,3};
    // Create the original input x.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT64, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an axis aclScalar.
    int32_t axis_value = 0;
    aclScalar *axis = nullptr;
    ret = CreateAclScalar(aclDataType::ACL_INT32, axis_value, &axis);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create a result aclTensor.
    std::vector<int64_t> resultHostData(3, 0);
    std::vector<int64_t> resultShape = {3};
    void *resultDeviceAddr = nullptr;
    aclTensor *result = nullptr;
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_INT64, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    aclDataType dtype = ACL_INT64;
    void *workspaceAddr = nullptr;
    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnCumprod.
    ret = aclnnCumprodGetWorkspaceSize(input, axis, dtype, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprod allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnCumprod.
    ret = aclnnCumprod(workspaceAddr, workspaceSize, executor, stream);
    // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnInplaceCumprod.
    ret = aclnnInplaceCumprodGetWorkspaceSize(input, axis, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprod allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnInplaceCumprod.
    ret = aclnnInplaceCumprod(workspaceAddr, workspaceSize, executor, stream);
    // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(resultShape, &xDeviceAddr);

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(input);
    aclDestroyScalar(axis);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
