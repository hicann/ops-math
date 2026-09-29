# aclnnCumprod&aclnnInplaceCumprod

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/cumprod)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: The `aclnnCumprod` API is added. The `cumprod` function calculates the cumulative product of the input tensor along a specified dimension. For example, if a tensor represents a series of values, `cumprod` can calculate the product sequence of these values from the start position to the current position.

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

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnCumprodGetWorkspaceSize` or `aclnnInplaceCumprodGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnCumprod` or `aclnnInplaceCumprod` is called to perform computation.

- `aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor* input, const aclScalar* dim, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnCumprod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

- `aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor* input, const aclScalar* dim, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnInplaceCumprod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnCumprodGetWorkspaceSize

* **Parameters**
  * `input` (aclTensor*, computation input): current input value (indicating the data for which the cumulative product needs to be calculated), `aclTensor` on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. Empty tensors are supported.
The data type can be FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, or UINT64. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `dim` (aclScalar*, computation input): current input value, specifying the dimension for calculating the cumulative product. For a two-dimensional tensor, `dim=0` indicates that the calculation is performed along the row direction, and `dim=1` indicates that the calculation is performed along the column direction. It is an `aclScalar` on the device. The value range is [-rank(input), rank(input)). The data type can be INT32.
  * `dtype` (aclDataType, computation input): data type of the input during computation. If the value is `ACL_DT_UNDEFINED`, the original type of the input is used for computation. If a specific type is specified (within the range of data types supported by the input), the input is converted to this type before computation.
  * `out` (aclTensor*, computation output): cumulative product result, `aclTensor` on the device. The [data format](../../../docs/en/context/data_format.md) can be ND. When `dtype` is `ACL_DT_UNDEFINED`, the data type must be the same as that of the input. When `dtype` is specified, the data type must be the same as that of `dtype`. The shape of `out` must be identical to that of `input`.
  * `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed input or dim is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of input or dim is not supported.
                                    2. dim is incompatible with the shape of input.
                                    3. The shapes of out and input are inconsistent.
  ```

## aclnnCumprod

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnCumprodGetWorkspaceSize`.
  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
## aclnnInplaceCumprodGetWorkspaceSize

* **Parameters**
  * `input` (aclTensor*, computation input | computation output): input and output tensor for the cumulative product, `aclTensor` on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. Empty tensors are not supported. The data type can be FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, or UINT64. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `dim` (aclScalar*, computation input): dimension for calculating the cumulative product. For a two-dimensional tensor, `dim=0` indicates that the calculation is performed along the row direction, and `dim=1` indicates that the calculation is performed along the column direction. It is an `aclScalar` on the device. The value range is [-rank(x), rank(x)]. The data type can be INT32.
  * `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*, output): operator executor, containing the operator computation process.

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed input or dim is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of input or dim is not supported.
                                    2. dim is incompatible with the shape of input.
  ```

## aclnnInplaceCumprod

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceCumprodGetWorkspaceSize`.
  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

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
