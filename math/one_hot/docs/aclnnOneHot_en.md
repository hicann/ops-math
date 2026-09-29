# aclnnOneHot

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/one_hot)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    √    |

## Function

- Operator function: Performs one_hot calculation on an input self whose length is n to obtain an output out whose number of elements is n x k. The value of k is numClasses.
  The output elements meet the following formula:
  
  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
  $$

- Example:

  ```text
  Example 1:
  self = tensor([0, 1, 2, 0, 1])
  numClasses = 5
  onValue = tensor([1])
  offValue = tensor([0])
  axis=-1
  The shape of out is (5,5).
  out = tensor([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]])

  Example 2:
  self = tensor([0, 1, 2, 0, 1])
  numClasses = 1
  onValue = tensor([1])
  offValue = tensor([0])
  axis=-1
  The shape of out is (5,1).
  out = tensor([[1],
                [0],
                [0],
                [1],
                [0]])

  Example 3:
  self = tensor([0, 1, 2, 0, 1])
  numClasses = 0
  onValue = tensor([1])
  offValue = tensor([0])
  axis=-1
  The shape of out is (5,0).
  out = tensor([])

  Example 4:
  self = tensor([[1,2,3]]) # shape (1,3)
  numClasses = 4
  onValue = tensor([1])
  offValue = tensor([0])
  axis=1
  The shape of out is (1, 4, 3).
  out = tensor([[[0. 0. 0.]
                 [1. 0. 0.]
                 [0. 1. 0.]
                 [0. 0. 1.]]]) # shape (1, 4, 3)
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnOneHotGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnOneHot` is called to perform computation.

- `aclnnStatus aclnnOneHotGetWorkspaceSize(const aclTensor* self, int numClasses, const aclTensor* onValue, const aclTensor* offValue, int64_t axis, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnOneHot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnOneHotGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): index tensor, `self` in the formula, aclTensor on the device. The shape supports one to eight dimensions. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT32 or INT64.
  - numClasses (int, computation input): number of classes. The data type must be INT64. If `self` is an empty tensor, the value of `numClasses` must be greater than 0. If `self` is not an empty tensor, the value of `numClasses` must be greater than or equal to 0. If the value of `numClasses` is 0, an empty tensor is returned. If any element in `self` is greater than `numClasses`, these elements are encoded as all offValues.
  - `onValue` (aclTensor*, computation input): padding value at the index position, `onValue` in the formula, aclTensor on the device. The shape supports one to eight dimensions. Only the first element value is used for computation. The data type is the same as that of `out`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, INT32, or INT64.
  - `offValue` (aclTensor*, computation input): padding value at a non-index position, `offValue` in the formula, aclTensor on the device. The shape supports one to eight dimensions. Only the first element value is used for computation. The data type is the same as that of `out`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, INT32, or INT64.
  - `axis` (int64_t, computation input): dimension to insert an encoding vector. The minimum value is -1, and the maximum value is the number of `self` dimensions. If the value is -1, the encoding vector is inserted into the last dimension of `self`.
  - `out` (aclTensor*, computation output): One-hot tensor, output `out` in the formula, aclTensor on the device. The shape supports 1 to 8 dimensions and is the same as the shape after numClasses is inserted into the self shape on the axis. non-contiguous tensor(../../../docs/en/context/non_contiguous_tensor.md) are supported. [Data Format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, INT32, or INT64.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, onValue, offValue, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, onValue, offValue, or out is not supported.
                                        2. The data types of onValue and offValue are different from those of out.
                                        3. self is an empty tensor, and numClasses is less than or equal to 0.
                                        4. self is not an empty tensor, and numClasses is less than 0.
                                        5. The value of axis is less than -1.
                                        6. The value of axis is greater than the number of dimensions of self.
                                        7. The dimension of out is not one more than that of self.
                                        8. The shape of out is different from that after numClasses is inserted into the axis of the self shape.
                                        9. The shape of self, onValue, offValue, or out exceeds eight dimensions.
  ```

## aclnnOneHot

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnOneHotGetWorkspaceSize.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnOneHot` defaults to a deterministic implementation.
- <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>:
  - When the data type of `offValue` is INT64, the value of the first element can only be 0 or 1.
  - The size of the input self is selfSize, the size of the output out is outSize, and the size of the ub is ubSize. When the value of axis is 0, the following scenarios are not supported:
    - `selfSize * 3` < ubSize - 16K < `outSize * 3 / 2`

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_one_hot.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize resources.
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of consecutive tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
      // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> selfShape = {4, 2};
    int numClasses = 4;
    std::vector<int64_t> outShape = {4, 2, 4};
    std::vector<int64_t> onValueShape = {1};
    std::vector<int64_t> offValueShape = {1};
    void *selfDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *onValueDeviceAddr = nullptr;
    void *offValueDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    aclTensor *onValue = nullptr;
    aclTensor *offValue = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 3, 2, 1, 0};
    std::vector<int32_t> outHostData = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> onValueHostData = {1};
    std::vector<int32_t> offValueHostData = {0};
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an onValue aclTensor.
    ret = CreateAclTensor(onValueHostData, onValueShape, &onValueDeviceAddr, aclDataType::ACL_INT32, &onValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an offValue aclTensor.
    ret = CreateAclTensor(offValueHostData, offValueShape, &offValueDeviceAddr, aclDataType::ACL_INT32, &offValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    int64_t axis = -1;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnoneHot.
    ret = aclnnOneHotGetWorkspaceSize(self, numClasses, onValue, offValue, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnOnehot.
    ret = aclnnOneHot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHot failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(onValue);
    aclDestroyTensor(offValue);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(onValueDeviceAddr);
    aclrtFree(offValueDeviceAddr);
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
