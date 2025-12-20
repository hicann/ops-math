# aclnnCumprod&aclnnInplaceCumprod

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/cumprod)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：新增aclnnCumprod接口，`cumprod`函数用于计算输入张量在指定维度上的累积乘积。例如，如果有一个张量表示一系列的数值，`cumprod`可以计算出这些数值从开始位置到当前位置的乘积序列。

- 计算公式：

  - **一维张量（向量）情况**
      当对于一维张量，累积乘积$y=[y_1,y_2,y_3...,y_n]$的计算公式为:

      $y_1=x_1$
      $y_2=x_1 \times x_2$
      $y_3=x_1 \times x_2\times x_3$
      ...
      $y_n=x_1\times x_2\times x_3\times x_n$

      用数学公式表示$y_i=\prod_{j=1}^ix_j, 其中i=1,2...,n$。

  - **高维张量情况（以二维张量为例， dim=0 沿行方向）**
    对于二维张量：  
    $$
    X=\begin{bmatrix}x_{11}&x_{12}&...&x_{1m}\\x_{21}&x_{22}&...&x_{2m}\\...&...&...&...&\\x_{n1}&x_{n2}&...&x_{nm}&\end{bmatrix}
    $$
    计算后的结果张量：
    $$
      Y=\begin{bmatrix}y_{11}&y_{12}&...&y_{1m}\\y_{21}&y_{22}&...&y_{2m}\\...&...&...&...&\\y_{n1}&y_{n2}&...&y_{nm}&\end{bmatrix}
    $$
    对于第一列(j=1):
    $$
    y_{i1}=x_{11}\times x_{21}\times ...\times x_{i1}(对于i=1,2,....n)
    $$
    所以对于任意列j，也有类似规律， 即:
    $$
    y_{ij}=\prod_{k=1}^{i} x_{kj}
    $$

  - **高维张量情况（以二维张量为例， dim=1 沿列方向情况）**
    所以对于任意列j，也有类似规律， 即:
    $$
    y_{ij}=\prod_{k=1}^{j} x_{ik}
    $$
  
  - **其它参数可以类似地根据上述规则进行推导**

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCumprodGetWorkspaceSize”或者“aclnnInplaceCumprodGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCumprod”或者“aclnnInplaceCumprod”接口执行计算。

- `aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor* input, const aclScalar* dim, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnCumprod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

- `aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor* input, const aclScalar* dim, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnInplaceCumprod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnCumprodGetWorkspaceSize

* **参数说明：**
  * input（aclTensor*, 计算输入）：当前输入值，表示需要计算累积乘积的数据，Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，支持空Tensor。
数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64。 [数据格式](common/数据格式.md)支持ND。
  * dim（aclScalar*, 计算输入）：当前输入值，指定计算累积乘积的维度，对于一个二维张量，dim=0表示沿着行方向计算，dim=1表示沿列方向计算，Device侧的aclScalar, 取值范围 [-rank(input), rank(input))。数据类型支持INT32。
  * dtype（aclDataType, 计算输入）：指定计算过程input的数据类型。若为ACL_DT_UNDEFINED，使用传入input的原始类型计算；若指定具体类型（需在input支持数据类型范围内），计算前将input转换为此类型。
  * out（aclTensor*, 计算输出）：累积乘积的结果，Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND。dtype=ACL_DT_UNDEFINED时，数据类型必须与input相同；dtype指定时，数据类型必须与dtype相同。out的shape必须与input一致。
  * workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

* **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1.传入的input、dim是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1.传入的input、dim的数据类型和格式不在支持的范围内。
                                       2.传入的dim与input的shape约束不满足要求。
                                       3.out与input的shape不一致。
  ```

## aclnnCumprod

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnCumprodGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  

## aclnnInplaceCumprodGetWorkspaceSize

* **参数说明：**
  * input（aclTensor*, 计算输入|计算输出）：表示需要计算累积乘积的数据和结果，Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor。数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64。 [数据格式](common/数据格式.md)支持ND。
  * dim（aclScalar*, 计算输入）：指定计算累积乘积的维度，对于一个二维张量，dim=0表示沿着行方向计算，dim=1表示沿列方向计算，Device侧的aclScalar，取值范围 [-rank(x), rank(x)]。数据类型支持INT32。
  * workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*，出参）：返回op执行器，包含了算子计算流程。

* **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1.传入的input、dim是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1.传入的input、dim的数据类型和格式不在支持的范围内。
                                       2.传入的dim与input的shape约束不满足要求。
  ```

## aclnnInplaceCumprod

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceCumprodGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnCumprod&aclnnInplaceCumprod默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

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
    // 固定写法，acl初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
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
    // 1.(固定写法)device/stream初始化, 参考acl对外接口列表, 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出，需要根据API的接口自定义构造
    void *xDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    std::vector<int64_t> xShape = {3};
    std::vector<int64_t> xHostData = {1,2,3};
    // 创建原始输入x
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT64, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建axis aclScalar
    int32_t axis_value = 0;
    aclScalar *axis = nullptr;
    ret = CreateAclScalar(aclDataType::ACL_INT32, axis_value, &axis);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建result aclTensor
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
    // 3.调用CANN算子库API
    // 调用aclnnCumprod第一段接口
    ret = aclnnCumprodGetWorkspaceSize(input, axis, dtype, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprod allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnCumprod第二段接口
    ret = aclnnCumprod(workspaceAddr, workspaceSize, executor, stream);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 3.调用CANN算子库API
    // 调用aclnnInplaceCumprod第一段接口
    ret = aclnnInplaceCumprodGetWorkspaceSize(input, axis, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprod allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnInplaceCumprod第二段接口
    ret = aclnnInplaceCumprod(workspaceAddr, workspaceSize, executor, stream);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(resultShape, &xDeviceAddr);

    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyScalar(axis);

    // // 7.释放device资源，需要根据具体API的接口定义修改
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