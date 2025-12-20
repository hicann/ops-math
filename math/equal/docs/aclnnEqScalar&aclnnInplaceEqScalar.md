# aclnnEqScalar&aclnnInplaceEqScalar

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/math/equal)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 接口功能：计算self中的元素的值与other的值是否相等，将self每个元素与other的值的比较结果写入out中。
- 计算公式：

  $$
  out_i = (self_i == \mathit{other} )  ?  [True] : [False]
  $$

## 函数原型

- aclnnEqScalar和aclnnInplaceEqScalar实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnEqScalar：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceEqScalar：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnEqScalarGetWorkspaceSize”或者“aclnnInplaceEqScalarGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEqScalar”或者“aclnnInplaceEqScalar”接口执行计算。

```Cpp
aclnnStatus aclnnEqScalarGetWorkspaceSize(
  const aclTensor *self, 
  const aclScalar *other, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnEqScalar(
  void*             workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor*    executor, 
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceEqScalarGetWorkspaceSize(
  const aclTensor *selfRef, 
  const aclScalar *other, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceEqScalar(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnEqScalarGetWorkspaceSize

- **参数说明：**

  * self（aclTensor*，计算输入）：公式中的`self`，Device侧的aclTensor。shape维度不高于8维，支持[非连续的Tensor](./../../../docs/zh/context/非连续的Tensor.md)，[数据格式](./../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64，且与other满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
  * other（aclScalar*, 计算输入）：公式中的`other`，Host侧的aclScalar。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64，且与self满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与self满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与self满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
  * out（aclTensor*，计算输出）：公式中的out，Device侧的aclTensor。数据类型BOOL可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)），shape与self的shape一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64、UINT32、UINT16。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
    * <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128。
  - workspaceSize(uint64_t \*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
 
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、other、out是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>self，other或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self和other数据类型不满足数据类型推导规则。</td>
    </tr>
    <tr>
      <td>self和out的shape不同。</td>
    </tr>
    <tr>
      <td>self和out的维度大于8。</td>
    </tr>
  </tbody>
  </table>

## aclnnEqScalar

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnEqScalarGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceEqScalarGetWorkspaceSize

- **参数说明：**

  * selfRef（aclTensor* 计算输入/输出）：公式中的`selfRef`，Device侧的aclTensor。shape维度不高于8维，支持[非连续的Tensor](./../../../docs/zh/context/非连续的Tensor.md)，[数据格式](./../../../docs/zh/context/数据格式.md)支持ND。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64，且与other满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与other满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
  * other（aclScalar*, 计算输入）：公式中的`other`，Host侧的aclScalar。
    * <term>昇腾910_95 AI处理器</term>：数据类型支持DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64，且与selfRef满足[TensorScalar互推导关系](../../../docs/zh/context/TensorScalar互推导关系.md)。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、INT32、INT8、UINT8、BOOL、 BFLOAT16、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与selfRef满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
    * <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、INT64、UINT64、INT32、INT8、UINT8、BOOL、UINT32、DOUBLE、INT16、COMPLEX64、COMPLEX128，且与selfRef满足[互推导关系](../../../docs/zh/context/互推导关系.md)。
  - workspaceSize(uint64_t \*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的selfRef和other是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>selfRef和other的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>selfRef和other的数据类型不满足数据类型推导规则。</td>
    </tr>
    <tr>
      <td>selfRef的维度大于8。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceEqScalar

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceEqScalarGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnEqScalar&aclnnInplaceEqScalar默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnEqScalar示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_scalar.h"

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
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, void** outDeviceAddr,
    aclTensor** self, aclScalar** other, aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 1.2, 0.3, 4.1, 5, 1.6, 7};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  double otherValue = 1.2;

  // 创建 self tensor
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 other scalar
  *other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*other != nullptr, return ret);

  // 创建 out tensor
  ret = CreateAclTensor(outHostData, outShape, &(*outDeviceAddr), aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclScalar* other, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // 获取 workspace 大小
  auto ret = aclnnEqScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 申请 workspace（释放放在 main 里）
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // 调用算子
  ret = aclnnEqScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqScalar failed. ERROR: %d\n", ret); return ret);

  // 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝输出
  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(char),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(selfShape, outShape, &selfDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 释放 Tensor / Scalar
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 释放 device 内存
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
**aclnnInplaceEqScalar示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_eq_scalar.h"

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
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(std::vector<int64_t>& selfShape, void** selfDeviceAddr, aclTensor** self, aclScalar** other)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  double otherValue = 2.0;

  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建other aclScalar
  *other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(*other != nullptr, return ACL_ERROR_INVALID_PARAM);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclScalar* other, void* selfDeviceAddr, std::vector<int64_t>& selfShape, aclrtStream stream,
    void** workspaceAddrOut)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  auto ret = aclnnInplaceEqScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // workspace 分配
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  *workspaceAddrOut = workspaceAddr;

  // 执行
  ret = aclnnInplaceEqScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceEqScalar failed. ERROR: %d\n", ret); return ret);

  // 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝输出
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclScalar* other = nullptr;

  ret = CreateInputs(selfShape, &selfDeviceAddr, &self, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  ret = ExecOpApi(self, other, selfDeviceAddr, selfShape, stream, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 销毁
  aclDestroyTensor(self);
  aclDestroyScalar(other);

  aclrtFree(selfDeviceAddr);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  // 释放
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```


