# aclnnSin&aclnnInplaceSin

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 算子功能：对输入Tensor完成sin运算。
- 计算公式：

  $$
  out = {\ {{sin}{(self)}}}
  $$

## 函数原型
- aclnnSin和aclnnInplaceSin实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnSin：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceSin：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnSinGetWorkspaceSize” 或者 “aclnnInplaceSinGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小，再调用 “aclnnSin” 或者 “aclnnInplaceSin” 接口执行计算。
  - `aclnnStatus aclnnSinGetWorkspaceSize(const aclTensor *self,  aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnSin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceSinGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnInplaceSin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSinGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*，计算输入): Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND，shape维度不大于8, 且shape需要与out一致，和out的数据类型满足数据类型推导规则。当输入的数据类型为INT8、INT16、INT32、INT64、UINT8、BOOL时，会转换为float数据类型做计算，计算结果转换成out的数据类型。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - out(aclTensor *，计算输出): Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND，且shape需要与self一致，和self的数据类型满足数据类型推导规则。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现如下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self或out是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. self和out的数据类型和数据格式不在支持的范围之内。
                         2. self和out的数据类型不满足数据类型推导规则。
                         3. self和out的维度大于8。
                         4. self和out的shape不一致。
```

## aclnnSin

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnSinGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceSinGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor *，计算输出|计算输入): 支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND，shape维度不大于8。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现如下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的selfRef是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. selfRef数据类型和数据格式不在支持的范围之内。
                                 2. selfRef的维度大于8
```

## aclnnInplaceSin

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceSinGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSin&aclnnInplaceSin默认确定性实现。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、BFLOAT16数据类型的输入数据范围为[-10^7, 10^7]时满足精度要求，超过数值范围无法保证，请使用CPU进行计算。
- <term>Atlas 推理系列产品</term>：FLOAT、FLOAT16数据类型的输入数据范围为[-65504, 65504]时满足精度要求，超过数值范围无法保证，请使用CPU进行计算。