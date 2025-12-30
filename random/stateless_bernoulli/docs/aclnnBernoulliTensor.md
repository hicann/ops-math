# aclnnBernoulliTensor&aclnnInplaceBernoulliTensor

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：
    从伯努利分布中提取二进制随机数（0 或 1），prob为生成二进制随机数的概率，输入的张量用于指定shape。

- 计算公式：

  $$
  out∼Bernoulli(prob)
  $$

  其中，当使用aclnnBernoulliTensor时，公式中的prob对应第一段接口中的prob，公式中的out对应第一段接口中的out；当使用aclnnInplaceBernoulliTensor时，公式中的prob对应第一段接口中的prob，公式中的out对应第一段接口中的selfRef。

## 函数原型

  - aclnnBernoulliTensor和aclnnInplaceBernoulliTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
    - aclnnBernoulliTensor：需新建一个输出张量对象存储计算结果。
    - aclnnInplaceBernoulliTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
  - 每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnBernoulliTensorGetWorkspaceSize”或者“aclnnInplaceBernoulliTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBernoulliTensor”或者“aclnnInplaceBernoulliTensor”接口执行计算。

    - `aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
    - `aclnnStatus aclnnBernoulliTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`
    - `aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)`
    - `aclnnStatus aclnnInplaceBernoulliTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBernoulliTensorGetWorkspaceSize

  - **参数说明：**
    - self（aclTensor*，计算输入）：用于指定输出out的shape，Device侧的aclTensor，shape支持0-8维，shape需要与out的shape一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
    - prob（aclTensor*，计算输入）：公式中的prob，Device侧的aclTensor，满足0≤prob≤1，shape支持0-8维，支持[非连续的Tensor](common/非连续的Tensor.md)，且[数据格式](common/数据格式.md)需要与self一致。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16。
    - seed（int64_t，计算输入）：Host侧的整型，设置随机数生成器的种子。
    - offset（int64_t，计算输入）：Host侧的整型，设置随机数偏移量。
    - out（aclTensor*,计算输出）：公式中的out，Device侧的aclTensor，shape支持0-8维，shape需要与self的shape一致，数据类型与self一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
      - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
    - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现如下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、prob或out是空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、prob或out的数据类型和数据格式不在支持的范围之内。
                                          2. self和out的数据类型不一致。
                                          3. self、prob或out的维度大于8。
                                          4. self和out的shape不一致。
    ```

## aclnnBernoulliTensor

  - **参数说明：**
    - workspace（void*，入参）：在Device侧申请的workspace内存地址。
    - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnBernoulliTensorGetWorkspaceSize获取。
    - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream，入参）：指定执行任务的Stream。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

