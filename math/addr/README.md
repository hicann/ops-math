# Addr

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出

- 计算公式：
  $$
  \text{out} = \beta\ \text{self} + \alpha\ (\text{vec1} \otimes\text{vec2})
  $$

- **参数说明：**
  
  - self(aclTensor*, 计算输入): 外积扩展矩阵，Device侧的aclTensor，shape维度不能超过2，并且需要与vec1、vec2满足[broadcast关系](common/broadcast关系.md), 支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - vec1(aclTensor*, 计算输入): 外积入参第一向量，一维向量，Device侧的aclTensor，shape需要与self满足[broadcast关系](common/broadcast关系.md), 支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - vec2(aclTensor*, 计算输入): 外积入参第二向量，一维向量，Device侧的aclTensor，shape需要与self满足[broadcast关系](common/broadcast关系.md), 支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - betaOptional(aclScalar*, 计算输入): 外积扩展矩阵比例因子，即公式中的β，host侧的aclScalar，如果betaOptional为bool类型，则self/vec1/vec2的数据类型只能是bool；如果self/vec1/vec2为整型，则betaOptional、alphaOptional不能为浮点型；[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - alphaOptional(aclScalar*, 计算输入): 外积比例因子，即公式中的α，host侧的aclScalar，如果alphaOptional为bool类型，则self/vec1/vec2的数据类型只能是bool；如果self/vec1/vec2为整型，则betaOptional、alphaOptional不能为浮点型；[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - out(aclTensor\*, 计算输出): 输出结果，Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16。

  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor\*\*, 出参): 返回op执行器，包含了算子计算流程。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_addr](./examples/test_aclnn_addr.cpp) | 通过[aclnnAddr](aclnnAddr&aclnnInplaceAddr.md)接口方式调用Addr算子。 |
