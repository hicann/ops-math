# LessEqual

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：判断输入self中的元素值是否小于等于other的值，并将self的每个元素的值与other值的比较结果写入out中。
- 计算公式：

  $$
  out_i = (self_i <= other) ? [True] : [False]
  $$

- **参数说明：**

    * selfRef(aclTensor*，计算输入|计算输出)：输入输出tensor，即公式中的self与out。shape维度不高于8维，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
        - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、UINT16、FLOAT、DOUBLE、BOOL，且与other满足[互推导关系](common/互推导关系.md)。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、UINT16、BFLOAT16、FLOAT、DOUBLE、BOOL，且与other满足[互推导关系](common/互推导关系.md)。
        - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、UINT16、BFLOAT16、FLOAT、DOUBLE、BOOL、UINT64，且与other满足[TensorScalar互推导关系](common/TensorScalar互推导关系.md)。
    * other(aclScalar*，计算输入)：公式中的other，Host侧的aclScalar。
        - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、UINT16、FLOAT、DOUBLE、BOOL，且与selfRef满足[互推导关系](common/互推导关系.md)。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、UINT16、BFLOAT16、FLOAT、DOUBLE、BOOL，且与selfRef满足[互推导关系](common/互推导关系.md)。
        - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、UINT16、BFLOAT16、FLOAT、DOUBLE、BOOL、UINT64，且与selfRef满足[TensorScalar互推导关系](common/TensorScalar互推导关系.md)。
    * workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    * executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_le_scalar](./examples/test_aclnn_le_scalar.cpp) | 通过[aclnnLeScalar&aclnnInplaceLeScalar](./docs/aclnnLeScalar&aclnnInplaceLeScalar.md)接口方式调用Less算子。    |