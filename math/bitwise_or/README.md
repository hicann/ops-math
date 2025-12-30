# BitwiseOr

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：计算输入张量self中每个元素和输入标量other的按位或。输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑或。
- 计算公式：

  $$
  \text{out}_i =
  \text{self}_i \, | \, \text{other}
  $$

## 参数说明：

  - self(aclTensor*, 计算输入)：公式中的输入self，Device侧的aclTensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](common/TensorScalar互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8，且数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
  - other(aclScalar*, 计算输入)：公式中的输入other，Host侧的aclScalar。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8，且数据类型与self的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](common/TensorScalar互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8，且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），推导后的数据类型需在支持的数据类型范围内。
  - out(aclTensor \*, 计算输出)：公式中的输出out，Device侧的aclTensor。数据类型需要是self与other推导之后可转换的数据类型，shape需要与self一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8、FLOAT、FLOAT16、DOUBLE。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8、FLOAT、FLOAT16、DOUBLE、BFLOAT16。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、INT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、UINT8、FLOAT、FLOAT16、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128。

## 约束说明

- 无

## 调用说明
    
| 调用方式   | 样例代码           | 说明                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_bitwise_or_scalar.cpp](examples/test_aclnn_bitwise_or_scalar.cpp) | 通过[aclnnBitwiseOrScalar](docs/aclnnBitwiseOrScalar.md)接口方式调用BitwiseOrScalar算子。 |
| aclnn接口  | [test_aclnn_bitwise_or_tensor.cpp](examples/test_aclnn_bitwise_or_tensor.cpp) | 通过[aclnnBitwiseOrTensor](docs/aclnnBitwiseOrTensor.md)接口方式调用BitwiseOrTensor算子。 |