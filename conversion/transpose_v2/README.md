# TransposeV2
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------- |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     | √        |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √        |
| <term>Kirin X90 处理器系列产品</term>                          | √        |
## 功能说明

- 算子功能：实现张量的维度置换（Permutation）操作，按照指定的顺序重新排列输入张量的维度。如输入self是shape为[2, 3, 5]的tensor，dims为(2, 0, 1)，则输出是shape为[5, 2, 3]的tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型 | 数据格式 |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| self   | 输入张量       | 需要进行维度置换的输入张量。                                 | 见下方   | ND       |
| dims   | 输入数组       | 整型数组，代表原来tensor的维度，指定新的轴顺序。取值需在[-self的维度数量，self的维度数量-1]范围内。 | INT64    | -        |
| out    | 输出           | 维度最大不超过8维，shape由dims和原self的shape共同决定，dtype需要与self一致。 | 同self   | ND       |

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、UINT64、INT64、UINT32、INT32、UINT16、INT16、UINT8、INT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16。
- <term>Kirin X90 处理器系列产品</term>：数据类型支持FLOAT、FLOAT16。
## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_transpose_v2](examples/test_aclnn_transpose_v2.cpp) | 通过[aclnnPermute](../transpose/docs/aclnnPermute.md)接口方式调用transposeV2算子。(transpose_v2与transpose共用aclnn，需联合出单算子包运行[test_aclnn_transpose_v2](examples/test_aclnn_transpose_v2.cpp)：bash build.sh --pkg --soc=${soc_version} --vendor_name=transpose_v2 --ops=transpose_v2,transpose) |