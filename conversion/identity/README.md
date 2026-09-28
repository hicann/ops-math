# Identity

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回一个与输入张量 `x` 具有相同形状和内容的张量 `y`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| y | 输出 | 与输入张量 `x` 具有相同形状和内容的张量。 | 与 `x` 相同 | ND |

## 约束说明

- `x` 和 `y` 的形状、数据类型及元素内容保持一致。
- 支持的数据类型以 [`identity_proto.h`](./op_graph/identity_proto.h) 中的 IR 定义为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_identity](examples/test_geir_identity.cpp)、[test_geir_identity_v2](examples/test_geir_identity_v2.cpp) | 通过[算子IR](op_graph/identity_proto.h)构图方式调用Identity算子。 |
