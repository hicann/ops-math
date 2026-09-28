# PreventGradient

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：前向计算时原样返回输入张量，不做任何数值变换；反向传播（求梯度）时该算子不提供梯度，一旦梯度计算需要流经此处，框架将立即报错并终止，用于显式阻断梯度回传。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| y | 输出 | 与输入 `x` 形状、数据类型和内容相同的张量。 | 与 `x` 相同 | ND |
| message | 属性 | 反向传播报错时输出的提示信息，默认值为空字符串。 | STRING | - |

## 约束说明

- `x` 和 `y` 的形状及数据类型保持一致。
- 该算子前向不执行数值变换；`message` 仅在反向传播报错时作为提示信息输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_prevent_gradient](examples/test_geir_prevent_gradient.cpp) | 通过[算子IR](op_graph/prevent_gradient_proto.h)构图方式调用PreventGradient算子。 |
