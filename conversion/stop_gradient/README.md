# StopGradient

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

- 算子功能：将输入张量原样传递到输出，同时阻止反向传播计算经过该节点。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| y | 输出 | 与输入 `x` 形状、数据类型和内容相同的张量。 | 与 `x` 相同 | ND |

## 约束说明

- `x` 和 `y` 的形状及数据类型保持一致。
- 该算子不执行数值变换，仅改变梯度传播行为。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_stop_gradient](examples/test_geir_stop_gradient.cpp) | 通过[算子IR](op_graph/stop_gradient_proto.h)构图方式调用StopGradient算子。 |
