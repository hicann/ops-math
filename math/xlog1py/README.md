# Xlog1py

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 算子功能：计算 x * log(1 + y)，当 x == 0 时结果为 0。
- 计算公式：

$$
z_i =
\begin{cases}
0,                                & x_i = 0 \\
x_i \cdot \log(1 + y_i),          & x_i \neq 0
\end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的 x，乘数因子。shape 需与 y 满足 broadcast 关系。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>公式中的 y，log1p 的自变量。shape 需与 x 满足 broadcast 关系。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>公式中的 z，计算结果。shape 为 x 与 y broadcast 后的最大值 shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 shape 维度最大为 8。
- 输入 x、y、输出 z 的数据类型必须一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_xlog1py](./examples/arch35/test_aclnn_xlog1py.cpp) | 通过[aclnnXlog1py](./docs/aclnnXlog1py.md)接口方式调用Xlog1py算子。 |
| 图模式调用 | [test_geir_xlog1py](./examples/test_geir_xlog1py.cpp)   | 通过[算子IR](./op_graph/xlog1py_proto.h)构图方式调用Xlog1py算子。 |
