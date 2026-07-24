# FusedMulAddNL2loss

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Kirin X90 处理器系列产品</term>                      | × |
| <term>Kirin 9030 处理器系列产品</term>                       | × |

## 功能说明

- 算子功能：融合标量乘加与 L2 Loss 计算。对输入张量逐元素执行乘加运算得到 y1，同时对 x1 执行全量平方归约得到 y2（0 维标量）。该算子为 GE 图融合内部算子（由 `Mul + AddN + L2Loss` 子图融合生成），Ascend 950PR/950DT 上的语义与 910b 内置实现一致。

- 计算公式：

$$
y_1 = x_1 \times x_3[0] + x_2
$$

$$
y_2 = \sum \frac{x_1^2}{2}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入张量x1，ND张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量x2，ND张量，shape和数据类型与x1相同。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>公式中的标量输入张量x3，单元素张量，计算时仅取x3[0]，数据类型与x1相同。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>公式中的输出张量y1，shape和数据类型与x1相同。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>公式中的输出张量y2，为x1的L2 Loss，0维标量，数据类型与x1相同。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x1、x2 的 shape 必须一致（tiling 强校验）。
- x1/x2/x3/y1/y2 数据类型必须一致，仅支持 FLOAT、FLOAT16。
- x3 为单元素标量张量，实现上仅取 x3[0]，不校验其元素个数。
- y2 为 0 维标量输出（空 dims）。
- FLOAT16 数据类型下，y2 结果超过 65504 时溢出（与 910b 内置实现一致，为数据类型固有限制）。
- 该算子不提供 aclnn 接口，仅支持图模式调用。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | 不支持 | 该算子为图融合内部算子，不提供aclnn接口。 |
| 图模式调用 | [test_geir_fused_mul_add_nl2loss](./examples/arch35/test_geir_fused_mul_add_nl2loss.cpp)   | 通过[算子IR](./op_graph/fused_mul_add_nl2loss_proto.h)构图方式调用FusedMulAddNL2loss算子。 |
