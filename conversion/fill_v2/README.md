# FillV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：根据dims指定的shape，创建一个用value填充的输出张量y。

- 计算公式：

$$y_i=value$$

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>dims</td>
      <td>输入</td>
      <td>1D张量，表示输出y的shape，维度数不超过8。</td>
      <td>INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入属性</td>
      <td>待填充值，公式中的value。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行计算的出参，公式中的y_i。</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入dims必须是1D张量，维度数不超过8。
- 输入dims的维度值不能超过dims自身数据类型的表示范围：超出时数值按该类型回绕，回绕为负数会被tiling校验拦截报错，不会以错误shape执行；回绕后仍为非负时无法检测，将按回绕后的值执行。
- 填充的value值超过输出数据类型能够表示的最大数值范围时，会溢出报错。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_fill_v2](./examples/test_geir_fill_v2.cpp)   | 通过[算子IR](./op_graph/fill_v2_proto.h)构图方式调用FillV2算子。 |
