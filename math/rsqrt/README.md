# Rsqrt

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

- 算子功能：求输入张量x每个元素的平方根的倒数，并将结果保存到输出张量y中。

- 计算公式：

$$
y = \frac{1}{\sqrt{x}}
$$

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
      <td>x</td>
      <td>输入</td>
      <td>输入张量，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，公式中的y，shape与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 200I/500 A2推理产品、Atlas推理系列产品、Atlas训练系列产品：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_rsqrt](./examples/test_aclnn_rsqrt.cpp) | 通过[aclnnRsqrt或aclnnInplaceRsqrt](./docs/aclnnRsqrt&aclnnInplaceRsqrt.md)接口方式调用Rsqrt算子。 |
| 图模式调用 | [test_geir_rsqrt](./examples/test_geir_rsqrt.cpp)   | 通过[算子IR](./op_graph/rsqrt_proto.h)构图方式调用Rsqrt算子。 |
