# Lerp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。

- 计算公式：
  $$
  \text { out }_i=\text { start }_i+\text { weight }_i \times\left(\text { end }_i-\text { start }_i\right)
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
      <td>self</td>
      <td>输入</td>
      <td>待进行lerp计算的入参，公式中的start_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>待进行lerp计算的入参，公式中的end_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>待进行lerp计算的入参，公式中的weight_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行lerp计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_lerp](./examples/test_aclnn_lerp.cpp) | 通过[aclnnLerp](./docs/math/lerp/docs/aclnnLerp&aclnnInplaceLerp.md)接口方式调用Lerp算子。 |
| 图模式调用 | [test_geir_lerp](./examples/test_geir_lerp.cpp)   | 通过[算子IR](./op_graph/lerp_proto.h)构图方式调用Lerp算子。 |
