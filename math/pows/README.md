# Pows

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Kirin X90 处理器系列产品</term>                         |    √     |

## 功能说明

- 算子功能：对input中的每个元素应用指数为exponent的幂运算。
- 计算公式：
$$
out_i = self_i^{exponent_i}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
  <col style="width: 370px">
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
      <td>公式中的输入`self`，Device侧的aclTensor。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exponent</td>
      <td>输入</td>
      <td>公式中的输入`exponent`，Device侧的aclScalar。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出`out`，Device侧的aclTensor。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、BOOL、INT8、UINT8、INT16、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
- Kirin X90 处理器系列产品: 不支持BFLOAT16。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_pows](./examples/test_geir_pows.cpp)   | 通过[算子IR](./op_graph/pows_proto.h)构图方式调用Pows算子。 |
