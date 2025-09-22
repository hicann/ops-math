# IsInf

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：判断张量中哪些元素是无限大值，即为inf、-inf。
- 计算公式：

  $$
  out_i=(input_i == ±inf)
  $$

- 示例：
  - 若x=[9, 6, 3]，isinf(x)的结果是[False, False, False]。
  - 若x=[[-3.14, inf], [2.7183, nan]]，isinf(x)的结果是[[False, True], [False, False]]。
  注意整型、布尔都是有界的。

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
      <td>x</td>
      <td>输入</td>
      <td>待进行判断的入参，公式中的input_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行判断的出参，公式中的out_i。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_is_inf](./examples/test_aclnn_is_inf.cpp) | 通过[aclnnIsInf](./docs/aclnnIsInf.md)接口方式调用IsInf算子。 |
| 图模式调用 | [test_geir_is_inf](./examples/test_geir_is_inf.cpp)   | 通过[算子IR](./op_graph/is_inf_proto.h)构图方式调用IsInf算子。 |
