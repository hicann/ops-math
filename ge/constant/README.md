# Constant

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

创建一个常量张量用于训练。Constant算子根据value属性中保存的Tensor值和类型，推导输出张量y的形状和数据类型。输出形状和数据类型与value属性中的Tensor一致。

## 参数说明

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>value</td>
      <td>属性</td>
      <td>必选。输出张量的值和类型，类型无限制。</td>
      <td>Tensor，默认值为Tensor()</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>常量张量，形状和数据类型与value属性一致。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

Constant算子为图引擎基础算子，通常作为图中常量节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_constant](./examples/test_geir_constant.cpp) | 通过[算子IR](./op_graph/constant_proto.h)构图方式调用Constant算子 |
