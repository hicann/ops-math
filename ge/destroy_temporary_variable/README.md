# DestroyTemporaryVariable

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

销毁临时变量并返回其最终值。临时变量的所有其他使用必须在本算子之前执行完毕。本算子与TemporaryVariable算子配对使用：通过var_name属性与TemporaryVariable匹配，读取该临时变量的最终值并触发销毁。输出张量y的形状和数据类型与输入张量x一致。

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
      <td>x</td>
      <td>输入</td>
      <td>临时变量张量的引用。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_name</td>
      <td>属性</td>
      <td>可选。临时变量的名称，必须与TemporaryVariable算子的var_name属性一致。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>临时变量张量的最终值，形状和数据类型与输入x一致。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 本算子为实验性功能（EXPERIMENTAL），请勿在生产环境使用。
- 本算子为 TensorFlow 兼容算子，图引擎将其注册为图优化阶段应被消除的节点（GeDeletedOp）：真实 TensorFlow 图中由框架解析器配对 TemporaryVariable 在解析阶段消除（var_name 匹配后删除本节点、消费者直连），因此本算子无法通过 RunGraph/BuildModel 独立编译执行。

## 调用说明

DestroyTemporaryVariable算子为图引擎基础算子，与[TemporaryVariable](../temporary_variable/README.md)配对使用：TemporaryVariable创建临时变量并供下游写入，本算子在所有使用完成后读取终值并销毁临时变量。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_destroy_temporary_variable](./examples/test_geir_destroy_temporary_variable.cpp) | 通过[算子IR](./op_graph/destroy_temporary_variable_proto.h)构图方式调用DestroyTemporaryVariable算子 |
