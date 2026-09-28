# GuaranteeConst

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

向运行时保证输入张量是一个常量。本算子为直通（pass-through）算子：输出张量y即输入张量x，不改变任何形状、数据类型和数值，仅在图中传递"输入是常量"的语义信息，供图优化决策使用。输出张量y的形状和数据类型与输入张量x一致。

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
      <td>需要保证为常量的张量。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，与输入x完全一致（形状、数据类型、数值）。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 本算子为实验性功能（EXPERIMENTAL），请勿在生产环境使用。
- 本算子为 TensorFlow 兼容的直通算子，图优化阶段会被消除（输出直连输入），运行时不产生实际计算任务。

## 调用说明

GuaranteeConst算子为图引擎基础算子，用于在图中标记常量语义，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_guarantee_const](./examples/test_geir_guarantee_const.cpp) | 通过[算子IR](./op_graph/guarantee_const_proto.h)构图方式调用GuaranteeConst算子 |
