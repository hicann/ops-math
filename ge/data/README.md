# Data

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

为其他算子提供输入数据。Data算子作为图中的数据输入节点，将输入张量x直接传递给输出张量y，同时通过index属性标识该数据节点在网络中的序号。

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
      <td>输入张量。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>属性</td>
      <td>输入张量的索引。数据类型为int32或int64。假设网络中有三个data节点，一个应设为0，另一个设为1，第三个设为2。</td>
      <td>Int，默认值为0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>与输入张量x具有相同形状和数据类型的张量</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

Data算子为图引擎基础算子，通常作为图中数据输入节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_data](./examples/test_geir_data.cpp) | 通过[算子IR](./op_graph/data_proto.h)构图方式调用Data算子 |
