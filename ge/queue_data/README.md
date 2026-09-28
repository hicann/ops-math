# QueueData

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

为其他算子提供队列数据。QueueData算子作为图中的队列数据输入节点，根据output_types和output_shapes属性计算输出张量的长度（每项含64字节信息头、维度描述及数据字节），输出为DT_UINT8类型的一维张量。

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
      <td>index</td>
      <td>属性</td>
      <td>输入张量的索引。数据类型为int32或int64。假设网络中有三个data节点，一个应设为0，另一个设为1，第三个应设为2。</td>
      <td>Int，默认值为0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>queue_name</td>
      <td>属性</td>
      <td>队列名称。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_types</td>
      <td>属性</td>
      <td>输出数据的数据类型列表。</td>
      <td>ListType，默认值为{}</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_shapes</td>
      <td>属性</td>
      <td>输出数据的形状列表。</td>
      <td>ListListInt，默认值为{{}, {}}</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>DT_UINT8类型的一维张量，长度为根据output_types和output_shapes计算的总字节数。</td>
      <td>DT_UINT8</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- output_types和output_shapes的长度必须一致。

## 调用说明

QueueData算子为图引擎基础算子，通常作为图中队列数据输入节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_queue_data](./examples/test_geir_queue_data.cpp) | 通过[算子IR](./op_graph/queue_data_proto.h)构图方式调用QueueData算子 |
