# PlaceHolder

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

为一个将始终被喂入的张量插入占位符。PlaceHolder算子接收输入张量x，输出y的形状和数据类型与输入x一致，由InferShape将输入shape复制到输出，InferDataType将输入dtype复制到输出。

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
      <td>peerIndex</td>
      <td>属性</td>
      <td>所连接的对应"end"节点的索引。</td>
      <td>Int，默认值为0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>parentId</td>
      <td>属性</td>
      <td>用于检查节点是否来自已保存的父节点的字符串标识。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>parentOpType</td>
      <td>属性</td>
      <td>原始节点的算子类型。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>anchorIndex</td>
      <td>属性</td>
      <td>用于检查节点是否来自已保存的anchor的索引。</td>
      <td>Int，默认值为0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>创建的占位张量，形状和数据类型与输入x一致。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- PlaceHolder为图引擎内部节点，由GE引擎切分（EnginePartitioner）在子图边界自动插入，需与配对的End节点及peerIndex属性配合使用，用户构图不可单独编译执行。

## 调用说明

PlaceHolder算子为图引擎基础算子，通常作为图中占位节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_placeholder](./examples/test_geir_placeholder.cpp) | 通过[算子IR](./op_graph/placeholder_proto.h)构图方式调用PlaceHolder算子 |
