# ConstPlaceHolder

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

创建常量占位张量。该算子用于处理地址不变但值不确定的输入，将输入转换为此类型有助于提升图的执行性能（避免输入地址被反复刷新）。根据origin_shape和dtype属性推导输出张量y的形状和数据类型。输出形状与origin_shape属性一致，输出数据类型与dtype属性一致。

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
      <td>origin_shape</td>
      <td>属性</td>
      <td>必选。张量的原始形状。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>origin_format</td>
      <td>属性</td>
      <td>必选。张量的原始格式。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>storage_shape</td>
      <td>属性</td>
      <td>必选。张量的存储形状。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>storage_format</td>
      <td>属性</td>
      <td>必选。张量的存储格式。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expand_dim_rules</td>
      <td>属性</td>
      <td>必选。从origin_shape、origin_format和storage_format转换到storage_shape的维度扩展规则。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>必选。张量的数据类型。</td>
      <td>Type</td>
      <td>-</td>
    </tr>
    <tr>
      <td>addr</td>
      <td>属性</td>
      <td>必选。张量的地址。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>size</td>
      <td>属性</td>
      <td>必选。地址大小。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>placement</td>
      <td>属性</td>
      <td>张量的放置位置，0表示Host，1表示Device。</td>
      <td>Int，默认值为1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>常量占位张量，形状与origin_shape属性一致，数据类型与dtype属性一致。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

ConstPlaceHolder算子为图引擎基础算子，通常作为图中常量占位节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_const_place_holder](./examples/test_geir_const_place_holder.cpp) | 通过[算子IR](./op_graph/const_place_holder_proto.h)构图方式调用ConstPlaceHolder算子 |
