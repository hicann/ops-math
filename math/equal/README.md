# Equal

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。
- 计算表达式：

  $$
  out = (self == other)  ?  True : False
  $$

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>计算公式中的输入 self。</td>
      <td>FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8、DOUBLE、INT16、INT64、COMPLEX64、COMPLEX128、QUINT8、QINT8、QINT32、STRING、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>计算公式中的输入 other。</td>
      <td>FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8、DOUBLE、INT16、INT64、COMPLEX64、COMPLEX128、QUINT8、QINT8、QINT32、STRING、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算公式中的输出 out。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、DOUBLE、INT64、INT16、UINT16、UINT32、UINT64。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、DOUBLE、INT64、INT16、UINT16、UINT32、UINT64、BFLOAT16。

## 约束说明

如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），场景为最后2轴合轴小于16，前面的轴合轴超大。

## 调用说明

| 调用方式 | 调用样例                                                   | 说明                                               |
|--------------|--------------------------------------------------------|--------------------------------------------------|
| aclnn调用 | [test_aclnn_equal.cpp](examples/test_aclnn_equal.cpp) | 通过[aclnnEqual](docs/aclnnEqual.md)接口方式调用Equal算子。 |
| 图模式调用 | [test_geir_equal.cpp](examples/test_geir_equal.cpp) | 通过[算子IR](op_graph/equal_proto.h)构图方式调用Equal算子。   |
