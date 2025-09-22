# DiagFlat

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：生成对角线张量。如果输入self为一维张量，则返回二维张量，self里元素为对角线值；如果输入self是二维及以上张量，则先进行扁平化（化简为一维张量），再转化为第一种场景处理。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1043px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 320px">
<col style="width: 96px">
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
    <td>输入张量</td>
    <td>表示填充到对角线的向量，最大维度支持8维，支持非连续的Tensor。</td>
    <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>diagonal</td>
    <td>输入属性</td>
    <td>指定对角线位置，diagonal = 0表示主对角线，diagonal &gt; 0表示主对角线上方的对角线，diagonal &lt; 0表示主对角线下方的对角线。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>输出的对角线张量。</td>
    <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

无。

## 调用说明

| 调用方式  | 样例代码                                                    | 说明                                                         |
| --------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_diag_flat](./examples/test_aclnn_diag_flat.cpp) | 通过[aclnnDiagFlat](docs/aclnnDiagFlat.md)接口方式调用DiagFlat算子。 |