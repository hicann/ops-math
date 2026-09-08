# DenseBincount

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term> | √ |
| <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- **算子功能**：DenseBincount对一维或二维整数输入逐行统计每个bin的出现次数或权重和。

- **计算公式**：

  设bin数量为`M = size[0]`。对于一维输入，输出形状为`[M]`；对于二维输入，输出形状为`[N, M]`，其中`N = input.shape[0]`。对有效的行号`r`和bin下标`m`，非二值输出为：

  $$
  output[r,m] = \sum_{i \in row(r)} \mathbf{1}(input_i=m) \times
  \begin{cases}
  weights_i, & numel(weights)>0 \\
  1, & numel(weights)=0
  \end{cases}
  $$

  `binary_output`为`true`时，输出为：

  $$
  output[r,m] = \mathbf{1}(\exists i \in row(r), input_i=m)
  $$

  输入值大于或等于`M`时忽略。一维输入中的负值忽略；二维输入中的负值按TensorFlow DenseBincount语义折返到前一行，折返后行号或bin下标仍越界时忽略。`binary_output`为`true`时不使用`weights`的数值。

## 参数说明

<table style="table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 14%">
<col style="width: 14%">
<col style="width: 12%">
<col style="width: 34%">
<col style="width: 16%">
<col style="width: 10%">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>是否必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>input</td>
    <td>输入</td>
    <td>是</td>
    <td>一维或二维bin下标张量。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>size</td>
    <td>输入</td>
    <td>是</td>
    <td>包含一个非负常量值的一维张量，<code>size[0]</code>表示bin数量。数据类型必须与<code>input</code>一致。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weights</td>
    <td>输入</td>
    <td>是</td>
    <td>权重张量。该输入不可省略；无权重时传零元素张量，表示按次数统计。非空时元素数量必须与<code>input</code>相同，并按展平下标一一对应。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>binary_output</td>
    <td>属性</td>
    <td>否</td>
    <td>是否仅输出bin命中标记，默认为<code>false</code>。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output</td>
    <td>输出</td>
    <td>是</td>
    <td>统计结果。一维输入对应形状<code>[size[0]]</code>，二维输入对应形状<code>[input.shape[0], size[0]]</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- `input`的rank只能为1或2。
- `size`必须是仅含一个非负常量值的一维ND张量，且数据类型必须与`input`相同。
- `weights`不可省略。无权重时传入零元素张量；非空时元素数量必须与`input`相同。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - `weights`支持任意零元素shape，非空时为FLOAT32类型ND张量。
  - 支持`input`任意维度大小为0或`size[0]`为0。`input`的shape为`[0]`时，输出shape为`[size[0]]`；shape为`[N, 0]`时，输出shape为`[N, size[0]]`；shape为`[0, N]`或`[0, 0]`时，输出shape为`[0, size[0]]`。`size[0]`为0时，输出最后一维大小为0。
  - 输出元素数`rows * size[0]`不能超过INT64索引可表示的FLOAT32元素数量上限，且输出张量所需存储必须能由运行环境分配；一维输入的`rows`取1。
- <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>、<term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>：
  - 沿用既有实现的空Tensor处理边界。
  - `weights`的rank不能超过2。
  - `size[0]`、输入维度大小和输入元素数量必须在INT32范围内。
- 本算子仅支持GE图模式和TensorFlow图解析通路调用，不提供公开的aclnn接口。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| 图模式调用 | [test_geir_dense_bincount.cpp](./examples/test_geir_dense_bincount.cpp) | 通过[算子IR](./op_graph/dense_bincount_proto.h)构图调用DenseBincount算子。 |
