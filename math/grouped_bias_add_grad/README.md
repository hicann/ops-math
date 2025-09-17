# GroupedBiasAddGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|昇腾910_95 AI处理器|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|
|Atlas 200/300/500 推理产品|×|

## 功能说明

- 算子功能：分组偏置加法（GroupedBiasAdd）的反向传播。

- 计算公式：

  (1) 有可选输入groupIdxOptional时：

  $$
  out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
  $$

  &emsp;&emsp;其中，gradY共2维，H表示gradY最后一维的大小，G表示groupIdxOptional第0维的大小，即groupIdxOptional有G个数，groupIdxOptional(j)表示第j个数的大小，计算后out为2维，shape为(G, H)。

  (2) 无可选输入groupIdxOptional时：

  $$
  out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
  $$

  &emsp;&emsp;其中，gradY共3维，G, C, H依次表示gradY第0-2维的大小，计算后out为2维，shape为(G, H)。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>grad_y</td>
      <td>输入</td>
      <td>公式中的输入gradY。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_idx</td>
      <td>可选输入</td>
      <td>公式中输入的groupIdxOptional。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_bias</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_idx_type</td>
      <td>可选属性</td>
      <td><ul><li>表示group_idx的重要性。</li><li>默认值为0。</li></td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- group_idx最多支持2048个组。
- 当存在输入group_idx时，需要确保张量的值不超过INT32的最大值并且是非负的。
- 当存在输入group_idx并且group_idx_type为0时，需要确保张量数据按升序排列，最后一个数值等于grad_y的第0维度的大小。
- 当存在输入group_idx并且group_idx_type为1时，必须确保张量值的总和必须等于0维中grad_y的大小。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_bias_add_grad](./examples/test_aclnn_grouped_bias_add_grad.cpp) | 通过[aclnnGroupedBiasAddGrad](./docs/aclnnGroupedBiasAddGrad.md)接口方式调用GroupedBiasAddGrad算子。 |
| 图模式调用 | [test_geir_grouped_bias_add_grad](./examples/test_geir_grouped_bias_add_grad.cpp)   | 通过[算子IR](./op_graph/grouped_bias_add_grad_proto.h)构图方式调用GroupedBiasAddGrad算子。 |
