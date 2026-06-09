# AmpUpdateScale

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：实现 AMP（Automatic Mixed Precision）训练中的动态 Scale 更新逻辑。根据当前 scale 值、growth tracker 计数器以及是否发现 Inf/NaN，动态调整 loss scale 大小。
- 计算公式：

  $$
  \text{updated\_scale} = \begin{cases}
  \text{current\_scale} \times \text{backoff\_factor} & \text{if found\_inf} \neq 0 \\
  \text{current\_scale} \times \text{growth\_factor} & \text{if growth\_tracker + 1 = growth\_interval and new\_scale is finite} \\
  \text{current\_scale} & \text{otherwise}
  \end{cases}
  $$

  $$
  \text{updated\_growth\_tracker} = \begin{cases}
  0 & \text{if found\_inf} \neq 0 \text{ or growth triggered} \\
  \text{growth\_tracker} + 1 & \text{otherwise}
  \end{cases}
  $$

  其中：
  - $\text{current\_scale}$：当前的 loss scale 值
  - $\text{found\_inf}$：是否检测到 Inf/NaN 的标志（0 表示正常，非 0 表示发现 Inf/NaN）
  - $\text{growth\_tracker}$：连续未出现 Inf/NaN 的步数计数器
  - $\text{growth\_factor}$：scale 增长因子（通常为 2.0）
  - $\text{backoff\_factor}$：scale 回退因子（通常为 0.5）
  - $\text{growth\_interval}$：触发 scale 增长的间隔步数

- 使用场景：AMP 训练中的动态损失缩放（Dynamic Loss Scaling），用于在 FP16/BF16 混合精度训练中防止梯度下溢。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 170px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 220px">
  <col style="width: 160px">
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
      <td>current_scale</td>
      <td>输入</td>
      <td>当前的 loss scale 值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>growth_tracker</td>
      <td>输入</td>
      <td>连续未出现 Inf/NaN 的步数计数器。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>found_inf</td>
      <td>输入</td>
      <td>是否检测到 Inf/NaN 的标志，0 表示正常，非 0 表示发现 Inf/NaN。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>growth_factor</td>
      <td>属性</td>
      <td>scale 增长因子，通常设置为 2.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backoff_factor</td>
      <td>属性</td>
      <td>scale 回退因子，通常设置为 0.5。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>growth_interval</td>
      <td>属性</td>
      <td>触发 scale 增长的间隔步数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>updated_scale</td>
      <td>输出</td>
      <td>更新后的 loss scale 值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updated_growth_tracker</td>
      <td>输出</td>
      <td>更新后的 growth tracker 计数器。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- shape 约束：所有输入输出张量均为标量，shape 为 [1]。
- 数据类型约束：current_scale 与 found_inf、updated_scale 的数据类型需一致。

## 约束说明

无

