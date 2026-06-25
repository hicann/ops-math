# trilu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>    |    ×     |
| <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：对输入张量的最后两个维度构成的矩阵，根据`upper`属性选择保留上三角或下三角区域的元素，其余元素置零。

- 计算公式：

$$
\text{当upper = 1时（上三角模式）:} \quad y[row, col] = \begin{cases} x[row, col] & \text{if } col - row \ge diagonal \\ 0 & \text{otherwise} \end{cases}
$$

$$
\text{当upper = 0时（下三角模式）:} \quad y[row, col] = \begin{cases} x[row, col] & \text{if } col - row \le diagonal \\ 0 & \text{otherwise} \end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>待进行三角掩码计算的入参，最后两维为矩阵。</td>
      <td>FLOAT16、FLOAT、INT32、INT64、INT8、INT16、UINT8、UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入（可选）</td>
      <td>对角线偏移量，0为主对角线，正数为超对角线，负数为次对角线。默认值为0。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>upper</td>
      <td>属性</td>
      <td>三角区域标志，1表示上三角，0表示下三角。默认值为0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>三角掩码后的输出张量，shape与输入x完全一致。</td>
      <td>与输入x相同</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入输出支持任意维度（ND格式），最后两维构成矩阵。
- 不支持float64和bool类型。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_trilu.cpp">test_geir_trilu</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
