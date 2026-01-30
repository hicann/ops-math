# stateless_random_choice_with_mask

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：
    根据输入x中值为true，选择出对应索引列表，其次根据seed, offset调用philox_random生成索引列表长度个随机数，利用随机数对索引列表做洗牌算法，得到随机交换后得索引列表，返回count个。
    如果count为0，输出实际索引列表个。如果列表不足count个，填充0，补充至count个。输出mask标记该索引是否为有效索引。

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
      <td>输入x值。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>输入</td>
      <td>期望输出索引个数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>获取随机种子。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>获取值的步长。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>洗牌后的输入x值为true的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输出</td>
      <td>标记该位置的索引是否为有效索引。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

无