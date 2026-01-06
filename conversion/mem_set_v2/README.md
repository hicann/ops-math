# MemSetV2

##  产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |

## 功能说明

- 算子功能：给下游算子指定的output和workspace初始化成指定值。


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
      <td>x</td>
      <td>输入</td>
      <td>是框架传递的待初始化的Tensor。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
      BF16、FLOAT16、FLOAT、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values_int</td>
      <td>属性</td>
      <td>指定对应位置的tensor的int类型的初始值。</td>
      <td>int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>values_float</td>
      <td>属性</td>
      <td>指定对应位置的tensor的float类型的初始值。</td>
      <td>float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>是框架传递的待初始化的Tensor，本算子的输出就是输入，原地进行初始化</td>
      <td>输入Tensor相同x</td>
      <td>ND</td>
    </tr>



  </tbody></table>

## 约束说明

- 无。

## 调用说明

- 无